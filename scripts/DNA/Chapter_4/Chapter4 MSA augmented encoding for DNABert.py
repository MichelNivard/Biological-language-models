
"""
This script is a **didactic reimplementation** inspired by the work of **Benegas et al. (2024)** published in *Nature Biotechnology*:
"A DNA language model based on multispecies alignment predicts the effects of genome-wide variants"
(https://doi.org/10.1038/s41587-024-02511-w).

The original GPN-MSA model introduced a powerful approach to variant effect prediction (VEP) by combining:
- A masked language modeling (MLM) objective.
- Multispecies sequence alignments (MSA) as auxiliary features.
- Transformer-based modeling with contextual embeddings informed by both sequence context and evolutionary history across species.

The key innovation in GPN-MSA was the explicit modeling of **aligned species sequences** alongside the human reference genome. 
At each genomic position, the nucleotide in the human genome is jointly encoded with a **concatenated one-hot encoding** 
of the corresponding nucleotides in other species at the same position. This allows the model to not only learn within-sequence
context (like traditional language models) but also between-species evolutionary constraints, much like conservation scores 
such as phyloP or phastCons, but in a flexible, high-dimensional representation.

This reimplementation adapts the **ModernBERT** architecture to accept both:
- The primary human sequence (one-hot encoded).
- Aligned species sequences (one-hot encoded, concatenated as auxiliary features).

The purpose of this script is educational — to demonstrate how to adapt Hugging Face's `Trainer` infrastructure to 
work with GPN-like auxiliary features in a genomics context.

Key differences from the original GPN-MSA:
- This implementation is **simplified** to operate on pre-tokenized sequences.
- Sequence length, auxiliary encoding, and masking strategy can be tuned by the user.
- It does not include phastCons/phyloP-based weighting of the loss function, which was a key part of the original paper.
- We use ModernBERT (a customizable transformer backbone) rather than RoFormer, although the approaches are largely compatible.

Original GPN-MSA Authors:
- Gonzalo Benegas, Carlos Albors, Alan J. Aw, Chengzhong Ye, and Yun S. Song

Reference:
Benegas et al., Nature Biotechnology, 2024. DOI: https://doi.org/10.1038/s41587-024-02511-w
"""


import torch
import wandb
import numpy as np
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, ModernBertConfig, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import ModernBertForMaskedLM
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

# --------------------------------
# 1. Initialize Weights & Biases
# --------------------------------
wandb.init(project="bert-dna-gpn", name="bert-dna-gpn-training")

# --------------------------------
# 2. Define DNA Tokenizer (Main Sequence Only)
# --------------------------------
dna_vocab = {
    "A": 0, "T": 1, "C": 2, "G": 3, "-": 4,
    "[UNK]": 5, "[PAD]": 6, "[CLS]": 7, "[SEP]": 8, "[MASK]": 9
}

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(WordLevel(vocab=dna_vocab, unk_token="[UNK]")),
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# --------------------------------
# 3. Preprocessing: Sequence Cleaner
# --------------------------------
def clean_sequence(sequence):
    """
    Maps all non-ATGC characters to '-'.
    """
    valid_bases = set("ATGC")
    return "".join(base if base in valid_bases else "-" for base in sequence)

# --------------------------------
# 4. Load and Tokenize Dataset (with species sequences)
# --------------------------------
dataset_name = "MichelNivard/Coding_Sequences_13_Species"
dataset = load_dataset(dataset_name, split="train")

def one_hot_encode_base(base):
    """One-hot encodes A, T, G, C, - (5 bases total)."""
    base_to_idx = {"A": 0, "T": 1, "C": 2, "G": 3, "-": 4}
    one_hot = np.zeros(5, dtype=np.float32)
    if base in base_to_idx:
        one_hot[base_to_idx[base]] = 1.0
    return one_hot

def tokenize_with_aux(examples):
    # Clean human and species sequences to A, T, G, C, -
    human_seq = clean_sequence(examples["sequence"])
    species_seqs = [clean_sequence(seq) for seq in examples["species_sequences"]]

    # Tokenize human sequence
    tokens = tokenizer(human_seq, truncation=True, padding="max_length", max_length=512)
    input_ids = tokens["input_ids"]

    # Process species sequences into concatenated one-hot vectors (aux features)
    seq_len = len(input_ids)
    num_species = len(species_seqs)

    aux_features = np.zeros((seq_len, num_species * 5), dtype=np.float32)

    for pos in range(seq_len):
        if pos >= len(human_seq):  # Handle padding case
            break
        for species_idx, species_seq in enumerate(species_seqs):
            if pos < len(species_seq):
                aux_features[pos, species_idx * 5:(species_idx + 1) * 5] = one_hot_encode_base(species_seq[pos])

    tokens["aux_features"] = aux_features.tolist()
    return tokens

tokenized_dataset = dataset.map(tokenize_with_aux, batched=False)

# --------------------------------
# 5. Define ModernBERT with GPNEmbedding
# --------------------------------
class GPNEmbedding(nn.Module):
    def __init__(self, config, n_species):
        super().__init__()
        self.config = config
        self.n_species = n_species
        self.vocab_size = 5  # A, T, G, C, -
        self.species_feature_size = self.n_species * self.vocab_size

    def forward(self, input_ids, aux_features):
        # One-hot encode human bases
        one_hot = F.one_hot(input_ids, num_classes=self.config.vocab_size).float()

        # Combine human one-hot + species auxiliary features
        combined = torch.cat([one_hot[..., :self.vocab_size], aux_features], dim=-1)

        # Pad to match hidden size (ModernBERT hidden_size)
        if combined.shape[-1] < self.config.hidden_size:
            pad = self.config.hidden_size - combined.shape[-1]
            combined = F.pad(combined, (0, pad))

        return combined

class ModernBertForMaskedLMWithGPN(ModernBertForMaskedLM):
    def __init__(self, config, n_species):
        super().__init__(config)
        self.gpn_embedding = GPNEmbedding(config, n_species)

    def forward(self, input_ids=None, aux_features=None, **kwargs):
        embeddings = self.gpn_embedding(input_ids, aux_features)
        outputs = self.bert.encoder(inputs_embeds=embeddings, **kwargs)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        loss = None
        labels = kwargs.get("labels", None)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": prediction_scores}

# --------------------------------
# 6. Training Configuration
# --------------------------------
config = ModernBertConfig(
    vocab_size=len(dna_vocab),  # using reduced ATGC- vocab
    hidden_size=256,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=512,
    max_position_embeddings=512,
    type_vocab_size=1,
    pad_token_id=dna_vocab["[PAD]"]
)

n_species = 12  # Assuming 12 species are aligned
model = ModernBertForMaskedLMWithGPN(config, n_species=n_species)

training_args = TrainingArguments(
    output_dir="./bert-dna-gpn",
    overwrite_output_dir=True,
    logging_steps=1,
    save_steps=1000,
    save_total_limit=2,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    learning_rate=5e-4,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="wandb",
)

# --------------------------------
# 7. Custom Data Collator
# --------------------------------
def data_collator(features):
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
        "aux_features": torch.tensor([f["aux_features"] for f in features], dtype=torch.float),
    }

# --------------------------------
# 8. Training
# --------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# --------------------------------
# 9. Save Model & Tokenizer
# --------------------------------
trainer.save_model("./bert-dna-gpn")
tokenizer.save_pretrained("./bert-dna-gpn")

print("🎉 Training complete with GPN-enhanced ModernBERT!")
