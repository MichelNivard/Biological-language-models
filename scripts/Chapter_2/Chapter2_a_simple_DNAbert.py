import torch
import wandb
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast
from transformers import ModernBertConfig, ModernBertForMaskedLM
from transformers import  Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Initialize Weights & Biases
wandb.init(project="bert-dna", name="bert-dna-training_v2")
# --------------------------------
# 1. DNA Tokenizer with Full FASTA Nucleic Acid Code
# --------------------------------

# Define vocabulary to include all FASTA nucleotides and symbols
dna_vocab = {
    "A": 0, "T": 1, "C": 2, "G": 3, "N": 4, "U": 5, "i": 6,  # Standard bases + Inosine
    "R": 7, "Y": 8, "K": 9, "M": 10, "S": 11, "W": 12,  # Ambiguous bases
    "B": 13, "D": 14, "H": 15, "V": 16,  # More ambiguity codes
    "-": 17,  # Gap character
    "[UNK]": 18, "[PAD]": 19, "[CLS]": 20, "[SEP]": 21, "[MASK]": 22
}

# Create tokenizer
tokenizer = Tokenizer(WordLevel(vocab=dna_vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Split("", "isolated")  # Character-level splitting

# Convert to Hugging Face-compatible tokenizer
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# --------------------------------
# 2. Load and Tokenize the DNA Dataset
# --------------------------------
dataset_name = "MichelNivard/Coding_Sequences_13_Species"
dataset = load_dataset(dataset_name,split="train")
 
column_name = "sequence"

def tokenize_function(examples):
    return hf_tokenizer(examples[column_name], truncation=True, padding="max_length", max_length=512)
 
# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=[column_name])


# --------------------------------
# 3. Save & Train on This Dataset
# --------------------------------
tokenized_dataset.save_to_disk("tokenized_dna_dataset")

# --------------------------------
# 34. Define the BERT Model from Scratch
# --------------------------------
config = ModernBertConfig(
    vocab_size=len(dna_vocab),
    hidden_size=256,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=512,
    max_position_embeddings=512,
    type_vocab_size=1,
)
config.pad_token_id = dna_vocab["[PAD]"]
model = ModernBertForMaskedLM(config)

# --------------------------------
# 5. Training Configuration (Prints Loss)
# --------------------------------
training_args = TrainingArguments(
    output_dir="./bert-dna",
    overwrite_output_dir=True,
    logging_steps=1,  # Log loss every step
    save_steps=1000,
    save_total_limit=2,
    per_device_train_batch_size=32,
    num_train_epochs=2,
    learning_rate=5e-4,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="wandb",  # Disables wandb logging
)


# MLM Data Collator (Automatically Creates `labels`)
data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer,mlm=True, mlm_probability=0.05)

# --------------------------------
# 6. Train the Model
# --------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=hf_tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Save the final model and tokenizer
trainer.save_model("./bert-dna")
hf_tokenizer.save_pretrained("./bert-dna")

print("🎉 Training complete! Model saved to ./bert-dna")


