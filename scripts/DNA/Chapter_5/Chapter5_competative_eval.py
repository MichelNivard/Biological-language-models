import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
from datetime import datetime
    

# Add visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# PRivate mports:

from Chapter4_MSA_augmented_encoding_for_DNABert import tokenize_with_aux

# --------------------------------
# 11. Competative evaluations!!
# --------------------------------


# Load GPN-enhanced ModernBERT (your custom model)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_gpn = torch.load("./bert-dna-gpn/gpn_bert_model.pt") # Assuming it's already trained and loaded
tokenizer_gpn = AutoTokenizer.from_pretrained("./bert-dna-gpn")

# Load the full model
model = torch.load("./bert-dna-gpn/gpn_bert_model.pt")
model.eval()


# Load DNABert-CDS-13Species
model_name_dnabert = "MichelNivard/DNABert-CDS-13Species-v0.1"
tokenizer_dnabert = AutoTokenizer.from_pretrained(model_name_dnabert)
model_dnabert = AutoModelForMaskedLM.from_pretrained(model_name_dnabert).to(device)
model_dnabert.eval()

# Helper to get vocab mapping
id_to_token_gpn = {v: k for k, v in tokenizer_gpn.get_vocab().items()}
id_to_token_dnabert = {v: k for k, v in tokenizer_dnabert.get_vocab().items()}

def analyze_model_predictions(model_gpn, tokenizer_gpn, model_dnabert, tokenizer_dnabert, dataset, device):
    """
    Analyze predictions from both models under different scenarios and save to DataFrame:
    1. Normal inference comparing both models
    2. Predictions with missing species data (all "-")
    3. Predictions where human sequence differs from majority of species (with high conservation)
    """

    results = []


    for sequence_idx, sequence in enumerate(dataset):
        print(f"\nAnalyzing sequence {sequence_idx + 1}/{len(dataset)}")
        
        try:
            # Get base inputs
            input_ids_gpn = torch.tensor(sequence['input_ids']).unsqueeze(0).to(device)
            attention_mask_gpn = torch.tensor(sequence['attention_mask']).unsqueeze(0).to(device)
            aux_features = torch.tensor(sequence['aux_features']).unsqueeze(0).to(device)
            # Make sure its all upper case text:
            human_sequence = sequence['human_sequence'].upper()
            species_sequences = [seq.upper() for seq in sequence['species_sequences']]

            # Process DNABERT inputs
            tokens_dnabert = tokenizer_dnabert(human_sequence.upper(), padding=True, truncation=True, return_tensors="pt", max_length=512)
            input_ids_dnabert = tokens_dnabert['input_ids'].to(device)
            attention_mask_dnabert = tokens_dnabert['attention_mask'].to(device)

            # Create masked version for missing species data (all "-")
            missing_aux_features = torch.zeros_like(aux_features)
            missing_aux_features[..., 4::5] = 1  # Set all species data to "-"

            # Analyze each position in the sequence
            seq_length = min(len(human_sequence), input_ids_gpn.shape[1])
            
            for pos in range(seq_length):
                if pos >= input_ids_gpn.shape[1] - 1:  # Skip padding
                    break
                    
                # Get original tokens
                original_human_base = human_sequence[pos]
                species_bases = [seq[pos] if pos < len(seq) else "-" for seq in species_sequences]
                
                # Check if this position shows ancestral bias
                valid_bases = [b for b in species_bases if b in "ATCG"]
                if valid_bases:
                    majority_base = max(set(valid_bases), key=valid_bases.count)
                    is_ancestral_case = (original_human_base != majority_base)
                    majority_count = valid_bases.count(majority_base)
                    conservation_score = majority_count / len(valid_bases)
                else:
                    is_ancestral_case = False
                    majority_base = "-"
                    conservation_score = 0.0

                # Analyze first 500 positions and all ancestral cases
                if not (pos < 500 or is_ancestral_case):
                    continue

                # Scenario 1: Normal inference
                predictions_normal = get_predictions(
                    pos, input_ids_gpn, attention_mask_gpn, aux_features,
                    input_ids_dnabert, attention_mask_dnabert,
                    model_gpn, model_dnabert, tokenizer_gpn, tokenizer_dnabert,
                    device
                )
                
                # Scenario 2: Missing species data
                predictions_missing = get_predictions(
                    pos, input_ids_gpn, attention_mask_gpn, missing_aux_features,
                    input_ids_dnabert, attention_mask_dnabert,
                    model_gpn, model_dnabert, tokenizer_gpn, tokenizer_dnabert,
                    device
                )
                
                # Store results
                result = {
                    'sequence_idx': sequence_idx,
                    'position': pos,
                    'human_base': original_human_base,
                    'majority_species_base': majority_base,
                    'conservation_score': conservation_score,
                    'is_ancestral_case': is_ancestral_case,
                    'species_bases': ','.join(species_bases),
                }
                
                # Add normal predictions
                for i in range(4):
                    result.update({
                        f'gpn_normal_pred_{i+1}': tokenizer_gpn.decode([predictions_normal['gpn_probs'].indices[i].item()]),
                        f'gpn_normal_prob_{i+1}': torch.exp(predictions_normal['gpn_probs'].values[i]).item(),
                        f'dnabert_normal_pred_{i+1}': tokenizer_dnabert.decode([predictions_normal['dnabert_probs'].indices[i].item()]),
                        f'dnabert_normal_prob_{i+1}': torch.exp(predictions_normal['dnabert_probs'].values[i]).item(),
                    })
                
                # Add missing species predictions
                for i in range(4):
                    result.update({
                        f'gpn_missing_pred_{i+1}': tokenizer_gpn.decode([predictions_missing['gpn_probs'].indices[i].item()]),
                        f'gpn_missing_prob_{i+1}': torch.exp(predictions_missing['gpn_probs'].values[i]).item(),
                    })
                
                results.append(result)

        except Exception as e:
            print(f"Error processing sequence: {e}")
            continue

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    
    # Compare model performance - only using top prediction
    def get_accuracy(row, model_prefix):
        return row[f'{model_prefix}_pred_1'] == row['human_base']
    
    df['gpn_normal_correct'] = df.apply(lambda x: get_accuracy(x, 'gpn_normal'), axis=1)
    df['dnabert_normal_correct'] = df.apply(lambda x: get_accuracy(x, 'dnabert_normal'), axis=1)
    df['gpn_missing_correct'] = df.apply(lambda x: get_accuracy(x, 'gpn_missing'), axis=1)
    
    # Add rank analysis
    def get_correct_rank(row, model_prefix):
        for i in range(1, 5):
            if row[f'{model_prefix}_pred_{i}'] == row['human_base']:
                return i
        return 5  # Not in top 4
    
    df['gpn_normal_rank'] = df.apply(lambda x: get_correct_rank(x, 'gpn_normal'), axis=1)
    df['dnabert_normal_rank'] = df.apply(lambda x: get_correct_rank(x, 'dnabert_normal'), axis=1)
    df['gpn_missing_rank'] = df.apply(lambda x: get_correct_rank(x, 'gpn_missing'), axis=1)
    
    # Get ancestral different cases with high conservation
    ancestral_diff = df[(df['is_ancestral_case']) & (df['conservation_score'] >= 0.75)]
    
    print("\nOverall Model Performance (Top Prediction Only):")
    print(f"Total positions analyzed: {len(df)}")
    print(f"GPN accuracy (normal): {df['gpn_normal_correct'].mean():.3f}")
    print(f"DNABERT accuracy: {df['dnabert_normal_correct'].mean():.3f}")
    print(f"GPN accuracy (missing species): {df['gpn_missing_correct'].mean():.3f}")
    
    print("\nModel Performance on Highly Conserved Ancestral Different Cases (Top Prediction Only):")
    print(f"Positions where human differs from ancestral (conservation ≥ 0.75): {len(ancestral_diff)}")
    if len(ancestral_diff) > 0:
        print(f"GPN accuracy (normal): {ancestral_diff['gpn_normal_correct'].mean():.3f}")
        print(f"DNABERT accuracy: {ancestral_diff['dnabert_normal_correct'].mean():.3f}")
        print(f"GPN accuracy (missing species): {ancestral_diff['gpn_missing_correct'].mean():.3f}")
    
    print("\nOverall Rank Analysis:")
    for model in ['gpn_normal', 'dnabert_normal', 'gpn_missing']:
        rank_dist = df[f'{model}_rank'].value_counts().sort_index()
        print(f"\n{model} ranks:")
        for rank, count in rank_dist.items():
            if rank < 5:
                print(f"Rank {rank}: {count} ({count/len(df):.3f})")
            else:
                print(f"Not in top 4: {count} ({count/len(df):.3f})")
    
    print("\nRank Analysis for Highly Conserved Ancestral Different Cases:")
    if len(ancestral_diff) > 0:
        for model in ['gpn_normal', 'dnabert_normal', 'gpn_missing']:
            rank_dist = ancestral_diff[f'{model}_rank'].value_counts().sort_index()
            print(f"\n{model} ranks (ancestral different only, conservation ≥ 0.75):")
            for rank, count in rank_dist.items():
                if rank < 5:
                    print(f"Rank {rank}: {count} ({count/len(ancestral_diff):.3f})")
                else:
                    print(f"Not in top 4: {count} ({count/len(ancestral_diff):.3f})")
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"dna_model_predictions_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    return df

def get_predictions(pos, input_ids_gpn, attention_mask_gpn, aux_features,
                   input_ids_dnabert, attention_mask_dnabert,
                   model_gpn, model_dnabert, tokenizer_gpn, tokenizer_dnabert,
                   device):
    """Helper function to get predictions from both models for a specific position"""
    
    # Mask the position in both models
    masked_input_ids_gpn = input_ids_gpn.clone()
    masked_input_ids_gpn[0, pos] = tokenizer_gpn.mask_token_id
    
    masked_input_ids_dnabert = input_ids_dnabert.clone()
    masked_input_ids_dnabert[0, pos] = tokenizer_dnabert.mask_token_id
    
    # Get predictions from GPN
    with torch.no_grad():
        output_gpn = model_gpn(
            input_ids=masked_input_ids_gpn,
            attention_mask=attention_mask_gpn,
            aux_features=aux_features
        )
        logits_gpn = output_gpn.logits
        log_probs_gpn = torch.log_softmax(logits_gpn[0, pos], dim=-1)
        
        # Get predictions from DNABERT
        output_dnabert = model_dnabert(
            masked_input_ids_dnabert,
            attention_mask=attention_mask_dnabert
        )
        logits_dnabert = output_dnabert.logits
        log_probs_dnabert = torch.log_softmax(logits_dnabert[0, pos], dim=-1)
    
    # Get top predictions
    top_preds_gpn = torch.topk(log_probs_gpn, k=4)
    top_preds_dnabert = torch.topk(log_probs_dnabert, k=4)
    
    return {
        'gpn_probs': top_preds_gpn,
        'dnabert_probs': top_preds_dnabert
    }

# Update the call to use the new function
dataset_name = "MichelNivard/Human-CDS-cross-species-alligned"
dataset = load_dataset(dataset_name, split="train")
tokenized_small_dataset = dataset.select(range(30))

tokenized_dataset = tokenized_small_dataset.map(tokenize_with_aux, batched=False)


# Run analysis and get DataFrame
results_df = analyze_model_predictions(
    model_gpn,
    tokenizer_gpn,
    model_dnabert,
    tokenizer_dnabert,
    tokenized_dataset,
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
)



################## VISUALIZE RESULTS:



print("\nGenerating visualization plots...")

df = results_df

# 1. Overall Accuracy Bar Plot
plt.figure(figsize=(10, 6))
accuracies = {
    'GPN (normal)': df['gpn_normal_correct'].mean(),
    'DNABERT': df['dnabert_normal_correct'].mean(),
    'GPN (no species)': df['gpn_missing_correct'].mean()
}
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Ancestral Cases Analysis
ancestral_df = df[(df['is_ancestral_case']) & (df['conservation_score'] >= 0.75)]
if len(ancestral_df) > 0:
    plt.figure(figsize=(10, 6))
    ancestral_accuracies = {
        'GPN (normal)': ancestral_df['gpn_normal_correct'].mean(),
        'DNABERT': ancestral_df['dnabert_normal_correct'].mean(),
        'GPN (no species)': ancestral_df['gpn_missing_correct'].mean()
    }
    plt.bar(ancestral_accuracies.keys(), ancestral_accuracies.values(), color=['blue', 'green', 'red'])
    plt.title('Accuracy on Highly Conserved Ancestral Different Cases')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 3. Rank Distribution Plot
plt.figure(figsize=(12, 6))
rank_data = []
for model in ['gpn_normal', 'dnabert_normal', 'gpn_missing']:
    ranks = df[f'{model}_rank']
    model_name = {'gpn_normal': 'GPN (normal)', 
                 'dnabert_normal': 'DNABERT',
                 'gpn_missing': 'GPN (no species)'}[model]
    rank_data.extend([(model_name, rank) for rank in ranks])

rank_df = pd.DataFrame(rank_data, columns=['Model', 'Rank'])
sns.violinplot(x='Model', y='Rank', data=rank_df, palette=['blue', 'green', 'red'])
plt.title('Distribution of Correct Prediction Ranks')
plt.ylabel('Rank (1 = top prediction)')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 4. Conservation Score vs Accuracy
plt.figure(figsize=(12, 6))
conservation_bins = np.linspace(0, 1, 11)
df['conservation_bin'] = pd.cut(df['conservation_score'], bins=conservation_bins)

for model, color in zip(['gpn_normal', 'dnabert_normal', 'gpn_missing'], ['blue', 'green', 'red']):
    accuracy_by_conservation = df.groupby('conservation_bin')[f'{model}_correct'].mean()
    plt.plot(conservation_bins[:-1] + 0.05, accuracy_by_conservation, 
            marker='o', label=model.replace('_', ' ').title(), color=color)

plt.title('Model Accuracy vs Conservation Score')
plt.xlabel('Conservation Score')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
