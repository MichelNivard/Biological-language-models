#####################################
# Chapter 3: Evaluate DNABert
#####################################

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd

# Load model & tokenizer
model_name = "MichelNivard/DNABert-CDS-13Species-v0.1"  # Replace if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Maximum context length — BERT's trained context window, minus 1 for python index?
MAX_CONTEXT_LENGTH = 511 


#####################################
# Chapter 3: Define usefull functions
#####################################


# A function to compute the log likelihood of a sequence given the model

def compute_log_likelihood(sequence, tokenizer, model):
    """Compute pseudo-log-likelihood (PLL) for the first 512 bases."""
    tokens = tokenizer(sequence, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    log_likelihood = 0.0
    seq_len = input_ids.shape[1]   # Exclude [CLS] and [SEP]

    with torch.no_grad():
        for i in range(1, seq_len + 1):
            masked_input = input_ids.clone()
            masked_input[0, i] = tokenizer.mask_token_id

            outputs = model(masked_input, attention_mask=attention_mask)
            logits = outputs.logits

            true_token_id = input_ids[0, i]
            log_probs = torch.log_softmax(logits[0, i], dim=-1)
            log_likelihood += log_probs[true_token_id].item()

    return log_likelihood


# A function to compute the log likelihood ratio of the refernece, or wild type versus a mutant base 


def compute_mutant_log_likelihood_ratio(wild_type, mutant, position, tokenizer, model):
    """Compare wild type and mutant likelihood at a single position (within 512 bases)."""
    assert len(wild_type) == len(mutant), "Wild type and mutant must have the same length"
    assert wild_type[position] != mutant[position], f"No mutation detected at position {position + 1}"

    tokens = tokenizer(wild_type[:MAX_CONTEXT_LENGTH], return_tensors='pt', add_special_tokens=True)
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    mask_position = position  # Shift for [CLS] token

    masked_input = input_ids.clone()
    masked_input[0, mask_position] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(masked_input, attention_mask=attention_mask)
        logits = outputs.logits

        log_probs = torch.log_softmax(logits[0, mask_position], dim=-1)

    wild_base_id = tokenizer.convert_tokens_to_ids(wild_type[position])
    mutant_base_id = tokenizer.convert_tokens_to_ids(mutant[position])

    log_prob_wild = log_probs[wild_base_id].item()
    log_prob_mutant = log_probs[mutant_base_id].item()

    # Convert to regular probability
    prob_wild = np.exp(log_prob_wild)

    llr = log_prob_wild - log_prob_mutant

    return llr, log_prob_wild, prob_wild



#####################################
# Chapter 3: Run the evaluations
#####################################



# Load dataset directly from Hugging Face dataset repo
dataset_url = "https://huggingface.co/datasets/MichelNivard/DRD2-mutations/raw/main/DRD2_mutations.csv"
df = pd.read_csv(dataset_url)

# Find wild-type sequence
wild_type_row = df[df['mutation_type'] == 'wildtype'].iloc[0]
wild_type_sequence = wild_type_row['sequence'][:MAX_CONTEXT_LENGTH]

results = []

# Process all sequences
for idx, row in df.iterrows():
    sequence = row['sequence'][:MAX_CONTEXT_LENGTH]
    mutation_type = row['mutation_type']
    mutation_position = row['mutation_position'] - 1  # Convert 1-based to 0-based

    # Skip mutants where the mutation position is beyond 512 bases
    if mutation_type != 'wildtype' and mutation_position >= MAX_CONTEXT_LENGTH:
        continue

    print(idx)

    llr = None
    log_prob_wild = None
    prob_wild = None

    if mutation_type != 'wildtype':
        llr, log_prob_wild, prob_wild = compute_mutant_log_likelihood_ratio(
            wild_type_sequence, sequence, int(mutation_position), tokenizer, model
        )

    # append results for each mutation:
    results.append({
        'sequence': sequence,
        'mutation_type': mutation_type,
        'pll': 0,
        'llr': llr,
        'wildtype_log_prob': log_prob_wild,
        'wildtype_prob': prob_wild,
        'mutation_position': mutation_position + 1
    })



# Convert to DataFrame for saving or inspection
results_df = pd.DataFrame(results)

# Save or print results
print(results_df)

# Optionally, save to CSV
results_df.to_csv("sequence_log_likelihoods.csv", index=False)


#####################################
# Chapter 3: Visualize the Evaluations
#####################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Filter to only mutations (skip wildtype which has no llr)
plot_df = results_df[results_df['mutation_type'].isin(['synonymous', 'missense'])].copy()


#####################################
# Violin plot
#####################################


# Plot violin plot comparing synonymous vs missense log-likelihood ratios
plt.figure(figsize=(12, 6))
sns.violinplot(x='mutation_type', y='llr', data=plot_df, palette="Set2", inner="box")
plt.title('Log Likelihood Ratio Distribution: Synonymous vs Missense')
plt.ylabel('Log Likelihood Ratio (LLR)')
plt.xlabel('Mutation Type')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

#####################################
# Side by side histogram
#####################################


# Optional: Side-by-side histograms
plt.figure(figsize=(12, 6))

# Synonymous
plt.subplot(1, 2, 1)
sns.histplot(plot_df[plot_df['mutation_type'] == 'synonymous']['llr'], bins=30, kde=True, color='green')
plt.title('Synonymous Mutations')
plt.xlabel('Log Likelihood Ratio (LLR)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.5)

# Missense
plt.subplot(1, 2, 2)
sns.histplot(plot_df[plot_df['mutation_type'] == 'missense']['llr'], bins=30, kde=True, color='orange')
plt.title('Missense Mutations')
plt.xlabel('Log Likelihood Ratio (LLR)')
plt.ylabel('Count')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

#####################################
# Plot likelihoods along the gene split by missense vs synonimous
#####################################


# Filter to only mutations
plot_df = results_df[results_df['mutation_type'].isin(['synonymous', 'missense'])].copy()

# Sort from high to low LLR (positive to negative)
plot_df = plot_df.sort_values('llr', ascending=False)

# Optional: Clip LLR to avoid excessive sizes
plot_df['size'] = plot_df['llr'].clip(-5, 5)  # LLRs smaller than -5 get maximum size

# Scatter plot with enhanced size scaling
plt.figure(figsize=(14, 5))
sns.scatterplot(
    x='mutation_position', 
    y='llr', 
    hue='mutation_type', 
    size='size',  # Use clipped size column
    sizes=(20, 200),  # Bigger range for better visibility
    alpha=0.7, 
    palette={'synonymous': 'green', 'missense': 'orange'},
    data=plot_df
)
plt.axhline(0, color='gray', linestyle='--', label='Neutral LLR')
plt.title('Mutation Log Likelihood Ratio (LLR) Along DRD2 Gene')
plt.xlabel('Position in Gene')
plt.ylabel('Log Likelihood Ratio (LLR)')
plt.legend(title='Mutation Type', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


#####################################
# Plot per base probability
#####################################

# Calculate the mean wildtype probability
mean_prob = results_df['wildtype_prob'].mean()

# Create the plot with the mean probability in the title
plt.figure(figsize=(12, 6))
plt.plot(results_df['mutation_position'], results_df['wildtype_prob'], marker='o', linestyle='-', color='b')
plt.title(f'Per-base Probability of Wild-type Sequence (Mean Probability: {mean_prob:.3f})')
plt.xlabel('Position in Sequence')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

#####################################
# Plot probabilities for A,T,C,G bases seperately
#####################################



# Ensure results_df has a column with the actual wild-type base (you may have already added this)
# If not, you can extract it from the wild-type sequence when you build the DataFrame.
# Example (inside your main processing loop):
# wildtype_base = wild_type_sequence[mutation_position]
# results.append({..., 'wildtype_base': wildtype_base})

# Group and compute mean probability per base
mean_probs_by_base = results_df.groupby('wildtype_base')['wildtype_prob'].mean().reset_index()

# Rename columns for clarity
mean_probs_by_base.columns = ['Base', 'Mean Probability']

# Sort for prettier table (optional)
mean_probs_by_base = mean_probs_by_base.sort_values('Base')

# Show table
print(mean_probs_by_base)

# Optionally, plot it
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.bar(mean_probs_by_base['Base'], mean_probs_by_base['Mean Probability'], color=['blue', 'orange', 'green', 'red'])
plt.title('Average Wild-type Base Probability by Nucleotide')
plt.ylabel('Mean Probability')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
