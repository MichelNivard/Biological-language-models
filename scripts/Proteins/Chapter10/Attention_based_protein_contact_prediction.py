import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from Bio.PDB import PDBParser
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import math

# Load model and tokenizer
model_name = "facebook/esm2_t33_650M_UR50D"
config = AutoConfig.from_pretrained(model_name, output_attentions=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example protein sequences and corresponding PDB files
sequences = [
    "MMELENIVANTVLLKAREGGGGNRKGKSKKWRQMLQFPHISQCEELRLSLERDYHSLCERQPIGRLLFREFCATRPELSRCVAFLDGVAEYEVTPDDKRKACGRQLTQNFLSHTGPDLIPEVPRQLVTNCTQRLEQGPCKDLFQELTRLTHEYLSVAPFADYLDSIYFNRFLQWKWLERQPVTKNTFRQYRVLGKGGFGEVCACQVRATGKMYACKKLEKKRIKKRKGEAMALNEKQILEKVNSRFVVSLAYAYETKDALCLVLTLMNGGDLKFHIYHMGQAGFPEARAVFYAAEICCGLEDLHRERIVYRDLKPENILLDDHGHIRISDLGLAVHVPEGQTIKGRVGTVGYMAPEVVKNERYTFSPDWWALGCLLYEMIAGQSPFQQRKKKIKREEVERLVKEVPEEYSERFSPQARSLCSQLLCKDPAERLGCRGGSAREVKEHPLFKKLNFKRLGAGMLEPPFKPDPQAIYCKDVLDIEQFSTVKGVELEPTDQDFYQKFATGSVPIPWQNEMVETECFQELNVFGLDGSVPPDLDWKGQPPAPPKKGLLQRLFSRQDCCGNCSDSEEELPTRL",
    "MESSPIPQSSGNSSTLGRVPQTPGPSTASGVPEVGLRDVASESVALFFMLLLDLTAVAGNAAVMAVIAKTPALRKFVFVFHLCLVDLLAALTLMPLAMLSSSALFDHALFGEVACRLYLFLSVCFVSLAILSVSAINVERYYYVVHPMRYEVRMTLGLVASVLVGVWVKALAMASVPVLGRVSWEEGAPSVPPGCSLQWSHSAYCQLFVVVFAVLYFLLPLLLILVVYCSMFRVARVAAMQHGPLPTWMETPRQRSESLSSRSTMVTSSGAPQTTPHRTFGGGKAAVVLLAVGGQFLLCWLPYFSFHLYVALSAQPISTGQVESVVTWIGYFCFTSNPFFYGCLNRQIRGELSKQFVCFFKPAPEEELRLPSREGSIEENFLQFLQGTGCPSESWVSRPLPSPKQEPPAVDFRIPGQIAEETSEFLEQQLTSDIIMSDSYLRPAASPRLES",
    "MDIQMANNFTPPSATPQGNDCDLYAHHSTARIVMPLHYSLVFIIGLVGNLLALVVIVQNRKKINSTTLYSTNLVISDILFTTALPTRIAYYAMGFDWRIGDALCRITALVFYINTYAGVNFMTCLSIDRFIAVVHPLRYNKIKRIEHAKGVCIFVWILVFAQTLPLLINPMSKQEAERITCMEYPNFEETKSLPWILLGACFIGYVLPLIIILICYSQICCKLFRTAKQNPLTEKSGVNKKALNTIILIIVVFVLCFTPYHVAIIQHMIKKLRFSNFLECSQRHSFQISLHFTVCLMNFNCCMDPFIYFFACKGYKRKVMRMLKRQVSVSISSAVKSAPEENSREMTETQMMIHSKSSNGK",
    "MGFNLTLAKLPNNELHGQESHNSGNRSDGPGKNTTLHNEFDTIVLPVLYLIIFVASILLNGLAVWIFFHIRNKTSFIFYLKNIVVADLIMTLTFPFRIVHDAGFGPWYFKFILCRYTSVLFYANMYTSIVFLGLISIDRYLKVVKPFGDSRMYSITFTKVLSVCVWVIMAVLSLPNIILTNGQPTEDNIHDCSKLKSPLGVKWHTAVTYVNSCLFVAVLVILIGCYIAISRYIHKSSRQFISQSSRKRKHNQSIRVVVAVFFTCFLPYHLCRIPFTFSHLDRLLDESAQKILYYCKEITLFLSACNVCLDPIIYFFMCRSFSRRLFKKSNIRTRSESIRSLQSVRRSEVRIYYDYTDV",
    "MTNSSFFCPVYKDLEPFTYFFYLVFLVGIIGSCFATWAFIQKNTNHRCVSIYLINLLTADFLLTLALPVKIVVDLGVAPWKLKIFHCQVTACLIYINMYLSIIFLAFVSIDRCLQLTHSCKIYRIQEPGFAKMISTVVWLMVLLIMVPNMMIPIKDIKEKSNVGCMEFKKEFGRNWHLLTNFICVAIFLNFSAIILISNCLVIRQLYRNKDNENYPNVKKALINILLVTTGYIICFVPYHIVRIPYTLSQTEVITDCSTRISLFKAKEATLLLAVSNLCFDPILYYHLSKAFRSKVTETFASPKETKAQKEKLRCENNA"]

pdb_filenames = ["AF-P43250-F1-model_v4.pdb", "AF-Q9BZJ8-F1-model_v4.pdb","AF-P32249-F1-model_v4.pdb","AF-Q9BY21-F1-model_v4.pdb","AF-O14626-F1-model_v4.pdb"]
chain_id = "A"


# calcualte sigmoid: 
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Function to generate true contact maps from PDB
def generate_contact_map(pdb_filename, chain_id, threshold=12.0):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_filename)
    chain = structure[0][chain_id]
    
    residues = [res for res in chain if "CA" in res]  # Ensure we only use residues with CA atoms
    seq_len = len(residues)
    
    dist_matrix = np.zeros((seq_len, seq_len))
    for i, res_one in enumerate(residues):
        for j, res_two in enumerate(residues):
            diff_vector = res_one["CA"].coord - res_two["CA"].coord
            dist_matrix[i, j] = np.sqrt(np.sum(diff_vector * diff_vector))
    
    contact_map = dist_matrix < threshold
    return contact_map.astype(int)

# Function to extract attention matrices
def extract_attention_matrices(sequence):
    inputs = tokenizer(sequence, return_tensors='pt')
    outputs = model(**inputs)
    attentions = outputs.attentions  # Tuple (num_layers, batch_size, num_heads, seq_len, seq_len)
    stacked_attentions = torch.cat([attn.squeeze(0) for attn in attentions], dim=0)
    return stacked_attentions.detach().numpy()

# Prepare training data and train logistic regression models
all_coefs = []
all_intercepts = []

for seq, pdb in zip(sequences[:3], pdb_filenames[:3]):
    true_contact_map = generate_contact_map(pdb, chain_id)
    attention_matrices = extract_attention_matrices(seq)
    
    # Prepare features
    seq_len = true_contact_map.shape[0]
    X = np.zeros((seq_len * (seq_len - 1) // 2, attention_matrices.shape[0]))
    y = np.zeros((seq_len * (seq_len - 1) // 2,))

    index = 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            feature_vector = attention_matrices[:, i, j]
            if j - i >= 1:  # Ignore near-diagonal contacts
                X[index] = feature_vector
                y[index] = true_contact_map[i, j]
            index += 1
    
    # Train logistic regression model
    clf = LogisticRegression()
    clf.fit(X, y)

    # Store learned coefficients
    all_coefs.append(clf.coef_)
    all_intercepts.append(clf.intercept_)

# Compute the average coefficients
avg_coefs = np.mean(np.array(all_coefs), axis=0)
avg_intercept = np.mean(np.array(all_intercepts), axis=0)

# Function to predict contact map using averaged coefficients
def predict_contact_map(attention_matrices, avg_coefs, avg_intercept):
    seq_len = attention_matrices.shape[1]
    predicted_contact_map = np.zeros((seq_len, seq_len))

    index = 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            feature_vector = attention_matrices[:, i, j]
            logit = np.dot(feature_vector, avg_coefs.T) + avg_intercept
            predicted_contact_map[i, j] = np.sqrt(sigmoid(logit))
            predicted_contact_map[j, i] = predicted_contact_map[i, j]
            index += 1
    
    return predicted_contact_map

def plot_combined_contact_map(true_contact_map, predicted_contact_map, title="True vs Predicted Contact Map"):
    """
    Plots a combined contact map where:
    - The upper triangle (above the diagonal) contains predicted contact probabilities.
    - The lower triangle (below the diagonal) contains true contact labels.

    Parameters:
    - true_contact_map: Ground truth contact map (seq_len, seq_len)
    - predicted_contact_map: Predicted contact map (seq_len, seq_len, probabilities or binary)
    - title: Title of the plot
    """
    seq_len = true_contact_map.shape[0]
    combined_map = np.zeros((seq_len, seq_len))

    # Fill upper triangle with predicted contacts
    upper_indices = np.triu_indices(seq_len, k=1)
    combined_map[upper_indices] = predicted_contact_map[upper_indices]

    # Fill lower triangle with true contacts
    lower_indices = np.tril_indices(seq_len, k=-1)
    combined_map[lower_indices] = true_contact_map[lower_indices]

    # Plot the combined contact map
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_map, cmap="viridis", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Residue Position")
    plt.ylabel("Residue Position")
    plt.show()


# Test prediction on a new sequence
test_seq = sequences[4]  # Replace with actual test sequence
test_attn = extract_attention_matrices(test_seq)
pred_contact_map = predict_contact_map(test_attn, avg_coefs, avg_intercept)

# Plot predicted contact map
def plot_matrix(matrix, title="Matrix", cmap="viridis"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap=cmap)
    plt.title(title)
    plt.xlabel("Residue Position")
    plt.ylabel("Residue Position")
    plt.show()

# Plot prediction
plot_matrix(pred_contact_map, title="Predicted Contact Map (Averaged Model)", cmap="Blues")

# Example usage
true_contact_map = generate_contact_map(pdb_filenames[4], chain_id)
# Plot true contact map

plot_matrix(true_contact_map, title="Protein Contact Map (Averaged Model)", cmap="Blues")


plot_combined_contact_map(true_contact_map, pred_contact_map, title="True vs Predicted Contact Map")
