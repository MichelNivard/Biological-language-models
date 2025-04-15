# === contact_map_extraction.py ===

import os
import numpy as np
import sidechainnet as scn
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# === PARAMETERS ===
output_dir = "contact_maps_all"
os.makedirs(output_dir, exist_ok=True)

def make_contact_map(coords, protein_id, binary=False, threshold=8.0):
    ca_coords = coords[:, 0, :]  # shape (L, 3)

    n_residues = ca_coords.shape[0]
    n_missing = np.isnan(ca_coords).any(axis=1).sum()
    frac_missing = n_missing / n_residues

    if frac_missing > 0.10:
        print(f"{protein_id} skipped: {n_missing}/{n_residues} Cα coords missing ({frac_missing:.1%})")
        return None
    if np.unique(ca_coords, axis=0).shape[0] <= 1:
        print(f"{protein_id} skipped: collapsed structure (identical Cα)")
        return None
    if n_residues < 10:
        print(f"{protein_id} skipped: too short ({n_residues} residues)")
        return None

    ca_coords -= ca_coords[0]

    dists = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)

    if binary:
        return (dists < threshold).astype(np.uint8) * 255
    else:
        levels = [0,5,10,15,20,25,30,35, 40,50, 60,70, 80,90, 100,110, 120,130, 140,150, 160,170, 180,190,200,210,220,230,240]
        contact_map = np.full_like(dists, 255, dtype=np.uint8)
        mask = dists < (2)
        contact_map[mask] = levels[0]
        for i in range(1, 29):
            lower = 1 + i
            upper = lower + 1
            mask = (dists >= lower) & (dists < upper)
            contact_map[mask] = levels[i]
        return contact_map

def crop_contact_map(contact_map, crop_size=128, pad_value=255, min_size=20):
    h, w = contact_map.shape
    if h < min_size or w < min_size:
        raise ValueError(f"Contact map too small: {h}x{w} (min required: {min_size})")
    canvas = np.full((crop_size, crop_size), pad_value, dtype=np.uint8)
    crop_h = min(h, crop_size)
    crop_w = min(w, crop_size)
    canvas[:crop_h, :crop_w] = contact_map[:crop_h, :crop_w]
    return Image.fromarray(canvas)

def main():
    casp_versions = [12]
    valid_ids = []

    for casp_version in casp_versions:
        print(f"🔄 Loading SideChainNet CASP{casp_version}...")
        data = scn.load(casp_version=casp_version, casp_thinning=70)

        print(f"🖼️ Generating and saving contact maps for CASP{casp_version}...")
        for sample in tqdm(data):
            try:
                protein_id = sample.id
                coords = sample.coords
                if coords is None or coords.shape[0] < 2:
                    continue

                distance_map = make_contact_map(coords, protein_id, binary=False)
                if distance_map is None or np.all(distance_map == 255):
                    continue

                img = crop_contact_map(distance_map, crop_size=128, pad_value=255)
                img.save(os.path.join(output_dir, f"{protein_id}.jpg"), format="JPEG")

                valid_ids.append(protein_id)

            except Exception as e:
                print(f"❌ Skipping {sample.id} due to error: {e}")

    with open("valid_proteins.txt", "w") as f:
        for pid in valid_ids:
            f.write(f"{pid}\n")

    print(f"✅ Contact map extraction complete. {len(valid_ids)} proteins saved.")


if __name__ == "__main__":
    main()





# === embedding_extraction.py ===

import os
import torch
import numpy as np
import sidechainnet as scn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# === PARAMETERS ===
model_name = "facebook/esm2_t33_650M_UR50D"
output_dir = "embeddings_all"
valid_ids_path = "valid_proteins.txt"
os.makedirs(output_dir, exist_ok=True)

# === Load Model ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device).eval()
# === embedding_extraction.py ===

os.makedirs(output_dir, exist_ok=True)

# === Load valid protein IDs from contact map step ===
with open(valid_ids_path, "r") as f:
    valid_ids = set(line.strip() for line in f if line.strip())


def get_embeddings(sequence):
    with torch.no_grad():
        inputs = tokenizer(sequence, return_tensors="pt", truncation=True, padding=True,max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        seq_emb = outputs.last_hidden_state[0].cpu().numpy()  # (L, D)
        seq_len, emb_dim = seq_emb.shape

        # ✅ Pad to 128 if needed
        if seq_len < 128:
            pad = np.zeros((128 - seq_len, emb_dim), dtype=np.float32)
            seq_emb = np.vstack([seq_emb, pad])
        else:
            seq_emb = seq_emb[:128]

        return seq_emb  # shape: (128, D)

def main():
    casp_versions = [12]
    print(f"🔄 Loading SideChainNet CASP datasets: {casp_versions}...")

    # Combine all samples
    all_samples = []
    for casp_version in casp_versions:
        data = scn.load(casp_version=casp_version, casp_thinning=70)
        all_samples.extend(data)

    print("🧬 Extracting embeddings only for valid proteins...")
    for sample in tqdm(all_samples):
        protein_id = sample.id
        if protein_id not in valid_ids:
            continue

        try:
            sequence = sample.sequence
            if sequence is None or len(sequence) < 10:
                continue
            cropped_sequence = sequence[:128]
            emb = get_embeddings(cropped_sequence)
            np.save(os.path.join(output_dir, f"{protein_id}.npy"), emb)
        except Exception as e:
            print(f"❌ Skipping {protein_id} due to error: {e}")

    print("✅ Embedding extraction complete.")

if __name__ == "__main__":
    main()
