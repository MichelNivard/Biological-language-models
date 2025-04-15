import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import sidechainnet as scn
from PIL import Image

# === PARAMETERS ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_residues = 128
epochs = 1500
restarts = 6
softness_values = [.25,.5, .75, 1]
lr_values = [0.5, 0.25]

# === Load SideChainNet once ===
scn_data = scn.load(casp_version=12, casp_thinning=70)

# === Load JPEG ===
def load_contact_map_image(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)
    return torch.tensor(arr, dtype=torch.float32, device=device)

def get_random_100_proteins(folder="contact_maps_all"):
    jpg_files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    sample_size = min(500, len(jpg_files))
    sampled = random.sample(jpg_files, sample_size)
    return [fname.replace(".jpg", "") for fname in sampled]

def classical_mds(D, output_dim=3):
    n = D.shape[0]
    D2 = D ** 2
    H = torch.eye(n, device=D.device) - torch.ones(n, n, device=D.device) / n
    B = -0.5 * H @ D2 @ H
    eigvals, eigvecs = torch.linalg.eigh(B)
    idx = eigvals.argsort(descending=True)[:output_dim]
    L = torch.diag(torch.sqrt(eigvals[idx].clamp(min=0)))
    V = eigvecs[:, idx]
    X = V @ L
    return X

class DistanceToPixelMap(nn.Module):
    def __init__(self, bin_centers, levels, init_softness=1.5):
        super().__init__()
        self.register_buffer("bin_centers", torch.tensor(bin_centers, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32).view(1, 1, -1))
        self.softness = nn.Parameter(torch.tensor(init_softness, dtype=torch.float32))

    def forward(self, dists):
        d = dists.unsqueeze(-1)
        c = self.bin_centers
        s = torch.clamp(self.softness, min=0.1)
        scores = -((d - c) ** 2) / s**2
        weights = F.softmax(scores, dim=-1)
        return (weights * self.levels).sum(dim=-1)

def pairwise_distances(X):
    diff = X[:, None, :] - X[None, :, :]
    return torch.norm(diff, dim=-1)

def contact_map_loss(pred_coords, target_img, map_fn):
    dists = pairwise_distances(pred_coords)
    pred_map = map_fn(dists)
    mse = F.mse_loss(pred_map, target_img)
    coord_penalty = (pred_coords ** 2).mean()
    return mse + 0.001 * coord_penalty

def optimize_coords(target_img, steps=100, restart_idx=0, lr=1.0, init_softness=2.5):
    map_fn = DistanceToPixelMap(
        bin_centers=[1,2.5,3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5,23.5,24.5,25.5,26.5,27.5,28.5,29.5,30.5],
        levels = [0,5,10,15,20,25,30,35, 40, 50, 60,70, 80,90, 100, 110, 120,130, 140,150, 160,170, 180,190,200,210,220,230, 240, 255],
        init_softness=init_softness
    ).to(device)

    with torch.no_grad():
        D = target_img.clone()
        D[D >= 255] = 255.0
        coords = classical_mds(D)
        jitter = 0 + restart_idx * 2
        coords += torch.randn_like(coords) * jitter

    coords.requires_grad_()
    optimizer = torch.optim.Adam([coords], lr=lr, weight_decay=0.01)

    for step in range(steps):
        optimizer.zero_grad()
        loss = contact_map_loss(coords, target_img, map_fn)
        loss.backward()
        optimizer.step()

    return coords.detach(), loss.item()

def align_coords(ref, target):
    ref_c = ref - ref.mean(0)
    target_c = target - target.mean(0)
    C = target_c.T @ ref_c
    U, _, Vt = torch.linalg.svd(C)
    R = U @ Vt
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    aligned = target_c @ R
    rmsd = torch.sqrt(torch.mean((aligned - ref_c) ** 2)).item()
    return aligned + ref.mean(0), rmsd

def load_true_coords(pid):
    for sample in scn_data:
        if sample.id == pid:
            coords = sample.coords[:, 0, :]  # shape: (L, 3)
            if np.isnan(coords).any():
                raise ValueError(f"{pid} contains NaN coordinates")
            if coords.shape[0] < 128:
                raise ValueError(f"{pid} has fewer than 128 residues")
            if coords.shape[0] > 256:
                raise ValueError(f"{pid} has more than 256 residues")
            if coords.shape[0] < 128:
                pad = np.zeros((128 - coords.shape[0], 3))
                coords = np.vstack([coords, pad])
            return torch.tensor(coords[:128], dtype=torch.float32, device=device)
    raise ValueError(f"Protein {pid} not found.")


# === MAIN ===
if __name__ == "__main__":
    sampled_proteins = get_random_100_proteins()
    print(f"🎯 Sampled {len(sampled_proteins)} proteins. Filtering invalid ones...")

    valid_proteins = []
    protein_coords = {}

    # Pre-filter proteins with valid coords
    for pid in sampled_proteins:
        try:
            coords = load_true_coords(pid)
            if torch.isnan(coords).any():
                print(f"⚠️ Skipping {pid} due to NaNs")
                continue
            valid_proteins.append(pid)
            protein_coords[pid] = coords
        except Exception as e:
            print(f"⚠️ Skipping {pid}: {e}")

    print(f"✅ {len(valid_proteins)} valid proteins found.\n")

    results = []

    for protein_id in valid_proteins:
        try:
            print(f"\n📥 Processing {protein_id}...")
            image_path = f"contact_maps_all/{protein_id}.jpg"
            target_img = load_contact_map_image(image_path)
            true_coords = protein_coords[protein_id]

            best_coords, best_loss = None, float("inf")
            best_cfg = None

            for softness in softness_values:
                for lr in lr_values:
                    for restart in range(restarts):
                        coords, loss = optimize_coords(
                            target_img,
                            steps=epochs,
                            restart_idx=restart,
                            lr=lr,
                            init_softness=softness
                        )
                        if loss < best_loss:
                            best_loss = loss
                            best_coords = coords
                            best_cfg = (softness, lr, restart)

            aligned_coords, rmsd = align_coords(true_coords, best_coords)
            print(f"✅ {protein_id} | loss: {best_loss:.4f} | RMSD: {rmsd:.4f} Å | Softness: {best_cfg[0]} | LR: {best_cfg[1]} | Restart: {best_cfg[2]}")
            results.append((protein_id, rmsd, *best_cfg))

        except Exception as e:
            print(f"❌ Failed on {protein_id}: {e}")

    rmsds = [r for _, r, *_ in results]
    median_rmsd = np.median(rmsds)
    print(f"\n📊 Median RMSD across {len(results)} proteins: {median_rmsd:.4f} Å")

    # Plot histogram
    plt.figure()
    plt.hist(rmsds, bins=10, edgecolor='black')
    plt.title('Histogram of RMSD Values')
    plt.xlabel('RMSD (Å)')
    plt.ylabel('Frequency')
    plt.show()
