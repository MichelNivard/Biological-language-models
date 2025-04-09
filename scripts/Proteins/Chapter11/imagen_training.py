import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from torchvision import transforms
from tqdm import tqdm

# === PARAMS ===
image_dir = "contact_maps_all/"
embedding_dir = "embeddings_all/"

embedding_dim = 640  # ESM-2 150M
image_size = 128
batch_size = 24
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Custom Dataset ===
class ProteinDataset(Dataset):
    def __init__(self, image_dir, embedding_dir):
        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        self.common_ids = self._get_common_ids()
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (H, W) → (1, H, W)
        ])

    def _get_common_ids(self):
        img_ids = {f[:-4] for f in os.listdir(self.image_dir) if f.endswith('.jpg')}
        emb_ids = {f[:-4] for f in os.listdir(self.embedding_dir) if f.endswith('.npy')}
        return sorted(img_ids & emb_ids)

    def __len__(self):
        return len(self.common_ids)

    def __getitem__(self, idx):
        pid = self.common_ids[idx]
        image = Image.open(os.path.join(self.image_dir, f"{pid}.jpg")).convert("L")
        image_tensor = self.transform(image)  # shape: (1, 128, 128)
        embedding = torch.from_numpy(np.load(os.path.join(self.embedding_dir, f"{pid}.npy"))).float()  # (128, 640)
        return image_tensor, embedding  # image first, condition second!

# === Dataset ===
data = ProteinDataset(image_dir, embedding_dir)

# === Eval & Train ===

# Set the number of eval samples
eval_size = 128
total_size = len(data)
all_indices = np.arange(total_size)
np.random.shuffle(all_indices)

# Split indices
eval_indices = all_indices[:eval_size]
train_indices = all_indices[eval_size:]

# Subsets
train_subset = Subset(data, train_indices)
eval_subset = Subset(data, eval_indices)


# === Eval function ===
def evaluate(trainer, dataloader):
    trainer.eval()
    total_eval_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images, cond = batch
            images, cond = images.to(device), cond.to(device)
            loss = trainer.forward_loss(
                images=images,
                text_embeds=cond,
                unet_number=1
            )
            total_eval_loss += loss.item()
    return total_eval_loss / len(dataloader)


# === Define U-Nets for Imagen Cascade ===
unet1 = Unet(
    dim=64,
    cond_dim=embedding_dim,
    dim_mults=(1, 2, 4),
    channels=1,
    num_resnet_blocks=2,
    layer_attns=(False, False, True),
    layer_cross_attns=(False, True, True),
)


# === Imagen ===
imagen = Imagen(    
    unets=(unet1),
    image_sizes=(128),
    timesteps=1000,
    cond_drop_prob=0.1,
    text_embed_dim=embedding_dim,
    channels=1
).to(device)

# === ImagenTrainer ===
# Initialize the trainer
trainer = ImagenTrainer(imagen,fp16=True).to(device)


# FIX: Wrap with cond_images_dataset so (cond, image) is handled correctly
trainer.add_train_dataset(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
trainer.add_valid_dataset(eval_subset, batch_size=batch_size, shuffle=True, num_workers=0)

# set up eval...
eval_losses = []  # store losses for plotting later
eval_every = 100  # steps


steps_per_epoch = len(train_subset) // batch_size


# === Training Loop ===
print(" Starting training...")
for epoch in range(epochs):
    total_loss = 0.0
    for step in tqdm(range(steps_per_epoch)):
        loss1 = trainer.train_step(unet_number=1)
        total_loss += loss1

        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss1:.4f}")

        if step % eval_every == 0:
            eval_loss = trainer.valid_step(unet_number=1)
            eval_losses.append((epoch, step, eval_loss))
            print(f"📉 Eval Loss @ Epoch {epoch}, Step {step}: {eval_loss:.4f}")

    print(f"[Epoch {epoch + 1}/{epochs}]  Loss: {total_loss / steps_per_epoch:.4f}")
    if (epoch+1) % 5 == 0:
        trainer.save(f"imagen_cont/imagen_protein_epoch{epoch + 1}.pt")

print("Training complete.")





### Eyeball a few samples

import matplotlib.pyplot as plt
import torch
import numpy as np

# Load dataset and model
dataset = ProteinDataset(image_dir, embedding_dir)
imagen.eval()

# Indices to sample
sample_indices = eval_indices[:4]

# Store results
generated_images = []
true_images = []

# Generate all samples first
for idx in sample_indices:
    true_image, conditioning = dataset[idx]
    conditioning = conditioning.unsqueeze(0).to(device)

    with torch.no_grad():
        generated = imagen.sample(batch_size=1, cond_scale=3.0, text_embeds=conditioning)
        generated_image = generated[0].cpu().squeeze().numpy()

    # Store ground truth and generated images
    true_images.append(true_image.squeeze().numpy())
    generated_images.append(generated_image)

# Plot all at once
plt.figure(figsize=(8, 12))  # 4 rows × 2 columns
for i in range(4):
    # Ground truth
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(true_images[i]*-1, cmap='viridis')
    plt.title(f"Ground Truth {sample_indices[i]}")
    plt.axis('off')

    # Generated
    plt.subplot(4, 2, 2 * i + 2)
    plt.imshow(generated_images[i]*-1, cmap='viridis')
    plt.title(f"Generation {sample_indices[i]}")
    plt.axis('off')

plt.tight_layout()
# Optionally save to file:
# plt.savefig("contactmap_comparison.png", dpi=150)
plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Imagen Parameters: {count_parameters(imagen):,}")