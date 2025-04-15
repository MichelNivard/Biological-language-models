import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from accelerate import Accelerator

# -------- Dataset -------- #
class ContactMapDataset(Dataset):
    def __init__(self, folder, image_size=128):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        array = np.load(path).astype(np.float32)  # shape: (128, 128)

        # Optional: assert correct shape or pad
        if array.shape != (self.image_size, self.image_size):
            raise ValueError(f"Invalid shape: {array.shape} in {path}")

        tensor = torch.from_numpy(array).unsqueeze(0)  # shape: [1, H, W]
        return tensor

# -------- Model & Schedulers -------- #
image_size = 128
batch_size = 8
num_steps = 60
lr = 8e-5

model = UNet2DModel(
    sample_size=image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),  # like (1, 2, 4, 8) dim_mults
    down_block_types=(
        "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
    )
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# -------- Data -------- #
dataset = ContactMapDataset("contact_maps_float", image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------- Training -------- #
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

model.train()
for step in tqdm(range(num_steps)):
    for batch in dataloader:
        clean_images = batch
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device).long()

        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = model(noisy_images, timesteps).sample

        loss = nn.functional.mse_loss(noise_pred, noise)
        accelerator.backward(loss)

        print(f"Step check?  Loss: {loss.item():.4f}")

        optimizer.step()
        optimizer.zero_grad()
    print(f"Step {step+1}/{num_steps} - Loss: {loss.item():.4f}")

# -------- Sampling -------- #
model.eval()
ddim = DDIMScheduler.from_config(noise_scheduler.config)
ddim.set_timesteps(150)

sample = torch.randn((1, 1, image_size, image_size)).to(model.device)
for t in tqdm(ddim.timesteps):
    with torch.no_grad():
        model_output = model(sample, t).sample
    sample, _ = ddim.step(model_output, t, sample)

final_img = sample[0, 0].cpu().numpy()
plt.imshow(final_img, cmap='gray')
plt.title("Final Sampled Contact Map")
plt.axis('off')
plt.show()

# -------- Show Sampling Steps -------- #
# To visualize the intermediate steps:
timesteps_to_show = 5
ddim.set_timesteps(50)
samples = []
sample = torch.randn((1, 1, image_size, image_size)).to(model.device)

for idx, t in enumerate(ddim.timesteps):
    with torch.no_grad():
        model_output = model(sample, t).sample
    sample, _ = ddim.step(model_output, t, sample)

    if idx in np.linspace(0, len(ddim.timesteps)-1, timesteps_to_show, dtype=int):
        samples.append(sample[0, 0].detach().cpu().numpy())

fig, axs = plt.subplots(1, timesteps_to_show, figsize=(15, 3))
for i, frame in enumerate(samples):
    axs[i].imshow(frame, cmap='gray')
    axs[i].set_title(f"Step {i}")
    axs[i].axis('off')
plt.tight_layout()
plt.show()
