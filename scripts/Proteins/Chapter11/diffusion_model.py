

# Full training script for protein contact map diffusion model
# using LucidRain's denoising-diffusion-pytorch (grayscale input)

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import matplotlib.pyplot as plt


model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True,
    channels=1
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 150    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

trainer = Trainer(
    diffusion,
    'contact_maps_128/',
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 60,         # total training steps
    gradient_accumulate_every = 16,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()

samples = diffusion.sample(batch_size=1, return_all_timesteps=True)
# shape: (timesteps, batch, channels, height, width)

# samples shape: (timesteps, batch, channels, height, width)
final_img = samples[-1][0][0].cpu().numpy()  # last frame, batch=0, channel=0

plt.figure()
plt.imshow(final_img, cmap='gray')
plt.title("Final Sampled Contact Map")
plt.axis('off')
plt.show()


samples = diffusion.sample(batch_size=1, return_all_timesteps=True)

print("Samples shape:", samples.shape)
samples = samples[0]  # shape: (51, 1, 256, 256)

timesteps = samples.shape[0]
step_indices = np.linspace(0, timesteps - 1, 5, dtype=int)

fig, axs = plt.subplots(1, 5, figsize=(15, 3))

for i, step_idx in enumerate(step_indices):
    frame = samples[step_idx, 0].detach().cpu().numpy()  # shape: (256, 256)
    axs[i].imshow(frame, cmap='gray')
    axs[i].set_title(f"Step {step_idx}")
    axs[i].axis('off')

plt.tight_layout()
plt.show()