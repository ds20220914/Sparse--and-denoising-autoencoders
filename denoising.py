import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import os



# NOISE FUNCTION

def add_noise(x, noise_factor=0.3):
    noise = noise_factor * torch.randn_like(x)
    x_noisy = x + noise
    return torch.clamp(x_noisy, 0., 1.)



# DENOISING AUTOENCODER

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=28*28, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_noisy = add_noise(x)   # add noise to the input image
        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)
        return x_hat



# TRAINING MODEL

def train(model, loader, device, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) # use adam optimizer to give weights for each connection between neurons 

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)

            x_hat = model(x)

            loss = F.mse_loss(x_hat, x)  # mean squared error loss between the original and reconstructed images

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Denoising epoch {epoch+1}/{epochs} - Loss value: {total_loss/len(loader):.4f}")
    print(f"Denoising training completed. Final Loss value: {total_loss/len(loader):.4f}")



# SAVE IMAGES FROM TESTING

def save_image(tensor, path):
    img = tensor.detach().cpu().view(28, 28)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).byte().numpy()

    img = Image.fromarray(img, mode="L")
    img = img.resize((256, 256))

    img.save(path)



# TEST IF MODEL WORKS WRITE USING MNIST TEST SET 
def test(model, loader, device, save_dir="./denoising_images", n=10):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    x, _ = next(iter(loader))
    x = x[:n].to(device)

    x_flat = x.view(n, -1)

    with torch.no_grad():
        x_hat = model(x_flat)

    for i in range(n):
        save_image(x_hat[i], f"{save_dir}/recon_{i}.jpg")

    print(f"Saved {n} reconstructed images to {save_dir}")

