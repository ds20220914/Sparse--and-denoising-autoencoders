import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import os



# SPARSE AUTOENCODER
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, latent_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def kl_sparsity(self, z, p=0.05):
        p_hat = torch.mean(z, dim=0)

        kl = p * torch.log(p / (p_hat + 1e-8)) + \
             (1 - p) * torch.log((1 - p) / (1 - p_hat + 1e-8))

        return torch.sum(kl)



# TRAIN THE SPARSE AUTOENCODER MODEL USING MNIST DATASET
def train(model, loader, device, epochs=5, lr=1e-3, beta=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for x, _ in loader:
            x = x.view(x.size(0), -1).to(device)

            x_hat, z = model(x)

            recon_loss = F.mse_loss(x_hat, x)
            sparsity_loss = model.kl_sparsity(z)

            loss = recon_loss + beta * sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Sparse epoch {epoch+1}/{epochs} - Loss value: {total_loss/len(loader):.4f}")
    print(f"Sparse training completed. Final Loss value: {total_loss/len(loader):.4f}")




# SAVE DECODED IMAGES AFTER TESTING INTO ./images FOLDER
def save_image(tensor, path):
    img = tensor.detach().cpu().view(28, 28)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).byte().numpy()

    img = Image.fromarray(img, mode="L")
    img = img.resize((256, 256))
    img.save(path)



# TEST FUNCTION USING MNIST TEST SET
def test(model, loader, device, save_dir="./sparse_images", n=10):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    x, _ = next(iter(loader))
    x = x[:n].view(n, -1).to(device)

    with torch.no_grad():
        x_hat, _ = model(x)

    for i in range(n):
        save_image(x_hat[i], f"{save_dir}/recon_{i}.jpg")

    print(f"Saved {n} reconstructed images to {save_dir}")


