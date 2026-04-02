import torch
import os
import torch.nn.functional as F
from torchvision import datasets, transforms
from sparse import SparseAutoencoder, train as train_sparse, test as test_sparse
from denoising import DenoisingAutoencoder, train as train_denoising, test as test_denoising
from torchvision.utils import save_image
from PIL import Image



class main():

    def save_original_images(dataset, save_dir, n):
        os.makedirs(save_dir, exist_ok=True)

        for i in range(n):
            img, label = dataset[i]  # (1, 28, 28)

            # varmista shape
            if img.dim() == 2:
                img = img.unsqueeze(0)

            # resize 28x28 -> 256x256
            img = img.unsqueeze(0)  # (1,1,28,28)
            img = F.interpolate(img, size=(256, 256), mode="bilinear", align_corners=False)
            img = img.squeeze(0)    # (1,256,256)

            # convert PIL
            img = img.squeeze(0).detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype("uint8")

            img = Image.fromarray(img, mode="L")

            img.save(f"{save_dir}/img_{i}_label_{label}.png")


    device = "cpu" # use cpu for training and testing
    transform = transforms.ToTensor() # transform the images to tensors

    # load the MNIST dataset for training
    train_data = datasets.MNIST( 
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    # load the MNIST dataset for testing
    test_data = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    # use dataloader to load the data in batches
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=128,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=128,
        shuffle=False
    )

    # MODEL
    sparse_model = SparseAutoencoder()
    denoising_model = DenoisingAutoencoder()
    save_original_images(train_data, "./original_images", n=10)

    # TRAIN
    train_sparse(sparse_model, train_loader, device, epochs=5, lr=1e-3, beta=1e-3)
    train_denoising(denoising_model, train_loader, device, epochs=5)

    # TEST 
    test_sparse(sparse_model, test_loader, device, n=10)
    test_denoising(denoising_model, test_loader, device, n=10)

if __name__ == "__main__":
    main()