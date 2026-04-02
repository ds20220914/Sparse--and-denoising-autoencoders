This project implements and compares Sparse Autoencoders and Denoising Autoencoders using the MNIST dataset. The goal is to study how different architectures and training strategies affect reconstruction quality.

Install the required libraries before running:
pip3 install -r requirements.txt

Models
1. Sparse Autoencoder
- Encourages sparse latent representations
- Uses regularization ( KL divergence)
- Learns compressed features of digits

2. Denoising Autoencoder
- Takes noisy input images
- Learns to reconstruct clean images
- Improves robustness against noise

3. Dataset
- MNIST handwritten digits
- 28×28 grayscale images (resized to 256×256)
- Automatically downloaded via torchvision

4. Each model is trained using:
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Batch size: 128
- Epochs: configurable (default 5)

5. Results
- By running python3 main.py in the command line at the root of the repository, the original images are saved in the original_images folder, while the reconstructed images are saved in the denoising_images and sparse_images folders.
- Based on the squared loss value printed in the command line, we can evaluate how well both methods perform in reconstruction. The denoising autoencoder performs slightly better, as it achieves a lower loss value, which indicates that the reconstructed images are more similar to the original images.