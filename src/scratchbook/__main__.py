import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ======================================================
# Hyperparameters & Configuration
# ======================================================

# Device configuration
device = th.device(
    "cuda"
    if th.cuda.is_available()
    else "mps"
    if th.backends.mps.is_available()
    else "cpu"
)

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 20
latent_dim = 10

# ======================================================
# Data
# ======================================================


# Create paired dataset (we'll pair each image with itself for now, but you can modify this)
def create_paired_dataset(dataset):
    images = dataset.data.float() / 255.0
    images = images.unsqueeze(1)  # Add channel dimension
    labels = dataset.targets

    # One-hot encode the labels
    labels = nn.functional.one_hot(labels, num_classes=10).float()

    # For demonstration, we're pairing each image with itself
    # In a real scenario, you might want to pair based on specific criteria
    paired_images = th.stack([images, images], dim=1)

    return TensorDataset(paired_images, labels)


# ======================================================
# Model
# ======================================================


# Define the VAE model
class PairedVAE(nn.Module):
    def __init__(self, latent_dim):
        super(PairedVAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # Calculate flattened size
        self.flat_size = 64 * 7 * 7

        # Latent space
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        # mu = nn.functional.tanh(mu)
        logvar = self.fc_logvar(x)
        # logvar = nn.functional.tanh(logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        # x shape: [batch_size, 2, 1, 28, 28]
        batch_size = x.size(0)
        # Reshape to [batch_size, 2, 28, 28]
        x = x.view(batch_size, 2, 28, 28)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)

        return reconstructed, mu, logvar


# ======================================================
# Loss function & Training
# ======================================================


# Loss function
def loss_function(recon_x, x, mu, logvar, labels):
    # Extract the original images from the paired input
    original = x[:, 0]  # Shape: [batch_size, 1, 28, 28]

    # Reconstruction loss (binary cross entropy)
    BCE = nn.BCELoss(reduction="sum")(recon_x, original)

    # KL divergence
    KLD = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Label BCE
    softmax_mu = nn.functional.softmax(mu, dim=1)
    label_BCE = nn.BCELoss(reduction="sum")(softmax_mu, labels)

    return BCE + KLD + label_BCE


# Training loop
def train():
    model.train()
    train_loss = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, labels)

        # Print out a single mu sample
        if batch_idx % 100 == 0:
            with np.printoptions(precision=2, suppress=True):
                # print(f"{mu[0].shape}")
                # print(f"Mu sample: {mu[0].detach().cpu().numpy()}")
                pass

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item() / len(data):.6f}"
            )

    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_dataset):.4f}")


# Testing loop
def test():
    model.eval()
    test_loss = 0

    with th.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            labels = labels.to(device)

            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar, labels)

            test_loss += loss.item()

    test_loss /= len(test_dataset)
    print(f"====> Test set loss: {test_loss:.4f}")
    return test_loss


# Visualization function
def visualize_results():
    model.eval()
    with th.no_grad():
        # Get a batch from test loader
        data, _ = next(iter(test_loader))
        data = data.to(device)

        # Get reconstructions
        recon_batch, _, _ = model(data)

        # Plot original and reconstructed images
        n = min(8, batch_size)
        plt.figure(figsize=(12, 6))

        for i in range(n):
            # Original first image in pair
            plt.subplot(2, n, i + 1)
            plt.imshow(data[i, 0, 0].cpu().numpy(), cmap="gray")
            plt.axis("off")

            # Reconstructed image
            plt.subplot(2, n, i + 1 + n)
            plt.imshow(recon_batch[i, 0].cpu().numpy(), cmap="gray")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"results/reconstruction_epoch_{epoch}.png")
        plt.close()


if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    train_paired_dataset = create_paired_dataset(train_dataset)
    test_paired_dataset = create_paired_dataset(test_dataset)

    train_loader = DataLoader(train_paired_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_paired_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = PairedVAE(latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Create results directory
    if not os.path.exists("results"):
        os.makedirs("results")

    # Main training loop
    best_loss = float("inf")
    for epoch in tqdm(range(1, num_epochs + 1)):
        train()
        test_loss = test()

        # Visualize results
        if epoch % 5 == 0 or epoch == 1:
            visualize_results()

        # Save model
        if test_loss < best_loss:
            best_loss = test_loss
            th.save(model.state_dict(), "results/best_model.pth")

    print("Training complete!")

    # Load best model and generate final results
    model.load_state_dict(th.load("results/best_model.pth"))
    visualize_results()
