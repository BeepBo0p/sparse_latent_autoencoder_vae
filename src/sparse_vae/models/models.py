import torch as th
import torch.nn as nn


class SimpleAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SimpleAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


if __name__ == "__main__":
    device = th.device(
        "cuda"
        if th.cuda.is_available()
        else "mps"
        if th.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device:\t{device}")

    model = SimpleAutoEncoder(input_dim=784, hidden_dim=256, latent_dim=64).to(device)
    x = th.randn(16, 784).to(device)  # Example input batch of size 16
    y = model(x)
    print(f"Input shape:\t{x.shape}")
    print(f"Output shape:\t{y.shape}")
