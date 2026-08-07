"""
Tabular Variational Autoencoder (TabularVAE) architecture for CARE-LG research framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TabularVAE(nn.Module):
    """
    Tabular VAE architecture:
    - Encoder: Input -> 64 -> 32 -> (mu, logvar) [latent dim d=4]
    - Reparameterization: z = mu + std * eps
    - Decoder: z (d=4) -> 32 -> 64 -> Input reconstruction
    """

    def __init__(self, input_dim, latent_dim=4):
        super(TabularVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, 32)
        self.decoder_fc2 = nn.Linear(32, 64)
        self.decoder_out = nn.Linear(64, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h1 = F.relu(self.decoder_fc1(z))
        h2 = F.relu(self.decoder_fc2(h1))
        return self.decoder_out(h2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes VAE loss as sum of MSE reconstruction loss and beta-scaled KL divergence.
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld


def get_device():
    """
    Select best available PyTorch device (MPS, CUDA, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_vae(train_loader, input_dim, latent_dim=4, epochs=50, beta=1.0, lr=1e-3, device=None):
    """
    Trains the TabularVAE model.

    Args:
        train_loader (DataLoader): PyTorch DataLoader for training data.
        input_dim (int): Input feature dimension.
        latent_dim (int): Latent dimension size (default: 4).
        epochs (int): Number of epochs (default: 50).
        beta (float): Weight for KL divergence term (default: 1.0).
        lr (float): Learning rate (default: 1e-3).
        device (torch.device, optional): Device for training.

    Returns:
        vae (TabularVAE): Trained Tabular VAE model in eval mode.
    """
    if device is None:
        device = get_device()

    vae = TabularVAE(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    vae.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(X_batch)
            loss = loss_function(recon_batch, X_batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    vae.eval()
    return vae
