"""
Black-box Risk Classifier model for CARE-LG research framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class RiskClassifier(nn.Module):
    """
    3-layer MLP classifier (Input -> 32 -> 16 -> 1) with Sigmoid output for heart disease risk prediction.
    """

    def __init__(self, input_dim):
        super(RiskClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


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


def train_blackbox_model(train_loader, input_dim, epochs=20, lr=1e-3, device=None):
    """
    Trains the RiskClassifier model using BCE loss.

    Args:
        train_loader (DataLoader): PyTorch DataLoader for training.
        input_dim (int): Number of feature dimensions.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (torch.device, optional): Device to train on.

    Returns:
        model (RiskClassifier): Trained classifier in eval mode.
    """
    if device is None:
        device = get_device()

    model = RiskClassifier(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    return model
