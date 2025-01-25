import torch
from torch import nn, optim
import numpy as np


class Autoencoder(nn.Module):
    """
    A PyTorch implementation of an Autoencoder.
    """
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)  # Compressed representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)  # Reconstruct original input
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_predict_autoencoder(model, data, epochs=50, batch_size=256, lr=0.001):
    """
    Train the autoencoder model and make the predictions.

    Args:
        model: Autoencoder instance
        data (np.ndarray): NumPy array (input data)
        epochs: Number of training epochs
        batch_size: Size of each training batch
        lr: Learning rate

    Returns:
        trained_model: Trained Autoencoder model
        embeddings: Encoded representation of the input data (as a NumPy array)
    """
    # Convert NumPy array to PyTorch tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i + batch_size]
            optimizer.zero_grad()
            encoded, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Generate embeddings after training
    with torch.no_grad():
        embeddings_tensor = model.encoder(data_tensor)

    # Convert embeddings back to NumPy
    embeddings = embeddings_tensor.numpy()

    return model, embeddings
