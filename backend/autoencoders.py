import numpy as np
import torch
from torch import nn, optim


class Autoencoder(nn.Module):
    """
    A PyTorch implementation of an Autoencoder with flexible architecture.
    """

    def __init__(self, input_dim, encoding_dim, hidden_dim, dropout_rate):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, encoding_dim),  # Compressed representation
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # Reconstruct original input
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_predict_autoencoder(
    model,
    data,
    epochs=50,
    batch_size=256,
    lr=0.001,
    l1_penalty=0.001,
    weight_decay=1e-5,
    debug=True,
):
    """
    Train the autoencoder model and keep the one with the lowest loss.

    Args:
        model: Autoencoder instance
        data (np.ndarray): NumPy array (input data)
        epochs: Number of training epochs
        batch_size: Size of each training batch
        lr: Learning rate
        l1_penalty: Weight of the L1 regularization on encodings
        weight_decay: Weight decay to apply
        debug: If to print debug messages

    Returns:
        best_model: Trained Autoencoder model (with the lowest loss)
        embeddings: Encoded representation of the input data (as a NumPy array)
        best_loss: Best average loss recorded during training
    """
    # Convert NumPy array to PyTorch tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track the best loss and best model state
    best_loss = float("inf")
    best_epoch = -1
    best_model_state = None

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i : i + batch_size]
            optimizer.zero_grad()

            # Forward pass
            encoded, decoded = model(batch)

            # Reconstruction loss (MSE)
            reconstruction_loss = criterion(decoded, batch)

            # Regularization loss (L1 penalty on encoded values)
            l1_loss = l1_penalty * torch.mean(torch.abs(encoded))

            # Total loss
            loss = reconstruction_loss + l1_loss

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model_state = model.state_dict()  # Save best model state

        if debug:
            print(
                f"Epoch {epoch}/{epochs}, Avg Loss: {avg_loss:.4f}, "
                f"Best Loss: {best_loss:.4f} at Epoch {best_epoch}"
            )

    # Restore the best model state
    model.load_state_dict(best_model_state)

    # Generate embeddings after training
    with torch.no_grad():
        embeddings_tensor = model.encoder(data_tensor)

    # Convert embeddings back to NumPy
    embeddings = embeddings_tensor.numpy()

    if debug:
        print(f"\n✅ Training Completed! Best Loss: {best_loss:.4f} at Epoch {best_epoch}")

    return model, embeddings, best_loss


def objective(trial, data):
    """
    Optuna objective function for tuning the autoencoder hyperparameters.
    """
    # Sample hyperparameters
    encoding_dim = trial.suggest_int("encoding_dim", 10, 30)
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.1, 0.3)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    l1_penalty = trial.suggest_loguniform("l1_penalty", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    epochs = 75

    input_dim = data.shape[1]

    # Create model
    model = Autoencoder(input_dim, encoding_dim, hidden_dim, dropout_rate)

    # Train the model
    _, embeddings, best_loss = train_predict_autoencoder(
        model,
        data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        l1_penalty=l1_penalty,
        weight_decay=weight_decay,
        debug=False,
    )

    return best_loss  # Optuna minimizes this
