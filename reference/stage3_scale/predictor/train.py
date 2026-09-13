"""
Training loop for GRU workload predictor.

Trains per-pattern or combined models with early stopping, saves best checkpoint
and normalization parameters as a self-contained model directory.
"""

from pathlib import Path
import json
import time
import torch
import torch.nn as nn
import yaml
from typing import Optional

from .data import load_training_data, NormalizationParams
from .gru_model import WorkloadGRU


def get_device(prefer_mps: bool = True) -> torch.device:
    """Select best available device. MPS on Apple Silicon, else CPU."""
    if prefer_mps and torch.backends.mps.is_available():
        try:
            # Quick smoke test — some MPS ops can fail silently
            _ = torch.zeros(1).to('mps')
            return torch.device('mps')
        except Exception:
            pass
    return torch.device('cpu')


def train_model(
    csv_path: str,
    output_dir: str,
    h: int = 60,
    k: int = 2,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 200,
    patience: int = 20,
    prefer_mps: bool = True,
) -> dict:
    """Train a GRU model on a single workload pattern.

    Returns dict with training metrics for documentation.
    """
    device = get_device(prefer_mps)
    print(f"Training on device: {device}")

    # --- Data ---
    train_loader, val_loader, test_loader, norm = load_training_data(
        csv_path, h=h, k=k, batch_size=batch_size, device=str(device)
    )
    print(f"Train: {len(train_loader.dataset)} samples, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # --- Model ---
    model = WorkloadGRU(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=k,
        dropout=dropout,
    ).to(device)

    print(f"Model: {model.param_count:,} params, "
          f"~{model.model_size_bytes / 1024:.1f} KB on disk")

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    t0 = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / 'model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:3d}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}")

    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s, "
          f"best_val_loss={best_val_loss:.6f}")

    # --- Load best model for evaluation ---
    model.load_state_dict(torch.load(Path(output_dir) / 'model.pt',
                         weights_only=True, map_location=device))

    # --- Save config ---
    _save_config(output_dir, h, k, hidden_size, num_layers, dropout,
                 lr, batch_size, norm, train_time, best_val_loss,
                 len(train_loader.dataset))

    return {
        'best_val_loss': float(best_val_loss),
        'train_time_s': train_time,
        'epochs_run': epoch + 1,
        'param_count': model.param_count,
        'model_size_kb': model.model_size_bytes / 1024,
    }


def _save_config(
    output_dir: str,
    h: int, k: int, hidden_size: int, num_layers: int, dropout: float,
    lr: float, batch_size: int,
    norm: NormalizationParams,
    train_time: float, best_val_loss: float, n_samples: int,
):
    config = {
        'model': {
            'input_size': 1,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': k,
            'dropout': dropout,
        },
        'preprocessing': {
            'h': h,
            'k': k,
            'normalization': {
                'mu': norm.mu,
                'sigma': norm.sigma,
            },
        },
        'training': {
            'lr': lr,
            'batch_size': batch_size,
            'n_samples': n_samples,
            'best_val_loss': best_val_loss,
            'train_time_s': train_time,
        },
    }
    with open(Path(output_dir) / 'gru_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
