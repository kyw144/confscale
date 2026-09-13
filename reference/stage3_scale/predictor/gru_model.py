"""
2-layer stacked GRU for workload forecasting.

Architecture: GRU(64) → Dropout(0.2) → GRU(64) → Linear(k) → ReLU
Parameters: ~50K — intentionally small for fast inference (<5ms).
"""

import torch
import torch.nn as nn
from torch import Tensor


class WorkloadGRU(nn.Module):
    """Stacked GRU predictor for per-service request rate forecasting.

    Input:  (batch, seq_len=h, features=1)  — normalized RPS history
    Output: (batch, k)                       — direct multi-output forecast
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, 1) — normalized RPS window

        Returns:
            (batch, k) — normalized RPS predictions (may be negative in norm-space)
        """
        # GRU: out shape (batch, seq_len, hidden_size)
        out, _ = self.gru(x)
        # Use the last hidden state — no activation (ReLU breaks normalized data)
        return self.linear(out[:, -1, :])

    def predict(self, x: Tensor) -> Tensor:
        """Inference-only forward pass (no grad)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def model_size_bytes(self) -> int:
        """Estimated size on disk (float32 weights only)."""
        return self.param_count * 4
