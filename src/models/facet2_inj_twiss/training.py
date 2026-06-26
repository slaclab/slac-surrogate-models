"""Training module for the FACET-II beam-evolution surrogate.

This is a copy of the model architecture and column definitions from the
modeling-twissparameters/train.py, included in the package so that torch.load
can reconstruct the model class.
"""

import torch
import torch.nn as nn


# ── Column definitions ────────────────────────────────────────────────────────
ID_COLUMNS = ["sample_idx", "csv_row_idx"]
TARGET_COLUMNS = [
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "norm_emit_x",
    "norm_emit_y",
    "emit_geom_x",
    "emit_geom_y",
    "beta_x",
    "alpha_x",
    "beta_y",
    "alpha_y",
    "mean_kinetic_energy",
]


# ── Model ─────────────────────────────────────────────────────────────────────
class BeamEvolutionModel(nn.Module):
    """MLP: (19 controls + s) -> 12 beam-evolution targets."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        y_mean: torch.Tensor | None = None,
        y_std: torch.Tensor | None = None,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.backbone = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.ELU(),
            nn.Linear(100, 200),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(200, 300),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(300, 200),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
        )
        self.head = nn.Linear(100, n_outputs)

        if y_mean is None:
            y_mean = torch.zeros(n_outputs, dtype=torch.float32)
        if y_std is None:
            y_std = torch.ones(n_outputs, dtype=torch.float32)
        self.register_buffer("y_mean", y_mean)
        self.register_buffer("y_std", y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions in normalized (z-score) space."""
        return self.head(self.backbone(x))

    def predict_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions in original physical units."""
        return self.forward(x) * self.y_std + self.y_mean


def build_model(n_inputs, n_outputs, y_mean=None, y_std=None) -> BeamEvolutionModel:
    return BeamEvolutionModel(n_inputs, n_outputs, y_mean=y_mean, y_std=y_std)
