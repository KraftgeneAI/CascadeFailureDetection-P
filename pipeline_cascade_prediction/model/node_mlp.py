"""
Node Feature MLP (Pipeline Digital Twin)
========================================
Encodes the 124-feature per-node-per-timestep vector produced by
the pipeline data generator into the shared embedding space.

Feature layout (124 total, cascade susceptibility):
  [0:18]   SCADA measurements (Pressure, Flow, Temp)    (18)
  [18:26]  High-Frequency Transient / Acoustic Sensors  ( 8)
  [26:36]  Equipment status (Valves, Pumps, Compressors)(10)
  [36:37]  Fluid mass/volumetric injection rate         ( 1)
  [37:38]  Pressure boost (Pump/Compressor head)        ( 1)
  [38:76]  1-step temporal deltas                       (38)
  [76:114] 2-step temporal deltas                       (38)
  [114]    Normalised timestep position                 ( 1)
  [115]    TTF pressure (steps to MAOP violation)       ( 1)
  [116]    TTF temp     (steps to thermal failure)      ( 1)
  [117]    TTF surge    (steps to compressor surge)     ( 1)
  [118]    TTF loading  (steps to flow capacity failure)( 1)
  [119]    mean_adjacent_pipe_loading                   ( 1)
  [120]    cascade_initiation_risk (Flow_inj/limits)    ( 1)
  [121]    cascade_reception_risk (Surge from neighbors)( 1)
  [122]    max_adjacent_pipe_loading                    ( 1)
  [123]    loading_x_max_pipe                           ( 1)

The MLP is a 3-layer feed-forward network with BatchNorm and dropout.
"""

import torch
import torch.nn as nn
from pipeline_cascade_prediction.data.generator.config import Settings


class NodeFeatureMLP(nn.Module):
    """
    3-layer MLP that maps (*, 124) → (*, embedding_dim).

    Accepts any leading batch dimensions so it works for both:
      - (B, N, 124)    — single-timestep inference
      - (B, T, N, 124) — full temporal sequence (reshape → process → reshape)
    """

    def __init__(
        self,
        in_features:   int = Settings.Embedding.NODE_FEATURE_DIM,
        hidden_1:      int = Settings.Embedding.NODE_MLP_HIDDEN_1,
        hidden_2:      int = Settings.Embedding.NODE_MLP_HIDDEN_2,
        embedding_dim: int = Settings.Model.EMBEDDING_DIM,
        dropout:       float = Settings.Embedding.DROPOUT_FC,
    ):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1 — input projection (124 → 256)
            nn.Linear(in_features, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2 — compression (256 → 128)
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3 — output projection into embedding space (128 → embedding_dim)
            nn.Linear(hidden_2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 124) float tensor — any number of leading dimensions.

        Returns:
            (..., embedding_dim) float tensor — same leading shape.
        """
        leading = x.shape[:-1]              
        flat = x.reshape(-1, x.shape[-1])   
        out  = self.net(flat)               
        return out.reshape(*leading, -1)