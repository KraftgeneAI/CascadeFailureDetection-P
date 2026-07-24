"""
Prediction Heads Module (Oil & Gas Pipeline Twin)
=================================================
Multi-task prediction heads for pipeline cascade failure prediction.

This module contains all specialized prediction heads:
- Failure probability prediction
- Pressure prediction (physics-informed fluid dynamics)
- Fluid flow prediction (volumetric/mass flow)
- Temperature prediction
- Risk assessment (7-dimensional)
- Timing prediction (cascade propagation)
- Parent prediction (causal inference)
"""

import torch
import torch.nn as nn

from config import Settings


class FailureProbabilityHead(nn.Module):
    """
    Predicts node (valve/pump/junction) failure probability.
    
    Output: [batch_size, num_nodes, 1] — raw logits (no Sigmoid).
    Apply .sigmoid() at inference time for probabilities.
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_HIGH):
        super(FailureProbabilityHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PressureHead(nn.Module):
    """
    Predicts nodal fluid pressure.
    
    Physics-informed: Absolute pressure must be positive.
    Replaces the electrical VoltageHead. Uses Softplus to prevent dying-unit risks
    while ensuring the output is always strictly positive.
    
    Output: [batch_size, num_nodes, 1] with Softplus activation
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_LOW):
        super(PressureHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Smooth positive activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FluidFlowHead(nn.Module):
    """
    Predicts fluid flow rate (mass or volumetric) on pipeline segments.
    
    Physics-informed: Can be positive or negative (bidirectional flow).
    Replaces the Active/Reactive Power Flow heads. No activation allows for 
    linear output representing directional flow.
    
    Output: [batch_size, num_edges, 1]
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_LOW):
        super(FluidFlowHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
            # No activation - allows positive and negative directional flow
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: Concatenated source and destination node embeddings
                          [batch_size, num_edges, hidden_dim * 2]
        """
        return self.head(edge_features)


class TemperatureHead(nn.Module):
    """
    Predicts node/pipeline segment temperatures (normalised).
    
    Physics-informed: Temperature (in Kelvin or absolute scaled terms) must be positive.
    
    Output: [batch_size, num_nodes, 1] with ReLU activation
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_LOW):
        super(TemperatureHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class RiskHead(nn.Module):
    """
    Predicts seven-dimensional risk assessment.
    
    Output: [batch_size, num_nodes, Settings.Model.RISK_DIM] with sigmoid activation (0-1 range)
    """

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_HIGH):
        super(RiskHead, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, Settings.Model.RISK_DIM),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TimingHead(nn.Module):
    """
    Predicts cascade failure timing (absolute-normalised time-to-failure per node).
    
    Output: [batch_size, num_nodes, 1] in (0, 1) — absolute-normalised time.
    """

    TIMING_DROPOUT = 0.25

    def __init__(self, hidden_dim: int, dropout: float = Settings.Model.HEAD_DROPOUT_HIGH):
        super(TimingHead, self).__init__()

        mid = hidden_dim // 2
        _drop = self.TIMING_DROPOUT

        self.layer1 = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Dropout(_drop),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(mid, mid),
            nn.LayerNorm(mid),
            nn.ReLU(),
            nn.Dropout(_drop),
        )
        self.out = nn.Linear(mid, 1)
        self.residual_proj = nn.Linear(hidden_dim, mid)
        self.activation = nn.Sigmoid()

        # Output bias initialisation: centres predictions around typical cascade region
        nn.init.constant_(self.out.bias, 1.386)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        h = self.layer1(x)
        h = self.layer2(h) + residual
        return self.activation(self.out(h))


class ParentPredictionHead(nn.Module):
    """
    Predicts the causal parent for each node in a cascade failure.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.trigger_key = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.dropout     = nn.Dropout(dropout)
        self.scale       = hidden_dim ** -0.5

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        Q = self.dropout(self.query_proj(h))
        K = self.dropout(self.key_proj(h))
        trigger = self.key_proj(self.trigger_key)
        B = h.shape[0]
        trigger = trigger.unsqueeze(0).expand(B, -1, -1)
        
        all_keys = torch.cat([K, trigger], dim=1)
        scores = torch.bmm(Q, all_keys.transpose(1, 2)) * self.scale
        return scores