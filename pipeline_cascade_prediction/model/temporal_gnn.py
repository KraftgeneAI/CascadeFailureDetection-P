"""
Temporal GNN Cell (Oil & Gas Pipeline Twin)
===========================================
Combines graph attention with LSTM for temporal dynamics modeling 
of pipeline fluid networks.

Processes sequences of spatial pipeline embeddings (nodes/valves/pumps) 
and temporal state changes (pressure/flow waves) to track cascade 
propagation over time.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .graph_attention import GraphAttentionLayer
from config import Settings


class TemporalGNNCell(nn.Module):
    """Temporal GNN Cell combining graph attention with LSTM for fluid networks."""
    
    def __init__(self, node_features: int, hidden_dim: int,
                 edge_dim: Optional[int] = None, num_heads: int = 4, dropout: float = 0.3):
        super(TemporalGNNCell, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.gat_out_channels_per_head = hidden_dim // num_heads
        self.gat_out_dim = num_heads * self.gat_out_channels_per_head
        
        # Spatial processing: How pressure/flow propagates to adjacent pipeline nodes
        self.gat = GraphAttentionLayer(
            in_channels=node_features,
            out_channels=self.gat_out_channels_per_head,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
        if self.gat_out_dim != hidden_dim:
            self.projection = nn.Linear(self.gat_out_dim, hidden_dim)
        else:
            self.projection = None
        
        # Temporal processing: How local pipeline stress builds up over time
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=Settings.Model.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=Settings.Model.LSTM_DROPOUT
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                edge_mask: Optional[torch.Tensor] = None,
                h_prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, _ = x.shape
        
        # Pass mask to GAT to handle isolated/closed valves
        spatial_features = self.gat(x, edge_index, edge_attr, edge_mask=edge_mask)
        
        if self.projection is not None:
            spatial_features = self.projection(spatial_features)
        
        # Residual connection requires same dimensionality
        spatial_features = spatial_features + x   
        
        if h_prev is None:
            h_prev = (
                torch.zeros(Settings.Model.LSTM_NUM_LAYERS, B * N, self.hidden_dim, device=x.device),
                torch.zeros(Settings.Model.LSTM_NUM_LAYERS, B * N, self.hidden_dim, device=x.device)
            )
        
        spatial_flat = spatial_features.reshape(B * N, 1, self.hidden_dim)
        
        output, (h_new, c_new) = self.lstm(spatial_flat, h_prev)
        
        h_out = output.squeeze(1).reshape(B, N, self.hidden_dim)
        h_out = self.layer_norm(h_out)
        
        return h_out, (h_new, c_new)