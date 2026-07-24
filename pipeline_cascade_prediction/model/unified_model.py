"""
Unified Cascade Prediction Model (Pipeline SCADA Engine)
========================================================
Simplified model focusing purely on internal pipeline telemetry.
- Processes 124-dim SCADA/telemetry vectors directly.
- No environmental or multi-modal fusion.
- Temporal dynamics with LSTM.
- Fluid dynamics modeling (Pressure & Flow).
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging

from config import Settings

# Only import the node MLP (which processes the SCADA telemetry)
from .node_mlp_2 import NodeFeatureMLP

# Import layers
from .layers import GraphAttentionLayer, TemporalGNNCell

# Import pipeline-specific prediction heads
from .prediction_heads_2 import (
    FailureProbabilityHead,
    RiskHead,
    TimingHead,
    ParentPredictionHead,
    PressureHead,      
    TemperatureHead,
    FluidFlowHead,     
)

from .physics_informed import PhysicsInformedLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class UnifiedPipelinePredictionModel(nn.Module):
    """
    Streamlined pipeline model processing purely SCADA/telemetry data.
    """
    
    def __init__(
        self,
        embedding_dim: int = Settings.Model.EMBEDDING_DIM,
        hidden_dim: int = Settings.Model.HIDDEN_DIM,
        num_gnn_layers: int = Settings.Model.NUM_GNN_LAYERS,
        heads: int = Settings.Model.HEADS,
        dropout: float = Settings.Model.DROPOUT
    ):
        super().__init__()
        
        # Sole input encoder for pipeline SCADA telemetry
        self.scada_encoder = NodeFeatureMLP(embedding_dim=embedding_dim)

        # Temporal GNN layers
        self.temporal_gnn = TemporalGNNCell(
            node_features=embedding_dim,
            hidden_dim=hidden_dim,
            edge_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout
        )
        
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_channels=hidden_dim, out_channels=hidden_dim // heads,
                heads=heads, concat=True, dropout=dropout, edge_dim=hidden_dim
            ) for _ in range(num_gnn_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # Edge embedding (Pipeline physical limits and properties)
        self.edge_embedding = nn.Sequential(
            nn.Linear(Settings.Model.EDGE_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Long-range cascade heads
        self.failure_prob_head = FailureProbabilityHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_HIGH)
        self.failure_time_head = TimingHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_HIGH)
        self.risk_head         = RiskHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_HIGH)
        self.parent_head       = ParentPredictionHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_HIGH)

        # Short-range fluid physics heads
        self.pressure_head = PressureHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_LOW)
        self.temp_head     = TemperatureHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_LOW)
        self.flow_head     = FluidFlowHead(hidden_dim, dropout=Settings.Model.HEAD_DROPOUT_LOW)

        self.physics_loss = PhysicsInformedLoss()
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Direct encoding of SCADA features (Bypassing complex fusion)
        scada_emb = self.scada_encoder(batch['node_features'])
        has_temporal = scada_emb.dim() == 4 

        edge_attr_input = batch.get('edge_attr')
        if edge_attr_input is None:
            E = batch['edge_index'].shape[1]
            edge_attr_input = torch.zeros(scada_emb.shape[0], E, Settings.Model.EDGE_FEATURES, device=scada_emb.device)

        if edge_attr_input.dim() == 2:
            edge_attr_input = edge_attr_input.unsqueeze(0).expand(scada_emb.shape[0], -1, -1)
        
        edge_mask_input = batch.get('edge_mask')
        
        # 2. Process temporal sequences directly
        if has_temporal:
            B, T, N, D = scada_emb.shape
            T_in = max(T - 1, 1)   

            h_states, lstm_state = [], None

            for t in range(T_in):
                x_t = scada_emb[:, t, :, :]
                edge_attr_t = edge_attr_input[:, t, :, :] if edge_attr_input.dim() == 4 else edge_attr_input
                edge_embedded_t = self.edge_embedding(edge_attr_t)
                mask_t = edge_mask_input[:, t, :] if edge_mask_input is not None and edge_mask_input.dim() == 3 else edge_mask_input

                h_t, lstm_state = self.temporal_gnn(
                    x_t, batch['edge_index'], edge_embedded_t, edge_mask=mask_t, h_prev=lstm_state
                )
                h_states.append(h_t)

            _, c_lstm = lstm_state
            c_final = c_lstm[-1].reshape(B, N, self.temporal_gnn.hidden_dim)  

            last_t = T_in - 1
            edge_embedded = self.edge_embedding(edge_attr_input[:, last_t, :, :] if edge_attr_input.dim() == 4 else edge_attr_input)
            h_stack = torch.stack(h_states, dim=2)   

            if 'sequence_length' in batch:
                lengths = batch['sequence_length'].cpu()
                h = torch.stack([h_stack[b, :, max(min(int(lengths[b]) - 1, T_in - 1), 0), :] for b in range(B)], dim=0)
            else:
                h = h_stack[:, :, -1, :]
        else:
            c_final = None
            edge_embedded = self.edge_embedding(edge_attr_input)
            h, _ = self.temporal_gnn(scada_emb, batch['edge_index'], edge_embedded, edge_mask=edge_mask_input)
        
        final_mask = edge_mask_input[:, -1, :] if (edge_mask_input is not None and edge_mask_input.dim() == 3) else edge_mask_input

        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            h_new = gnn_layer(h, batch['edge_index'], edge_embedded, edge_mask=final_mask)
            h = layer_norm(h + h_new)

        # 3. Route to Heads
        long_range_rep = c_final if has_temporal else h
        failure_prob   = self.failure_prob_head(long_range_rep)
        failure_timing = self.failure_time_head(long_range_rep)
        risk_scores    = self.risk_head(long_range_rep)
        parent_logits  = self.parent_head(long_range_rep)    

        pressure_pred = self.pressure_head(h)
        temp_pred     = self.temp_head(h)
        
        src_idx, dst_idx = batch['edge_index'][0], batch['edge_index'][1]
        flow_pred = self.flow_head(torch.cat([h[:, src_idx, :], h[:, dst_idx, :]], dim=-1))

        return {
            'failure_probability': failure_prob,
            'cascade_timing':      failure_timing,
            'risk_scores':         risk_scores,
            'parent_logits':       parent_logits,
            'pressure_pred':       pressure_pred if has_temporal else None,
            'temp_pred':           temp_pred     if has_temporal else None,
            'flow_pred':           flow_pred     if has_temporal else None,
            'node_embeddings':     h,
        }
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], graph_properties: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.physics_loss(predictions, targets, graph_properties)