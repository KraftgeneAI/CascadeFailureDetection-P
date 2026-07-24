"""
Graph Attention Layer
=====================
Computes attention-weighted message passing between connected pipeline nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) adapted for batched pipeline networks.
    Supports multi-head attention and PyG-style keyword arguments.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        heads: int = 1, 
        concat: bool = True, 
        dropout: float = 0.2, 
        edge_dim: int = None, 
        alpha: float = 0.2
    ):
        super(GraphAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # If concatenating heads, the output feature size is heads * out_channels
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.a = nn.Linear(2 * out_channels, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr=None, edge_mask=None) -> torch.Tensor:
        """
        Forward pass handling batched [B, N, F] inputs.
        """
        is_batched = h.dim() == 3
        if not is_batched:
            h = h.unsqueeze(0)  
            
        B, N, _ = h.shape
        
        # Apply linear transformation and split by heads: [B, N, Heads, Channels]
        Wh = self.W(h).view(B, N, self.heads, self.out_channels)
        
        # Calculate attention scores (averaged across heads to save memory)
        Wh_mean = Wh.mean(dim=2) 
        Wh_i = Wh_mean.unsqueeze(2).expand(B, N, N, self.out_channels)
        Wh_j = Wh_mean.unsqueeze(1).expand(B, N, N, self.out_channels)
        
        # e = LeakyReLU(a[Wh_i || Wh_j])
        e = self.leakyrelu(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))
        
        # Create adjacency matrix
        adj = torch.zeros(B, N, N, device=h.device)
        src, dst = edge_index[0], edge_index[1]
        
        adj[:, dst, src] = 1.0  
        adj[:, range(N), range(N)] = 1.0  # Self-loops
        
        # Mask out non-existent edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax and dropout
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention weights to node features (Einstein summation handles the batching and heads easily)
        h_prime = torch.einsum('bji, bjhc -> bihc', attention, Wh)
        
        if self.concat:
            # Flatten heads: [B, N, heads * out_channels]
            h_prime = h_prime.reshape(B, N, self.heads * self.out_channels)
        else:
            # Average heads: [B, N, out_channels]
            h_prime = h_prime.mean(dim=2)
            
        if not is_batched:
            h_prime = h_prime.squeeze(0)
            
        return F.elu(h_prime)