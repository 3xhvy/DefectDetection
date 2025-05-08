"""
Conditional Graph Diffusion Model for Vulnerability Repair

This module extends the base graph diffusion model to support conditional generation,
which allows for guided vulnerability repair by conditioning the diffusion process
on security-related information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, to_networkx
import networkx as nx
import numpy as np
import logging

from graph_diffusion_model import GraphDiffusionModel, GraphUNet, GraphConvBlock, GCNConvWithTime

# Set up logging
logger = logging.getLogger(__name__)

class ConditionalGraphDiffusionModel(GraphDiffusionModel):
    """
    Conditional diffusion model for vulnerability repair
    """
    def __init__(self, node_dim, condition_dim, **kwargs):
        super(ConditionalGraphDiffusionModel, self).__init__(node_dim, **kwargs)
        self.condition_dim = condition_dim
        
        # Modify the denoiser to accept conditional information
        self.denoiser = ConditionalGraphUNet(
            node_dim=node_dim,
            condition_dim=condition_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            num_layers=kwargs.get('num_layers', 6),
            dropout=kwargs.get('dropout', 0.1)
        )
    
    def sample_conditional(self, num_nodes, edge_index, condition, batch=None, device='cuda'):
        """
        Sample a graph conditioned on vulnerability repair information
        """
        # Start from pure noise
        x_T = torch.randn(num_nodes, self.node_dim, device=device)
        
        # Iteratively denoise with condition
        x_t = x_T
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_nodes,), t, device=device, dtype=torch.long)
            x_t = self.reverse_diffusion_step_conditional(x_t, t_tensor, edge_index, condition, batch)
        
        return x_t
    
    def reverse_diffusion_step_conditional(self, x_t, t, edge_index, condition, batch=None):
        """
        Single step of conditional reverse diffusion
        """
        # Predict the noise component with condition
        predicted_noise = self.denoiser(x_t, t, edge_index, condition, batch)
        
        # Rest of the implementation follows the unconditional version
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t]
        beta = self.betas[t]
        
        coef1 = torch.sqrt(alpha_cumprod_prev) / torch.sqrt(1 - alpha_cumprod)
        coef2 = torch.sqrt(1 - alpha_cumprod_prev) - torch.sqrt(alpha_cumprod_prev) * torch.sqrt(1 - alpha_cumprod) / torch.sqrt(alpha_cumprod)
        mean = coef1 * (x_t - coef2 * predicted_noise)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))
            x_t_minus_1 = mean + sigma * noise
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    def loss_function_conditional(self, x_0, edge_index, condition, batch=None):
        """
        Calculate the conditional diffusion loss
        """
        batch_size = x_0.shape[0]
        
        # Sample timestep uniformly
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)
        
        # Forward diffusion to get x_t and the noise
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # Predict the noise component with condition
        predicted_noise = self.denoiser(x_t, t, edge_index, condition, batch)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

class ConditionalGraphUNet(GraphUNet):
    """
    U-Net style architecture for conditional graph denoising
    """
    def __init__(self, node_dim, condition_dim, **kwargs):
        super(ConditionalGraphUNet, self).__init__(node_dim, **kwargs)
        self.condition_dim = condition_dim
        
        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Modify layers to accept condition
        self.down_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.down_layers.append(
                ConditionalGraphConvBlock(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    time_channels=self.hidden_dim,
                    condition_channels=self.hidden_dim,
                    dropout=kwargs.get('dropout', 0.1)
                )
            )
        
        # Middle
        self.middle = ConditionalGraphConvBlock(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            time_channels=self.hidden_dim,
            condition_channels=self.hidden_dim,
            dropout=kwargs.get('dropout', 0.1)
        )
        
        # Up path with skip connections
        self.up_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.up_layers.append(
                ConditionalGraphConvBlock(
                    in_channels=2 * self.hidden_dim,
                    out_channels=self.hidden_dim,
                    time_channels=self.hidden_dim,
                    condition_channels=self.hidden_dim,
                    dropout=kwargs.get('dropout', 0.1)
                )
            )
    
    def forward(self, x, t, edge_index, condition, batch=None):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Condition embedding
        c_emb = self.condition_embed(condition)
        
        # Initial projection
        h = self.input_proj(x)
        
        # Down path with skip connections
        skips = [h]
        for layer in self.down_layers:
            h = layer(h, t_emb, c_emb, edge_index)
            skips.append(h)
        
        # Middle
        h = self.middle(h, t_emb, c_emb, edge_index)
        
        # Up path with skip connections
        for layer, skip in zip(self.up_layers, reversed(skips)):
            h = torch.cat([h, skip], dim=-1)
            h = layer(h, t_emb, c_emb, edge_index)
        
        # Final projection
        output = self.output_proj(h)
        
        return output

class ConditionalGraphConvBlock(nn.Module):
    """
    Graph convolution block with time and condition conditioning
    """
    def __init__(self, in_channels, out_channels, time_channels, condition_channels, dropout=0.1):
        super(ConditionalGraphConvBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = GCNConvWithTimeAndCondition(in_channels, out_channels, time_channels, condition_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.conv2 = GCNConvWithTimeAndCondition(out_channels, out_channels, time_channels, condition_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection handling
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t_emb, c_emb, edge_index):
        # First conv block
        h = self.norm1(x)
        h = self.conv1(h, t_emb, c_emb, edge_index)
        h = F.silu(h)
        h = self.dropout(h)
        
        # Second conv block
        h = self.norm2(h)
        h = self.conv2(h, t_emb, c_emb, edge_index)
        h = F.silu(h)
        h = self.dropout(h)
        
        # Skip connection
        return h + self.skip(x)

class GCNConvWithTimeAndCondition(nn.Module):
    """
    Graph convolution layer with time and condition conditioning
    """
    def __init__(self, in_channels, out_channels, time_channels, condition_channels):
        super(GCNConvWithTimeAndCondition, self).__init__()
        from torch_geometric.nn import GCNConv
        self.conv = GCNConv(in_channels, out_channels)
        self.time_proj = nn.Linear(time_channels, out_channels)
        self.condition_proj = nn.Linear(condition_channels, out_channels)
    
    def forward(self, x, t_emb, c_emb, edge_index):
        h = self.conv(x, edge_index)
        time_emb = self.time_proj(t_emb)
        cond_emb = self.condition_proj(c_emb)
        
        # Add time and condition embeddings to each node
        if len(time_emb.shape) == 2:
            time_emb = time_emb.unsqueeze(1)
        if len(cond_emb.shape) == 2:
            cond_emb = cond_emb.unsqueeze(1)
            
        return h + time_emb + cond_emb
