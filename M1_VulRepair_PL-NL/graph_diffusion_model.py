"""
Graph Diffusion Model for Code Vulnerability Detection and Repair

This module implements a diffusion-based generative model for code graphs,
which can be used for:
1. Generating synthetic vulnerable code examples
2. Providing vulnerability repair suggestions
3. Improving explainability of vulnerability detection

The implementation is based on denoising diffusion probabilistic models (DDPM)
adapted for graph-structured data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.nn import GCNConv, GATConv
import logging

# Set up logging
logger = logging.getLogger(__name__)

class GraphDiffusionModel(nn.Module):
    """
    Diffusion model for code graph generation and transformation.
    """
    def __init__(
        self,
        node_dim,           # Dimension of node features
        edge_dim=None,      # Dimension of edge features (if any)
        hidden_dim=128,     # Hidden dimension for the denoising network
        num_layers=6,       # Number of layers in the denoising network
        num_timesteps=1000, # Number of diffusion timesteps
        beta_schedule='linear', # Schedule for noise level
        dropout=0.1,        # Dropout rate
    ):
        super(GraphDiffusionModel, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Set up noise schedule (beta values)
        if beta_schedule == 'linear':
            self.betas = torch.linspace(1e-4, 0.02, num_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        
        # Calculate diffusion parameters
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # Build the denoising network (U-Net style for graphs)
        self.denoiser = GraphUNet(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward_diffusion(self, x_0, t):
        """
        Forward diffusion process: q(x_t | x_0)
        Gradually adds noise to the input graph features
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Mean + variance
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def reverse_diffusion_step(self, x_t, t, edge_index, batch=None):
        """
        Single step of reverse diffusion: p(x_{t-1} | x_t)
        """
        # Predict the noise component
        predicted_noise = self.denoiser(x_t, t, edge_index, batch)
        
        # Get diffusion parameters for this timestep
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod_prev[t]
        beta = self.betas[t]
        
        # Compute the mean for p(x_{t-1} | x_t)
        coef1 = torch.sqrt(alpha_cumprod_prev) / torch.sqrt(1 - alpha_cumprod)
        coef2 = torch.sqrt(1 - alpha_cumprod_prev) - torch.sqrt(alpha_cumprod_prev) * torch.sqrt(1 - alpha_cumprod) / torch.sqrt(alpha_cumprod)
        mean = coef1 * (x_t - coef2 * predicted_noise)
        
        # Add noise for sampling, only if t > 0
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod))
            x_t_minus_1 = mean + sigma * noise
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    def sample(self, num_nodes, edge_index, batch=None, device='cuda'):
        """
        Sample a graph from the diffusion model
        """
        # Start from pure noise
        x_T = torch.randn(num_nodes, self.node_dim, device=device)
        
        # Iteratively denoise
        x_t = x_T
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_nodes,), t, device=device, dtype=torch.long)
            x_t = self.reverse_diffusion_step(x_t, t_tensor, edge_index, batch)
        
        return x_t
    
    def loss_function(self, x_0, edge_index, batch=None):
        """
        Calculate the diffusion loss
        """
        batch_size = x_0.shape[0]
        
        # Sample timestep uniformly
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)
        
        # Forward diffusion to get x_t and the noise
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # Predict the noise component
        predicted_noise = self.denoiser(x_t, t, edge_index, batch)
        
        # Calculate loss (typically MSE between actual and predicted noise)
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

class GraphUNet(nn.Module):
    """
    U-Net style architecture for graph denoising
    """
    def __init__(self, node_dim, edge_dim=None, hidden_dim=128, num_layers=4, dropout=0.1):
        super(GraphUNet, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initial projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        # Down path (encoder)
        self.down_layers = nn.ModuleList()
        for i in range(num_layers):
            self.down_layers.append(
                GraphConvBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    time_channels=hidden_dim,
                    dropout=dropout
                )
            )
        
        # Middle
        self.middle = GraphConvBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            time_channels=hidden_dim,
            dropout=dropout
        )
        
        # Up path (decoder) with skip connections
        self.up_layers = nn.ModuleList()
        for i in range(num_layers):
            self.up_layers.append(
                GraphConvBlock(
                    in_channels=2 * hidden_dim,  # Double for skip connection
                    out_channels=hidden_dim,
                    time_channels=hidden_dim,
                    dropout=dropout
                )
            )
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, node_dim)
    
    def forward(self, x, t, edge_index, batch=None):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Initial projection
        h = self.input_proj(x)
        
        # Down path with skip connections
        skips = [h]
        for layer in self.down_layers:
            h = layer(h, t_emb, edge_index)
            skips.append(h)
        
        # Middle
        h = self.middle(h, t_emb, edge_index)
        
        # Up path with skip connections
        for layer, skip in zip(self.up_layers, reversed(skips)):
            h = torch.cat([h, skip], dim=-1)
            h = layer(h, t_emb, edge_index)
        
        # Final projection
        output = self.output_proj(h)
        
        return output

class GraphConvBlock(nn.Module):
    """
    Graph convolution block with time conditioning
    """
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super(GraphConvBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.conv1 = GCNConvWithTime(in_channels, out_channels, time_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.conv2 = GCNConvWithTime(out_channels, out_channels, time_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection handling
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, t_emb, edge_index):
        # First conv block
        h = self.norm1(x)
        h = self.conv1(h, t_emb, edge_index)
        h = F.silu(h)
        h = self.dropout(h)
        
        # Second conv block
        h = self.norm2(h)
        h = self.conv2(h, t_emb, edge_index)
        h = F.silu(h)
        h = self.dropout(h)
        
        # Skip connection
        return h + self.skip(x)

class GCNConvWithTime(nn.Module):
    """
    Graph convolution layer with time conditioning
    """
    def __init__(self, in_channels, out_channels, time_channels):
        super(GCNConvWithTime, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.time_proj = nn.Linear(time_channels, out_channels)
    
    def forward(self, x, t_emb, edge_index):
        h = self.conv(x, edge_index)
        time_emb = self.time_proj(t_emb)
        # Add time embedding to each node
        return h + time_emb.unsqueeze(1) if len(time_emb.shape) == 2 else h + time_emb
