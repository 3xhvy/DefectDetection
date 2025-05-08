"""
Utility Functions for Graph Diffusion Models

This module provides utility functions for training, evaluation, and 
using graph diffusion models for vulnerability detection and repair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_networkx, to_networkx
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os

# Import from our project
from graph_diffusion_model import GraphDiffusionModel
from conditional_graph_diffusion import ConditionalGraphDiffusionModel
from improved_gnn_model import build_enhanced_graph_from_code, enrich_node_features_advanced

# Set up logging
logger = logging.getLogger(__name__)

def train_diffusion_model(model, train_loader, optimizer, device, epochs=100, 
                          conditional=False, save_path=None, eval_loader=None,
                          scheduler=None, log_interval=10):
    """
    Train the diffusion model
    
    Args:
        model: The diffusion model to train
        train_loader: DataLoader containing training samples
        optimizer: Optimizer for training
        device: Device to train on (cuda/cpu)
        epochs: Number of training epochs
        conditional: Whether the model is conditional
        save_path: Path to save model checkpoints
        eval_loader: DataLoader for evaluation during training
        scheduler: Learning rate scheduler
        log_interval: How often to log progress
        
    Returns:
        Trained model and dictionary of training metrics
    """
    model.train()
    metrics = {'train_loss': [], 'eval_loss': []}
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        # Training loop
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if conditional:
                # For conditional model, we need to extract condition from the batch
                # This depends on how you structure your data
                if hasattr(batch, 'condition'):
                    condition = batch.condition
                else:
                    # If no explicit condition, use a default (e.g., binary vulnerability label)
                    condition = batch.y.float()
                
                loss = model.loss_function_conditional(batch.x, batch.edge_index, condition, batch.batch)
            else:
                loss = model.loss_function(batch.x, batch.edge_index, batch.batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
        
        # Calculate average loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        logger.info(f"Epoch {epoch+1}/{epochs} - Average Train Loss: {avg_train_loss:.6f}")
        
        # Evaluation if eval_loader is provided
        if eval_loader is not None:
            eval_loss = evaluate_diffusion_model(model, eval_loader, device, conditional)
            metrics['eval_loss'].append(eval_loss)
            logger.info(f"Epoch {epoch+1}/{epochs} - Evaluation Loss: {eval_loss:.6f}")
        
        # Save model checkpoint
        if save_path is not None and (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_path, f"diffusion_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, checkpoint_path)
            logger.info(f"Model checkpoint saved to {checkpoint_path}")
    
    # Save final model
    if save_path is not None:
        final_path = os.path.join(save_path, "diffusion_model_final.pt")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': metrics['train_loss'][-1],
        }, final_path)
        logger.info(f"Final model saved to {final_path}")
    
    return model, metrics

def evaluate_diffusion_model(model, eval_loader, device, conditional=False):
    """
    Evaluate the diffusion model on a validation/test set
    
    Args:
        model: The diffusion model to evaluate
        eval_loader: DataLoader containing evaluation samples
        device: Device to evaluate on (cuda/cpu)
        conditional: Whether the model is conditional
        
    Returns:
        Average loss on the evaluation set
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = batch.to(device)
            
            if conditional:
                if hasattr(batch, 'condition'):
                    condition = batch.condition
                else:
                    condition = batch.y.float()
                
                loss = model.loss_function_conditional(batch.x, batch.edge_index, condition, batch.batch)
            else:
                loss = model.loss_function(batch.x, batch.edge_index, batch.batch)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(eval_loader)
    return avg_loss

def generate_repaired_code(model, vulnerable_graph, device, condition_type='secure'):
    """
    Generate repaired code from a vulnerable graph
    
    Args:
        model: Trained conditional diffusion model
        vulnerable_graph: Graph representation of vulnerable code
        device: Device to run generation on (cuda/cpu)
        condition_type: Type of condition to apply ('secure', 'optimized', etc.)
        
    Returns:
        Tuple of (repaired_code, repaired_graph)
    """
    # Convert graph to PyG data object if needed
    if isinstance(vulnerable_graph, nx.Graph):
        data = from_networkx(vulnerable_graph)
    else:
        data = vulnerable_graph
    
    # Move to device
    data = data.to(device)
    
    # Create condition based on condition_type
    if condition_type == 'secure':
        # For secure code, use a one-hot encoding or learned embedding
        condition = torch.zeros(data.num_nodes, model.condition_dim, device=device)
        condition[:, 0] = 1.0  # Assuming index 0 represents "secure"
    elif condition_type == 'optimized':
        condition = torch.zeros(data.num_nodes, model.condition_dim, device=device)
        condition[:, 1] = 1.0  # Assuming index 1 represents "optimized"
    else:
        # Default condition
        condition = torch.ones(data.num_nodes, model.condition_dim, device=device)
    
    # Generate repaired graph features
    repaired_features = model.sample_conditional(
        data.num_nodes, 
        data.edge_index, 
        condition, 
        batch=data.batch if hasattr(data, 'batch') else None,
        device=device
    )
    
    # Create a new graph with the repaired features
    repaired_graph = data.clone()
    repaired_graph.x = repaired_features
    
    # Convert back to code (this requires a separate function)
    repaired_code = graph_to_code(repaired_graph)
    
    return repaired_code, repaired_graph

def graph_to_code(graph_data):
    """
    Convert a graph back to code
    This is a placeholder - you'll need to implement this based on your AST representation
    
    Args:
        graph_data: PyG Data object representing code graph
        
    Returns:
        String of generated code
    """
    # Convert PyG data to NetworkX graph
    nx_graph = to_networkx(graph_data)
    
    # This is where you would implement the conversion from graph back to code
    # It depends on how your AST is structured
    
    # Placeholder
    code = "// Generated repaired code\n"
    code += "// This is a placeholder - implement graph to code conversion\n"
    
    return code

def visualize_graph_pair(original_graph, repaired_graph, title="Original vs Repaired Code Graph"):
    """
    Visualize the original and repaired code graphs side by side
    
    Args:
        original_graph: Original graph (PyG Data or NetworkX graph)
        repaired_graph: Repaired graph (PyG Data or NetworkX graph)
        title: Plot title
    """
    # Convert to NetworkX if needed
    if not isinstance(original_graph, nx.Graph):
        original_nx = to_networkx(original_graph)
    else:
        original_nx = original_graph
        
    if not isinstance(repaired_graph, nx.Graph):
        repaired_nx = to_networkx(repaired_graph)
    else:
        repaired_nx = repaired_graph
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original graph
    pos1 = nx.spring_layout(original_nx, seed=42)
    nx.draw(original_nx, pos1, ax=ax1, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold')
    ax1.set_title("Original Vulnerable Code Graph")
    
    # Plot repaired graph
    pos2 = nx.spring_layout(repaired_nx, seed=42)
    nx.draw(repaired_nx, pos2, ax=ax2, with_labels=True, node_color='lightgreen', 
            node_size=500, font_size=10, font_weight='bold')
    ax2.set_title("Repaired Code Graph")
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def prepare_vulnerability_dataset(code_samples, labels, language, batch_size=32, 
                                  train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                                  shuffle=True, num_workers=4):
    """
    Prepare dataset for training the diffusion model
    
    Args:
        code_samples: List of code samples
        labels: List of vulnerability labels (1 for vulnerable, 0 for secure)
        language: Tree-sitter language object
        batch_size: Batch size for DataLoader
        train_ratio, val_ratio, test_ratio: Dataset split ratios
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for DataLoader
        
    Returns:
        Dictionary containing train_loader, val_loader, test_loader
    """
    from torch_geometric.loader import DataLoader
    from torch.utils.data import random_split
    
    # Create a custom dataset
    class VulnerabilityGraphDataset(torch.utils.data.Dataset):
        def __init__(self, code_samples, labels, language):
            self.code_samples = code_samples
            self.labels = labels
            self.language = language
            
        def __len__(self):
            return len(self.code_samples)
        
        def __getitem__(self, idx):
            code = self.code_samples[idx]
            label = self.labels[idx]
            
            # Convert code to graph
            try:
                graph = build_enhanced_graph_from_code(code, self.language)
                graph_data = from_networkx(graph)
                
                # Enrich node features
                graph_data = enrich_node_features_advanced(graph_data, code)
                
                # Add label as graph attribute
                graph_data.y = torch.tensor([float(label)], dtype=torch.float)
                
                return graph_data
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {str(e)}")
                # Return a minimal fallback graph
                graph = nx.Graph()
                graph.add_node(0, type="fallback", start_byte=0, end_byte=0)
                graph_data = from_networkx(graph)
                graph_data.x = torch.tensor([[0.0]], dtype=torch.float)
                graph_data.y = torch.tensor([float(label)], dtype=torch.float)
                return graph_data
    
    # Create the dataset
    dataset = VulnerabilityGraphDataset(code_samples, labels, language)
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }

def create_diffusion_model(node_dim, condition_dim=None, hidden_dim=128, num_layers=4,
                          num_timesteps=1000, beta_schedule='linear', dropout=0.1,
                          device='cuda', conditional=False):
    """
    Create a diffusion model for code graphs
    
    Args:
        node_dim: Dimension of node features
        condition_dim: Dimension of condition (for conditional model)
        hidden_dim: Hidden dimension for the model
        num_layers: Number of layers in the U-Net
        num_timesteps: Number of diffusion timesteps
        beta_schedule: Schedule for noise level ('linear' or 'cosine')
        dropout: Dropout rate
        device: Device to create model on
        conditional: Whether to create a conditional model
        
    Returns:
        Created model
    """
    if conditional and condition_dim is not None:
        model = ConditionalGraphDiffusionModel(
            node_dim=node_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            dropout=dropout
        ).to(device)
    else:
        model = GraphDiffusionModel(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            dropout=dropout
        ).to(device)
    
    return model
