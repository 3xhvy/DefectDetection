"""train_diffusion_model.py
Diffusion Model Implementation for Code Vulnerability Repair
Integrated with existing GNN codebase
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.loader import DataLoader
import argparse
import logging
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import datasets
from sklearn.model_selection import train_test_split
from tree_sitter import Language, Parser

# Reuse existing components from gnn_main.py
from gnn_main import (
    build_graph_from_code,
    enrich_node_features,
    CodeDefectDataset,
    set_seed,
    print_dataset_stats,
    print_sample_entries,
    print_graph_sample
)

logger = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Input is (batch_size, 1) or (1,)
        # Output is (batch_size, dim) or (1, dim)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Handle different input shapes
        if len(time.shape) == 0:  # scalar
            time = time.unsqueeze(0)  # (1,)
        elif len(time.shape) > 1 and time.shape[1] == 1:
            # If time is [batch_size, 1], flatten it to [batch_size]
            time = time.squeeze(-1)
        
        # Ensure time is float for multiplication
        time = time.float()
        
        # Create embeddings
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)  # (batch, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Pad if dim is odd
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1, 0, 0))
            
        return embeddings


class CodeDiffusionModel(nn.Module):
    """Diffusion model aligned with existing GNN architecture"""
    def __init__(self, node_dim, hidden_dim=256, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Noise schedule parameters - register as buffers so they're moved to the correct device
        # Using a cosine schedule which often works better than linear
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = torch.minimum(torch.ones(num_timesteps), 1 - alpha_bar[1:] / alpha_bar[:-1])
        
        # Clip betas to reasonable range
        betas = torch.clip(betas, 1e-5, 0.999)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # Shared encoder with GNN model architecture
        self.convs = nn.ModuleList([
            GCNConv(node_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])

        # Enhanced time embedding layer with sinusoidal embeddings
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, node_dim)  # Match node feature dimension
        )

        # Decoder layers
        self.decoder = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, node_dim)
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        ])

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, t):
        # Time embedding - ensure t is properly formatted
        # If t is a scalar, convert to tensor
        if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            t = torch.tensor([t], device=x.device)
            
        # Process time embedding
        t_embed = self.time_embed(t)
        
        # Process through encoder
        h = x
        for i, (conv, bn) in enumerate(zip(self.convs, self.bn_layers)):
            # Apply GCN convolution
            h = conv(h, edge_index)
            
            # Apply batch normalization
            h = bn(h)
            
            # Add time embedding to each node
            # Ensure t_embed has the right shape for all nodes
            if t_embed.size(0) == 1 and h.size(0) > 1:
                # If we have a single time embedding but multiple nodes,
                # expand the time embedding to match the number of nodes
                expanded_t_embed = t_embed.expand(h.size(0), -1)
            
            # Add time embedding (ensure dimensions match)
            if h.size(1) == expanded_t_embed.size(1):
                h = h + expanded_t_embed
            else:
                # If dimensions don't match, project time embedding to match h
                time_projection = nn.Linear(expanded_t_embed.size(1), h.size(1)).to(h.device)
                h = h + time_projection(expanded_t_embed)
            
            h = F.silu(h)
            h = self.dropout(h)

        # Process through decoder
        for i, conv in enumerate(self.decoder):
            h = conv(h, edge_index)
            if i < len(self.decoder) - 1:  # Apply activation to all but last layer
                h = F.silu(h)

        return h

    def diffuse(self, x_0, t):
        """Apply forward diffusion"""
        # Ensure all tensors are on the same device
        device = x_0.device
        
        # Generate noise on the same device as x_0
        noise = torch.randn_like(x_0)
        
        # Handle single timestep case
        if t.size(0) == 1 and x_0.size(0) > 1:
            # If t is a single value but we have multiple nodes,
            # use the same timestep for all nodes
            t_idx = t[0]
            sqrt_alpha = torch.sqrt(self.alphas_cumprod[t_idx]).unsqueeze(0).expand(x_0.size(0), 1)
            sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas_cumprod[t_idx]).unsqueeze(0).expand(x_0.size(0), 1)
        else:
            # Normal case - get alphas_cumprod for the timesteps t
            sqrt_alpha = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1)
            sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1)
        
        # Apply diffusion
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

def train_diffusion(args, model, train_loader, optimizer, device):
    """Training loop aligned with existing GNN training"""
    model.train()
    total_loss = 0
    batch_count = 0
    
    # Enable gradient accumulation to reduce memory usage
    accumulation_steps = args.gradient_accumulation_steps

    # Only zero gradients at the beginning
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        try:
            # Move batch to device and immediately clear unnecessary data
            batch = batch.to(device)
            
            # Sample a single random timestep for the entire batch
            t = torch.randint(0, model.num_timesteps, (1,), device=device).item()
            
            # Apply diffusion - use the same timestep for all nodes
            t_expanded = torch.full((batch.num_nodes,), t, device=device)
            x_noisy, noise = model.diffuse(batch.x, t_expanded)
            
            # Free memory by deleting unnecessary tensors
            del t_expanded
            
            # Predict noise - use scalar t for simplicity
            pred_noise = model(x_noisy, batch.edge_index, t)

            # Calculate loss and scale it for gradient accumulation
            loss = F.mse_loss(pred_noise, noise) / accumulation_steps
            
            # Backprop
            loss.backward()
            
            # Free memory
            del x_noisy, noise, pred_noise
            
            # Only update weights after accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Track loss (use the unscaled loss for reporting)
            total_loss += loss.item() * accumulation_steps
            batch_count += 1
            
        except RuntimeError as e:
            # Handle out of memory errors gracefully
            if 'out of memory' in str(e):
                # Clear cache and skip this batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.warning(f"Skipping batch {batch_idx} due to OOM error: {str(e)}")
                # Reset gradients
                optimizer.zero_grad()
                continue
            else:
                # Re-raise other runtime errors
                raise e

    # Avoid division by zero if all batches were skipped due to OOM
    return total_loss / max(batch_count, 1)

def evaluate_diffusion(args, model, eval_loader, device):
    """Evaluation loop for diffusion model"""
    model.eval()
    total_loss = 0
    batch_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            try:
                batch = batch.to(device)

                # Sample a single random timestep for the entire batch
                t = torch.randint(0, model.num_timesteps, (1,), device=device).item()
                
                # Apply diffusion - use the same timestep for all nodes
                t_expanded = torch.full((batch.num_nodes,), t, device=device)
                x_noisy, noise = model.diffuse(batch.x, t_expanded)
                
                # Free memory
                del t_expanded

                # Predict noise - pass single timestep to model
                pred_noise = model(x_noisy, batch.edge_index, t)
                
                # Calculate loss
                loss = F.mse_loss(pred_noise, noise)
                total_loss += loss.item()
                batch_count += 1
                
                # Free memory
                del x_noisy, noise, pred_noise, batch
                
                # Clear cache periodically
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                # Handle out of memory errors gracefully
                if 'out of memory' in str(e) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.warning(f"Skipping batch {batch_idx} in evaluation due to OOM error")
                    continue
                else:
                    # Re-raise other runtime errors
                    raise e

    # Avoid division by zero if all batches were skipped due to OOM
    return total_loss / max(batch_count, 1)

def graph_to_code(graph_data, original_code=None):
    """Convert graph back to code - placeholder implementation"""
    # This is a placeholder function that would need to be implemented
    # based on your specific AST structure and code generation approach
    
    # For demonstration purposes, return a simple fixed code
    if original_code:
        # In a real implementation, you would modify the original code
        # based on the graph changes
        return original_code.replace("strcpy(buffer, input)", 
                                    "strncpy(buffer, input, sizeof(buffer) - 1); buffer[sizeof(buffer) - 1] = '\\0';")
    else:
        return "void fixed_func() {\n  char buffer[10];\n  strncpy(buffer, input, sizeof(buffer) - 1);\n  buffer[sizeof(buffer) - 1] = '\\0';\n}"

def generate_repair(model, vulnerable_code, language, device, num_timesteps):
    """Generate repaired code using diffusion process"""
    # Create graph from vulnerable code
    try:
        graph = build_graph_from_code(vulnerable_code, language)
        graph_data = enrich_node_features(graph)
        
        # Move to device
        graph_data = graph_data.to(device)
        
        # Start with random noise
        x_t = torch.randn_like(graph_data.x)
        
        # Reverse diffusion process (denoising)
        for t in tqdm(reversed(range(num_timesteps)), desc="Generating repair"):
            # Use scalar t for simplicity
            # The model will handle converting it to the proper tensor format
            
            # Get noise prediction
            with torch.no_grad():
                pred_noise = model(x_t, graph_data.edge_index, t)
            
            # One step of denoising
            alpha_t = model.alphas[t]
            alpha_cumprod_t = model.alphas_cumprod[t]
            beta_t = model.betas[t]
            
            # Add noise at each step except the last one
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
                
            # Update x_t using the denoising formula
            x_t = (1 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise
        
        # Convert back to code
        repaired_code = graph_to_code(x_t, vulnerable_code)
        return repaired_code
        
    except Exception as e:
        logger.error(f"Error in repair generation: {e}")
        import traceback
        traceback.print_exc()
        return "Error generating repair: " + str(e)

def run_demo(args, model, language, device):
    """Run a demonstration of the diffusion model"""
    # Sample vulnerable code
    vulnerable_code = """
    void copy_data(char *user_input) {
        char buffer[10];
        strcpy(buffer, user_input);  // Vulnerable: no bounds checking
        printf("Buffer contains: %s\\n", buffer);
    }
    """
    
    logger.info("Sample vulnerable code:")
    logger.info(vulnerable_code)
    
    try:
        # First create the graph to get the node feature dimension
        graph = build_graph_from_code(vulnerable_code, language)
        graph_data = enrich_node_features(graph)
        
        # Check the node feature dimension
        node_dim = graph_data.x.shape[1]
        logger.info(f"Node feature dimension: {node_dim}")
        
        # Reinitialize the model with the correct node dimension if needed
        if model.node_dim != node_dim:
            logger.info(f"Reinitializing model with correct node dimension: {node_dim}")
            model = CodeDiffusionModel(
                node_dim=node_dim,
                hidden_dim=args.hidden_dim,
                num_timesteps=args.num_timesteps
            ).to(device)
        
        # Generate repaired code
        logger.info("Generating repair...")
        repaired_code = generate_repair(model, vulnerable_code, language, device, args.num_timesteps)
        
        logger.info("\nGenerated repaired code:")
        logger.info(repaired_code)
    
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

def filter_large_graphs(dataset, max_nodes=1000):
    """Filter out graphs that are too large to avoid memory issues"""
    filtered_dataset = []
    for graph in dataset:
        if graph.num_nodes <= max_nodes:
            filtered_dataset.append(graph)
        else:
            logger.warning(f"Skipping graph with {graph.num_nodes} nodes (exceeds limit of {max_nodes})")
    
    logger.info(f"Filtered dataset: kept {len(filtered_dataset)}/{len(dataset)} graphs")
    return filtered_dataset

def main():
    parser = argparse.ArgumentParser(description="Code Repair Diffusion Model")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size (default: 128, use 256 if you have enough RAM)")
    parser.add_argument("--num_timesteps", type=int, default=250, help="Number of diffusion timesteps (lower for less memory, default: 250)")
    parser.add_argument("--num_layers", default=3, type=int, help="Number of GNN layers (default: 3)")
    
    # Training parameters
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--do_demo", action='store_true')
    parser.add_argument("--train_batch_size", default=16, type=int, help="Training batch size (default: 16)")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Evaluation batch size (default: 16)")
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", default=5e-3, type=float, help="Learning rate (default: 5e-3)")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay (default: 0.001)")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm for clipping (default: 1.0)")
    parser.add_argument("--epochs", default=20, type=int, help="Number of training epochs (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", default="./diffusion_checkpoints", type=str, help="Directory to save model checkpoints")
    parser.add_argument("--class_weights", action='store_true', help="Use class weights in loss function")
    parser.add_argument("--memory_efficient", action='store_true', help="Use memory efficient settings")
    parser.add_argument("--max_nodes", default=500, type=int, help="Maximum number of nodes in a graph (skip larger graphs)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    # Setup device
    device = torch.device(args.device)
    args.device = device
    logger.info(f"Using device: {device}")

    # Set seed
    set_seed(args.seed)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load C language for parsing
    try:
        language = Language('/home/hohoanghvy/tree-sitter-container/build/my-languages.so', 'c')
        logger.info("Successfully loaded C language")
    except Exception as e:
        logger.error(f"Failed to load C language: {e}")
        logger.info("Trying alternative language paths...")
        
        # Try alternative paths
        for lang_path in [
            '/home/hohoanghvy/PycharmProjects/DefectDetection/build/my-languages.so',
            '/home/hohoanghvy/PycharmProjects/GNN-CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU/parser/my-languages.so',
            os.path.expanduser('~/tree-sitter-languages/languages.so')
        ]:
            if os.path.exists(lang_path):
                try:
                    language = Language(lang_path, 'c')
                    logger.info(f"Successfully loaded C language from {lang_path}")
                    break
                except Exception:
                    continue
        else:
            logger.error("Could not load C language from any known path")
            # Create dummy language for demonstration
            language = None
            
            # Create dummy dataset for demonstration
            code_samples = [
                "void func() { char buffer[10]; strcpy(buffer, input); }",
                "void func() { char buffer[10]; strncpy(buffer, input, sizeof(buffer)); }"
            ] * 10
            labels = [1, 0] * 10  # 1 for vulnerable, 0 for secure
            
            # Create dummy model for demonstration
            model = CodeDiffusionModel(
                node_dim=10,  # Dummy dimension
                hidden_dim=args.hidden_dim,
                num_timesteps=args.num_timesteps
            ).to(device)
            
            logger.info("Using dummy model for demonstration")
            
            if args.do_demo:
                logger.info("Running demo with dummy model...")
                run_demo(args, model, language, device)
            
            return

    if args.do_train:
        # Load dataset similar to gnn_main.py
        logger.info("Loading training and test datasets...")
        try:
            train_data_whole = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="train")
            test_data_whole = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="test")

            # Combine both splits
            df = pd.DataFrame({
                "code": train_data_whole["func"] + test_data_whole["func"],
                "label": train_data_whole["target"] + test_data_whole["target"]
            })
            logger.info(f"Total samples after combining: {len(df)}")

            # Stratified split: train/val/test (80/10/10)
            train_data, temp_data = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df["label"])
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed, stratify=temp_data["label"])

            # Log distributions
            for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
                logger.info(f"{name} set: {len(data)} samples, label distribution: {data['label'].value_counts().to_dict()}")

            # Print dataset stats
            print_dataset_stats("Train", train_data)
            print_dataset_stats("Validation", val_data)
            print_sample_entries(train_data, n=2)

            # Create datasets
            train_dataset = CodeDefectDataset(
                train_data["code"].tolist(),
                train_data["label"].tolist(),
                language,
                args
            )
            eval_dataset = CodeDefectDataset(
                val_data["code"].tolist(),
                val_data["label"].tolist(),
                language,
                args
            )
            test_dataset = CodeDefectDataset(
                test_data["code"].tolist(),
                test_data["label"].tolist(),
                language,
                args
            )

            # Print graph sample
            print_graph_sample(train_dataset)

            # Create dataloaders with memory efficiency options
            if args.memory_efficient:
                # Use a custom collate function that skips graphs with too many nodes
                def memory_efficient_collate(batch):
                    # Filter out graphs that are too large
                    filtered_batch = [item for item in batch if item.num_nodes <= args.max_nodes]
                    if not filtered_batch:
                        # If all graphs were filtered out, return a small dummy graph
                        return next(iter(DataLoader(train_dataset, batch_size=1)))
                    return torch_geometric.data.Batch.from_data_list(filtered_batch)
                
                train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, 
                                         drop_last=False, collate_fn=memory_efficient_collate)
                eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, 
                                        drop_last=False, collate_fn=memory_efficient_collate)
                test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, 
                                       drop_last=False, collate_fn=memory_efficient_collate)
            else:
                train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=False)
                eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)
                test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

            # Get node feature dimension from the first sample
            sample_data = train_dataset[0]
            node_dim = sample_data.x.shape[1]
            logger.info(f"Node feature dimension: {node_dim}")

            # Initialize model with correct node dimension
            model = CodeDiffusionModel(
                node_dim=node_dim,
                hidden_dim=args.hidden_dim,
                num_timesteps=args.num_timesteps
            ).to(device)
            
            # Move model parameters to the correct device
            model.betas = model.betas.to(device)
            model.alphas = model.alphas.to(device)
            model.alphas_cumprod = model.alphas_cumprod.to(device)
            
            # Optimizer and scheduler
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )

            # Training loop
            best_val_loss = float('inf')
            for epoch in range(args.epochs):
                # Train
                train_loss = train_diffusion(args, model, train_loader, optimizer, device)
                logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.6f}")
                
                # Evaluate
                val_loss = evaluate_diffusion(args, model, eval_loader, device)
                logger.info(f"Epoch {epoch+1}/{args.epochs} - Validation Loss: {val_loss:.6f}")
                
                # Update scheduler
                scheduler.step(val_loss)
                
                # Save checkpoint if best
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    logger.info(f"Saved best model checkpoint to {checkpoint_path}")
            
            # Save final model
            final_path = os.path.join(args.output_dir, "final_model.pt")
            torch.save({
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, final_path)
            logger.info(f"Saved final model to {final_path}")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            import traceback
            traceback.print_exc()
            
            # Create dummy dataset for demonstration
            code_samples = [
                "void func() { char buffer[10]; strcpy(buffer, input); }",
                "void func() { char buffer[10]; strncpy(buffer, input, sizeof(buffer)); }"
            ] * 10
            labels = [1, 0] * 10
            
            train_dataset = CodeDefectDataset(code_samples, labels, language, args)
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            
            # Initialize model with dummy data
            model = CodeDiffusionModel(
                node_dim=10,  # Dummy dimension
                hidden_dim=args.hidden_dim,
                num_timesteps=args.num_timesteps
            ).to(device)
            
            logger.info("Using dummy model due to dataset loading error")
    
    if args.do_test:
        # Load dataset if not already loaded
        if 'test_dataset' not in locals() or 'test_loader' not in locals():
            logger.info("Loading test dataset...")
            try:
                train_data_whole = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="train")
                test_data_whole = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="test")

                # Combine both splits
                df = pd.DataFrame({
                    "code": train_data_whole["func"] + test_data_whole["func"],
                    "label": train_data_whole["target"] + test_data_whole["target"]
                })
                
                # Stratified split for test data
                _, temp_data = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df["label"])
                _, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed, stratify=temp_data["label"])
                
                # Create test dataset
                test_dataset = CodeDefectDataset(
                    test_data["code"].tolist(),
                    test_data["label"].tolist(),
                    language,
                    args
                )
                
                # Create test dataloader
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=args.eval_batch_size,
                    shuffle=False
                )
                
                # Get node feature dimension from the first sample
                node_dim = test_dataset[0].x.shape[1]
                logger.info(f"Node feature dimension: {node_dim}")
                
            except Exception as e:
                logger.error(f"Error loading test dataset: {e}")
                return
        
        # Load best model
        checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Initialize model with node dimension from test dataset
            model = CodeDiffusionModel(
                node_dim=node_dim,
                hidden_dim=args.hidden_dim,
                num_timesteps=args.num_timesteps
            ).to(device)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {checkpoint_path}")
            
            # Evaluate on test set
            test_loss = evaluate_diffusion(args, model, test_loader, device)
            logger.info(f"Test Loss: {test_loss:.6f}")
        else:
            logger.error(f"No model checkpoint found at {checkpoint_path}")
    
    if args.do_demo:
        # Load best model if available
        checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Initialize model
            try:
                model = CodeDiffusionModel(
                    node_dim=train_dataset[0].x.shape[1],
                    hidden_dim=args.hidden_dim,
                    num_timesteps=args.num_timesteps
                ).to(device)
                
                # Load state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from {checkpoint_path}")
            except:
                # If we can't determine node_dim from dataset
                model = CodeDiffusionModel(
                    node_dim=64,  # Default dimension
                    hidden_dim=args.hidden_dim,
                    num_timesteps=args.num_timesteps
                ).to(device)
                
                # Try to load state dict
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded model from {checkpoint_path}")
                except:
                    logger.warning("Could not load model state dict, using untrained model")
        else:
            # Initialize new model for demo
            logger.warning("No model checkpoint found, using untrained model for demo")
            model = CodeDiffusionModel(
                node_dim=64,  # Default dimension
                hidden_dim=args.hidden_dim,
                num_timesteps=args.num_timesteps
            ).to(device)
        
        # Run demo
        run_demo(args, model, language, device)

if __name__ == "__main__":
    main()
