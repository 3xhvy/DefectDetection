import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import argparse
import logging
import os
import random
from tqdm import tqdm
import pandas as pd
import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from improved_gnn_model import ImprovedGNNModel, build_enhanced_graph_from_code, enrich_node_features_advanced, FocalLoss, MemoryEfficientTrainer

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Dataset Class
class EnhancedCodeDefectDataset(Dataset):
    def __init__(self, code_samples, labels, language, args):
        super().__init__()
        self.code_samples = code_samples
        self.labels = [self._clean_label(l) for l in labels]  # Clean labels
        self.language = language
        self.args = args
        
        # Print class distribution for debugging
        label_counts = {}
        for label in self.labels:
            label_counts[float(label)] = label_counts.get(float(label), 0) + 1
        logger.info(f"Dataset created with label distribution: {label_counts}")
    
    def _clean_label(self, label):
        """Convert label to 0 or 1, handling special cases"""
        if isinstance(label, (int, float)):
            return float(label)
        # Handle string labels
        label = str(label).strip().lower()
        if label in ('1', 'true', 't', 'yes', 'y', 'positive'):
            return 1.0
        elif label in ('0', 'false', 'f', 'no', 'n', 'negative'):
            return 0.0
        else:
            return 0.0  # Default to negative class for unparseable labels
    
    def len(self):
        return len(self.code_samples)
    
    def get(self, idx):
        code = self.code_samples[idx]
        try:
            # Use enhanced graph construction
            graph_data = build_enhanced_graph_from_code(code, self.language)
            
            # Use advanced feature enrichment
            graph_data = enrich_node_features_advanced(graph_data, code)
            
            # Ensure we have a valid graph
            if not hasattr(graph_data, 'x') or graph_data.x is None or graph_data.x.shape[0] == 0:
                # Create fallback features
                graph_data.x = torch.zeros((1, 26), dtype=torch.float)
            
            # Add label
            graph_data.y = torch.tensor([float(self.labels[idx])], dtype=torch.float)
            return graph_data
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {str(e)}")
            # Fallback graph with minimal features
            import networkx as nx
            from torch_geometric.utils import from_networkx
            graph = nx.DiGraph()
            graph.add_node(0, type="fallback")
            graph_data = from_networkx(graph)
            graph_data.x = torch.zeros((1, 26), dtype=torch.float)
            graph_data.y = torch.tensor([float(self.labels[idx])], dtype=torch.float)
            return graph_data

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Apply CUDA memory optimizations
def optimize_cuda_memory():
    """Apply memory optimizations for CUDA devices"""
    # Empty CUDA cache
    torch.cuda.empty_cache()
    
    # Garbage collection
    import gc
    gc.collect()
    
    # Enable memory-efficient attention if PyTorch version supports it
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except:
        logger.info("Memory efficient SDP not available in this PyTorch version")
    
    # Set to deterministic mode for consistent memory usage
    torch.backends.cudnn.deterministic = True
    
    # Disable benchmarking for more stable memory usage
    torch.backends.cudnn.benchmark = False
    
    logger.info("Applied CUDA memory optimizations")

# Plot training history
def plot_training_history(history, output_dir):
    """Plot and save training metrics history"""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    
    for metric in metrics:
        if f'train_{metric}' in history and f'eval_{metric}' in history:
            plt.figure(figsize=(10, 6))
            plt.plot(history[f'train_{metric}'], label=f'Train {metric}')
            plt.plot(history[f'eval_{metric}'], label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} over epochs')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{metric}_history.png'))
            plt.close()

def main():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument("--hidden_channels", default=64, type=int)  # Reduced from 128
    parser.add_argument("--out_channels", default=32, type=int)     # Reduced from 64
    parser.add_argument("--num_layers", default=2, type=int)        # Kept at 2
    parser.add_argument("--heads", default=2, type=int, help="Number of attention heads in GAT layers")
    
    # Training parameters
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--train_batch_size", default=8, type=int)  # Reduced from 16
    parser.add_argument("--eval_batch_size", default=16, type=int)  # Reduced from 32
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--warmup_steps", default=0.1, type=float, help="Portion of training steps for LR warmup")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./improved_gnn_checkpoints", type=str)
    parser.add_argument("--focal_loss", action='store_true', help="Use focal loss instead of BCE")
    parser.add_argument("--focal_gamma", default=2.0, type=float, help="Gamma parameter for focal loss")
    parser.add_argument("--mixed_precision", action='store_true', help="Use mixed precision training")
    parser.add_argument("--scheduler", default="cosine", type=str, choices=["plateau", "cosine", "none"])
    parser.add_argument("--early_stopping", action='store_true', help="Use early stopping")
    parser.add_argument("--patience", default=10, type=int, help="Patience for early stopping")
    parser.add_argument("--use_checkpointing", action='store_true', help="Use gradient checkpointing to save memory")
    parser.add_argument("--memory_efficient", action='store_true', help="Use memory-efficient model settings")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info("device: %s", device)
    
    # Apply memory optimizations for CUDA
    if torch.cuda.is_available():
        optimize_cuda_memory()
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load C language for parsing
    try:
        from tree_sitter import Language
        language = Language('/home/hohoanghvy/tree-sitter-container/build/my-languages.so', 'c')
    except Exception as e:
        logger.error(f"Failed to load C language: {str(e)}")
        return
    
    # Training
    if args.do_train:
        # Load both train and test splits from HuggingFace dataset
        logger.info("Loading training and test datasets for custom split...")
        train_data_whole = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="train")
        test_data_whole = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="test")
        
        # Combine both splits
        df = pd.DataFrame({
            "code": train_data_whole["func"] + test_data_whole["func"],
            "label": train_data_whole["target"] + test_data_whole["target"]
        })
        logger.info("Total samples after combining: %d", len(df))
        
        # Stratified split: train/val/test (80/10/10)
        train_data, temp_data = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df["label"])
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed, stratify=temp_data["label"])
        
        # Log distributions
        for name, data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            logger.info("%s set: %d samples, label distribution: %s", name, len(data), data["label"].value_counts().to_dict())
        
        # Create datasets with enhanced features
        train_dataset = EnhancedCodeDefectDataset(
            train_data["code"].tolist(),
            train_data["label"].tolist(),
            language,
            args
        )
        eval_dataset = EnhancedCodeDefectDataset(
            val_data["code"].tolist(),
            val_data["label"].tolist(),
            language,
            args
        )
        test_dataset = EnhancedCodeDefectDataset(
            test_data["code"].tolist(),
            test_data["label"].tolist(),
            language,
            args
        )
        
        # Create dataloaders with optimized settings for memory efficiency
        train_loader = DataLoader(train_dataset,
                                batch_size=args.train_batch_size,
                                shuffle=True,
                                num_workers=2,  # Reduced from 4
                                pin_memory=True,
                                prefetch_factor=2,
                                persistent_workers=False)  # Changed from True to save memory
        eval_loader = DataLoader(eval_dataset,
                               batch_size=args.eval_batch_size,
                               shuffle=False,
                               num_workers=2,  # Reduced from 4
                               pin_memory=True,
                               prefetch_factor=2)
        test_loader = DataLoader(test_dataset,
                               batch_size=args.eval_batch_size,
                               shuffle=False,
                               num_workers=2,  # Reduced from 4
                               pin_memory=True,
                               prefetch_factor=2)
        
        # Initialize improved model with memory-efficient settings
        model = ImprovedGNNModel(
            in_channels=26,  # Feature dimension
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
            dropout=0.3,
            heads=args.heads,
            use_checkpointing=args.use_checkpointing,
            use_mixed_precision=args.mixed_precision,
            memory_efficient=args.memory_efficient
        ).to(device)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} total parameters, {trainable_params:,} trainable")
        
        # Initialize optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)
        
        # Choose loss function
        if args.focal_loss:
            # Calculate class weights for focal loss
            class_counts = train_data['label'].value_counts().to_dict()
            total_samples = len(train_data)
            weight_0 = total_samples / (2 * class_counts.get(0, 1))
            weight_1 = total_samples / (2 * class_counts.get(1, 1))
            alpha = weight_1 / (weight_0 + weight_1)  # Normalized weight for positive class
            
            logger.info(f"Using Focal Loss with alpha={alpha:.4f}, gamma={args.focal_gamma}")
            criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma)
        else:
            # Use weighted BCE loss
            class_counts = train_data['label'].value_counts().to_dict()
            if len(class_counts) > 1:
                total_samples = len(train_data)
                weight_0 = total_samples / (2 * class_counts.get(0, 1))
                weight_1 = total_samples / (2 * class_counts.get(1, 1))
                pos_weight = torch.tensor([weight_1 / weight_0]).to(device)
                logger.info(f"Using weighted BCE loss with pos_weight={pos_weight.item():.4f}")
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                criterion = nn.BCEWithLogitsLoss()
        
        # Initialize learning rate scheduler
        if args.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        elif args.scheduler == "cosine":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        else:
            scheduler = None
        
        # Create memory-efficient trainer
        trainer = MemoryEfficientTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=criterion,
            device=device,
            use_mixed_precision=args.mixed_precision
        )
        
        # Early stopping variables
        best_f1 = 0
        best_model_path = os.path.join(args.output_dir, 'model_best.pt')
        epochs_no_improve = 0
        
        # Training history
        history = {
            'train_loss': [],
            'eval_loss': [],
            'eval_accuracy': [],
            'eval_precision': [],
            'eval_recall': [],
            'eval_f1': [],
            'eval_roc_auc': [],
            'eval_pr_auc': []
        }
        
        # Training loop
        for epoch in range(args.epochs):
            logger.info(f"--------------------------EPOCH {epoch+1}/{args.epochs}-----------------------------------")
            
            # Train phase
            model.train()
            train_loss = 0
            all_preds = []
            all_labels = []
            
            # Use tqdm for progress bar
            progress_bar = tqdm(train_loader, desc="Training")
            
            for batch_idx, data in enumerate(progress_bar):
                # Train batch with memory-efficient trainer
                batch_loss = trainer.train_batch(data)
                train_loss += batch_loss
                
                # Log progress periodically
                if (batch_idx + 1) % 10 == 0:
                    # Check memory usage
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                        mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                        logger.info(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {batch_loss:.4f} | GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Evaluate phase
            val_loss, val_acc = trainer.evaluate(eval_loader)
            
            # Calculate detailed metrics for evaluation set
            model.eval()
            all_preds = []
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for data in tqdm(eval_loader, desc="Evaluating"):
                    data = data.to(device)
                    
                    if args.mixed_precision:
                        with torch.cuda.amp.autocast():
                            logits = model(data)
                    else:
                        logits = model(data)
                    
                    # Get probabilities and predictions
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(float)
                    labels = data.y.cpu().numpy()
                    
                    all_probs.extend(probs)
                    all_preds.extend(preds)
                    all_labels.extend(labels)
            
            # Calculate metrics
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            
            eval_metrics = {
                'accuracy': accuracy_score(all_labels, all_preds),
                'precision': precision_score(all_labels, all_preds, zero_division=0),
                'recall': recall_score(all_labels, all_preds, zero_division=0),
                'f1': f1_score(all_labels, all_preds, zero_division=0),
                'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
                'pr_auc': average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
            }
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['eval_loss'].append(val_loss)
            history['eval_accuracy'].append(eval_metrics['accuracy'])
            history['eval_precision'].append(eval_metrics['precision'])
            history['eval_recall'].append(eval_metrics['recall'])
            history['eval_f1'].append(eval_metrics['f1'])
            history['eval_roc_auc'].append(eval_metrics['roc_auc'])
            history['eval_pr_auc'].append(eval_metrics['pr_auc'])
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{args.epochs}:")
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Eval Loss: {val_loss:.4f}")
            logger.info(f"Eval Accuracy: {eval_metrics['accuracy']:.4f}")
            logger.info(f"Eval Precision: {eval_metrics['precision']:.4f}")
            logger.info(f"Eval Recall: {eval_metrics['recall']:.4f}")
            logger.info(f"Eval F1: {eval_metrics['f1']:.4f}")
            logger.info(f"Eval ROC AUC: {eval_metrics['roc_auc']:.4f}")
            logger.info(f"Eval PR AUC: {eval_metrics['pr_auc']:.4f}")
            
            # Update scheduler
            if scheduler is not None:
                if args.scheduler == "plateau":
                    scheduler.step(eval_metrics['f1'])
                else:
                    scheduler.step()
            
            # Save best model
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved with F1: {best_f1:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement for {epochs_no_improve} epochs (best F1: {best_f1:.4f})")
            
            # Early stopping
            if args.early_stopping and epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Free up memory after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        # Plot training history
        plot_training_history(history, args.output_dir)
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, 'model_final.pt')
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save lightweight model for inference
        try:
            model_scripted = torch.jit.script(model)
            model_scripted.save(os.path.join(args.output_dir, 'model_scripted.pt'))
            logger.info(f"Saved scripted model for inference to {os.path.join(args.output_dir, 'model_scripted.pt')}")
        except Exception as e:
            logger.warning(f"Failed to save scripted model: {str(e)}")
        
        # Load best model for testing
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}")
        
        # Test best model
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        # Calculate detailed metrics for test set
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Testing"):
                data = data.to(device)
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = model(data)
                else:
                    logits = model(data)
                
                # Get probabilities and predictions
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(float)
                labels = data.y.cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        test_metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
            'pr_auc': average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        }
        
        logger.info("Test Results with best model:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        logger.info(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"  PR AUC: {test_metrics['pr_auc']:.4f}")
    
    # Testing only
    elif args.do_test:
        # Load test dataset
        logger.info("Loading test dataset...")
        test_data_hf = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="test")
        test_dataset = EnhancedCodeDefectDataset(
            test_data_hf["func"],
            test_data_hf["target"],
            language,
            args
        )
        
        test_loader = DataLoader(test_dataset,
                               batch_size=args.eval_batch_size,
                               shuffle=False,
                               num_workers=2,  # Reduced from 4
                               pin_memory=True)
        
        # Initialize model with memory-efficient settings
        model = ImprovedGNNModel(
            in_channels=26,  # Feature dimension
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_layers=args.num_layers,
            dropout=0.3,
            heads=args.heads,
            use_checkpointing=args.use_checkpointing,
            use_mixed_precision=args.mixed_precision,
            memory_efficient=args.memory_efficient
        ).to(device)
        
        # Load best model
        model_path = os.path.join(args.output_dir, 'model_best.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.error(f"No saved model found at {model_path}")
            return
        
        # Use standard BCE loss for evaluation
        criterion = nn.BCEWithLogitsLoss()
        
        # Create memory-efficient trainer for evaluation
        trainer = MemoryEfficientTrainer(
            model=model,
            optimizer=None,  # Not needed for evaluation
            loss_fn=criterion,
            device=device,
            use_mixed_precision=args.mixed_precision
        )
        
        # Evaluate on test set
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        # Calculate detailed metrics
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Testing"):
                data = data.to(device)
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = model(data)
                else:
                    logits = model(data)
                
                # Get probabilities and predictions
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(float)
                labels = data.y.cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        test_metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
            'pr_auc': average_precision_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        }
        
        logger.info("Test Results:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        logger.info(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"  PR AUC: {test_metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()
