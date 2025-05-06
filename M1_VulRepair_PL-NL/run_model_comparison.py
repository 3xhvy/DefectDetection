#!/usr/bin/env python3
import argparse
import logging
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Run model comparison for vulnerability detection")
    
    # Model selection
    parser.add_argument("--models", type=str, default="gnn", 
                      help="Comma-separated list of models to run: lr,rf,bilstm,transformer,gcn,graphsage,gat,improved_gnn,all,gnn")
    
    # Training parameters
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--output_dir", default="./comparison_results", type=str, help="Output directory for results")
    parser.add_argument("--class_weights", action='store_true', help="Use class weights in loss function")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "comparison.log")),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting model comparison")
    
    # Parse models to run
    models_to_run = args.models.lower().split(',')
    run_all = "all" in models_to_run
    run_gnn_only = "gnn" in models_to_run
    
    # Create a custom run_comparison function for GNN models only
    if run_gnn_only:
        logger.info("Running GNN models comparison only")
        # Import only what's needed for GNN models
        try:
            from gnn_comparison import run_gnn_comparison
            
            # Convert args to the format expected by run_gnn_comparison
            comparison_args = argparse.Namespace()
            comparison_args.hidden_channels = 64
            comparison_args.out_channels = 32
            comparison_args.num_layers = 2
            comparison_args.train_batch_size = args.batch_size
            comparison_args.eval_batch_size = args.batch_size
            comparison_args.learning_rate = args.learning_rate
            comparison_args.weight_decay = 0.001
            comparison_args.max_grad_norm = 1.0
            comparison_args.epochs = args.epochs
            comparison_args.seed = args.seed
            comparison_args.class_weights = args.class_weights
            comparison_args.output_dir = args.output_dir
            
            # Run GNN comparison
            run_gnn_comparison(comparison_args)
            
        except ImportError as e:
            logger.error(f"Error importing GNN comparison module: {e}")
            # Create the GNN comparison file
            create_gnn_comparison_file()
            logger.info("Created gnn_comparison.py. Please run again.")
            sys.exit(1)
    else:
        # Prepare command line arguments for model_comparison.py
        cmd_args = [
            "--train_batch_size", str(args.batch_size),
            "--eval_batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--epochs", str(args.epochs),
            "--seed", str(args.seed)
        ]
        
        # Add class weights if specified
        if args.class_weights:
            cmd_args.append("--class_weights")
        
        # Add model flags
        if run_all:
            cmd_args.append("--run_all")
        else:
            if "lr" in models_to_run or "logistic_regression" in models_to_run:
                cmd_args.extend(["--run_traditional_ml", "--run_logistic_regression"])
            
            if "rf" in models_to_run or "random_forest" in models_to_run:
                cmd_args.extend(["--run_traditional_ml", "--run_random_forest"])
            
            if "bilstm" in models_to_run:
                cmd_args.extend(["--run_sequence_models", "--run_bilstm"])
            
            if "transformer" in models_to_run:
                cmd_args.extend(["--run_sequence_models", "--run_transformer"])
            
            if "gcn" in models_to_run:
                cmd_args.extend(["--run_gnn_models", "--run_gcn"])
            
            if "graphsage" in models_to_run:
                cmd_args.extend(["--run_gnn_models", "--run_graphsage"])
            
            if "gat" in models_to_run:
                cmd_args.extend(["--run_gnn_models", "--run_gat"])
            
            if "improved_gnn" in models_to_run:
                cmd_args.extend(["--run_gnn_models", "--run_improved_gnn"])
        
        # Import and run the comparison
        try:
            from model_comparison import run_comparison
            
            # Convert cmd_args to a namespace
            comparison_args = argparse.Namespace()
            
            # Set default values
            comparison_args.run_traditional_ml = False
            comparison_args.run_logistic_regression = False
            comparison_args.run_random_forest = False
            comparison_args.run_sequence_models = False
            comparison_args.run_bilstm = False
            comparison_args.run_transformer = False
            comparison_args.run_gnn_models = False
            comparison_args.run_gcn = False
            comparison_args.run_graphsage = False
            comparison_args.run_gat = False
            comparison_args.run_improved_gnn = False
            comparison_args.run_all = False
            
            comparison_args.hidden_channels = 64
            comparison_args.out_channels = 32
            comparison_args.num_layers = 2
            comparison_args.train_batch_size = args.batch_size
            comparison_args.eval_batch_size = args.batch_size
            comparison_args.learning_rate = args.learning_rate
            comparison_args.weight_decay = 0.001
            comparison_args.max_grad_norm = 1.0
            comparison_args.epochs = args.epochs
            comparison_args.seed = args.seed
            comparison_args.class_weights = args.class_weights
            
            # Set model flags based on models_to_run
            if run_all:
                comparison_args.run_all = True
                comparison_args.run_traditional_ml = True
                comparison_args.run_logistic_regression = True
                comparison_args.run_random_forest = True
                comparison_args.run_sequence_models = True
                comparison_args.run_bilstm = True
                comparison_args.run_transformer = True
                comparison_args.run_gnn_models = True
                comparison_args.run_gcn = True
                comparison_args.run_graphsage = True
                comparison_args.run_gat = True
                comparison_args.run_improved_gnn = True
            else:
                if "lr" in models_to_run or "logistic_regression" in models_to_run:
                    comparison_args.run_traditional_ml = True
                    comparison_args.run_logistic_regression = True
                
                if "rf" in models_to_run or "random_forest" in models_to_run:
                    comparison_args.run_traditional_ml = True
                    comparison_args.run_random_forest = True
                
                if "bilstm" in models_to_run:
                    comparison_args.run_sequence_models = True
                    comparison_args.run_bilstm = True
                
                if "transformer" in models_to_run:
                    comparison_args.run_sequence_models = True
                    comparison_args.run_transformer = True
                
                if "gcn" in models_to_run:
                    comparison_args.run_gnn_models = True
                    comparison_args.run_gcn = True
                
                if "graphsage" in models_to_run:
                    comparison_args.run_gnn_models = True
                    comparison_args.run_graphsage = True
                
                if "gat" in models_to_run:
                    comparison_args.run_gnn_models = True
                    comparison_args.run_gat = True
                
                if "improved_gnn" in models_to_run:
                    comparison_args.run_gnn_models = True
                    comparison_args.run_improved_gnn = True
            
            # Run the comparison
            logger.info(f"Running comparison with models: {models_to_run}")
            run_comparison(comparison_args)
            
        except ImportError as e:
            logger.error(f"Error importing model_comparison module: {e}")
            sys.exit(1)
    
    logger.info("Model comparison completed")
    logger.info(f"Results saved to {args.output_dir}")

def create_gnn_comparison_file():
    """Create a standalone GNN comparison file"""
    gnn_comparison_content = '''import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import logging
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

# Import GNN models
from graph_models import GCNModel, GraphSAGEModel
from gnn_main import GATDefectDetectionModel, CodeDefectDataset, build_graph_from_code, set_seed
from improved_gnn_model import ImprovedGNNModel

logger = logging.getLogger(__name__)

def train_gnn_model(model, train_loader, val_loader, test_loader, device, args):
    """Train a GNN model and return evaluation metrics"""
    # Initialize optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Use class weights if there's imbalance
    if args.class_weights:
        # Calculate class weights from training data
        y_train = torch.cat([data.y for data in train_loader.dataset])
        class_counts = torch.bincount(y_train.long())
        total_samples = len(y_train)
        weight_0 = total_samples / (2 * class_counts[0])
        weight_1 = total_samples / (2 * class_counts[1])
        pos_weight = torch.tensor([weight_1 / weight_0]).to(device)
        logger.info(f"Using class weights - pos_weight: {pos_weight.item():.4f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Add learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Early stopping variables
    patience = 10
    epochs_no_improve = 0
    best_f1 = 0
    
    # Training loop
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Training"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch)
            loss = criterion(logits, batch.y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):
                batch = batch.to(device)
                logits = model(batch)
                loss = criterion(logits, batch.y)
                val_loss += loss.item()
                
                # Get predictions
                preds = torch.sigmoid(logits) > 0.5
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate metrics
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}:")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"Val Precision: {val_precision:.4f}")
        logger.info(f"Val Recall: {val_recall:.4f}")
        logger.info(f"Val F1: {val_f1:.4f}")
        
        # Step the scheduler with the eval F1 score
        scheduler.step(val_f1)
        
        # Save best model and early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}.pt")
            logger.info(f"Best model saved with F1: {best_f1:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for testing
    model.load_state_dict(torch.load(f"best_{model.__class__.__name__}.pt"))
    
    # Test evaluation
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            logits = model(batch)
            
            # Get predictions
            preds = torch.sigmoid(logits) > 0.5
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch.y.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    
    return {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'confusion_matrix': cm
    }

def run_gnn_comparison(args):
    """Run comparison between different GNN models"""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info(f"Using device: {device}")
    
    # Load language for parsing
    try:
        from tree_sitter import Language
        language = Language('/home/hohoanghvy/tree-sitter-container/build/my-languages.so', 'c')
    except Exception as e:
        logger.error(f"Failed to load C language: {e}")
        return
    
    # Load dataset
    import datasets
    from sklearn.model_selection import train_test_split
    
    logger.info("Loading datasets...")
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
    
    # Results dictionary
    results = {}
    
    # Prepare GNN data
    logger.info("Preparing data for GNN models...")
    
    # Create datasets
    train_dataset = CodeDefectDataset(
        train_data["code"].tolist(),
        train_data["label"].tolist(),
        language,
        args
    )
    val_dataset = CodeDefectDataset(
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
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset,
                            batch_size=args.train_batch_size,
                            shuffle=True,
                            num_workers=4)
    val_loader = DataLoader(val_dataset,
                          batch_size=args.eval_batch_size,
                          shuffle=False,
                          num_workers=4)
    test_loader = DataLoader(test_dataset,
                           batch_size=args.eval_batch_size,
                           shuffle=False,
                           num_workers=4)
    
    # GCN
    logger.info("Training GCN model...")
    gcn_model = GCNModel(
        in_channels=7,  # Feature dimension from node features
        hidden_channels=64,
        out_channels=32,
        num_layers=2
    ).to(device)
    
    results['GCN'] = train_gnn_model(gcn_model, train_loader, val_loader, test_loader, device, args)
    
    # GraphSAGE
    logger.info("Training GraphSAGE model...")
    graphsage_model = GraphSAGEModel(
        in_channels=7,  # Feature dimension from node features
        hidden_channels=64,
        out_channels=32,
        num_layers=2
    ).to(device)
    
    results['GraphSAGE'] = train_gnn_model(graphsage_model, train_loader, val_loader, test_loader, device, args)
    
    # GAT (current model)
    logger.info("Training GAT model...")
    gat_model = GATDefectDetectionModel(
        in_channels=7,  # Feature dimension from node features
        hidden_channels=64,
        out_channels=32,
        num_layers=2
    ).to(device)
    
    results['GAT'] = train_gnn_model(gat_model, train_loader, val_loader, test_loader, device, args)
    
    # Improved GNN Model
    logger.info("Training Improved GNN model...")
    improved_model = ImprovedGNNModel(
        in_channels=7,  # Feature dimension from node features
        hidden_channels=64,
        out_channels=32,
        num_layers=2,
        dropout=0.3,
        heads=2
    ).to(device)
    
    results['ImprovedGNN'] = train_gnn_model(improved_model, train_loader, val_loader, test_loader, device, args)
    
    # Print and visualize results
    logger.info("\\n===== GNN MODEL COMPARISON RESULTS =====")
    
    # Create results table
    results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    
    for model_name, metrics in results.items():
        logger.info(f"\\n{model_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        
        # Add to dataframe
        new_row = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Non-vulnerable', 'Vulnerable'],
                   yticklabels=['Non-vulnerable', 'Vulnerable'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{model_name}.png")
        plt.close()
    
    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(args.output_dir, "gnn_comparison_results.csv"), index=False)
    
    # Plot comparative bar charts
    plt.figure(figsize=(12, 8))
    
    # Accuracy
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('Accuracy Comparison')
    plt.xticks(rotation=45)
    
    # Precision
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='Precision', data=results_df)
    plt.title('Precision Comparison')
    plt.xticks(rotation=45)
    
    # Recall
    plt.subplot(2, 2, 3)
    sns.barplot(x='Model', y='Recall', data=results_df)
    plt.title('Recall Comparison')
    plt.xticks(rotation=45)
    
    # F1
    plt.subplot(2, 2, 4)
    sns.barplot(x='Model', y='F1', data=results_df)
    plt.title('F1 Score Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "gnn_comparison_metrics.png"))
    plt.close()
    
    logger.info(f"Results saved to {os.path.join(args.output_dir, 'gnn_comparison_results.csv')}")
    logger.info("Visualizations saved as PNG files")
'''
    
    with open("gnn_comparison.py", "w") as f:
        f.write(gnn_comparison_content)

if __name__ == "__main__":
    main()
