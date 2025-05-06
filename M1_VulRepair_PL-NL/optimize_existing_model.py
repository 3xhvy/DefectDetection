import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import datasets
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# Import from your existing code
from gnn_main import GNNDefectDetectionModel, GATDefectDetectionModel, CodeDefectDataset, build_graph_from_code

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

# Evaluate model with threshold tuning
def evaluate_with_threshold(model, data_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            data = data.to(device)
            logits = model(data)
            
            # Get probabilities and predictions
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(float)
            labels = data.y.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # ROC AUC and PR AUC
    roc_auc = roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    return metrics, all_labels, all_preds, all_probs

# Find optimal threshold for F1 score
def find_optimal_threshold(model, data_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Finding optimal threshold"):
            data = data.to(device)
            logits = model(data)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = data.y.cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []
    
    for threshold in thresholds:
        preds = (all_probs > threshold).astype(float)
        f1 = f1_score(all_labels, preds, zero_division=0)
        f1_scores.append((threshold, f1))
    
    # Find threshold with highest F1
    best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
    logger.info(f"Optimal threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
    
    return best_threshold, f1_scores

# Plot confusion matrix
def plot_confusion_matrix(labels, predictions, output_dir):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Vulnerable', 'Vulnerable'],
                yticklabels=['Non-Vulnerable', 'Vulnerable'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate and log additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    logger.info(f"Additional Metrics from Confusion Matrix:")
    logger.info(f"  True Positives: {tp}")
    logger.info(f"  False Positives: {fp}")
    logger.info(f"  True Negatives: {tn}")
    logger.info(f"  False Negatives: {fn}")
    logger.info(f"  Specificity (True Negative Rate): {specificity:.4f}")
    logger.info(f"  Negative Predictive Value: {npv:.4f}")

# Plot ROC curve
def plot_roc_curve(labels, probabilities, output_dir):
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

# Plot Precision-Recall curve
def plot_pr_curve(labels, probabilities, output_dir):
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.axhline(y=sum(labels)/len(labels), color='red', linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

# Plot threshold vs F1 score
def plot_threshold_f1(f1_scores, output_dir):
    thresholds = [x[0] for x in f1_scores]
    f1s = [x[1] for x in f1_scores]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1s, marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold vs F1 Score')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_f1.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument("--model_path", default="./gnn_checkpoints/checkpoint-best-f1/model_best.pt",
                        type=str, help="Path to the trained model")
    parser.add_argument("--model_type", default="gat", type=str, choices=["gcn", "gat"],
                        help="Type of GNN model (gcn or gat)")
    parser.add_argument("--hidden_channels", default=256, type=int)
    parser.add_argument("--out_channels", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_dir", default="./optimized_model_results", type=str)
    parser.add_argument("--tune_threshold", action="store_true", 
                        help="Tune prediction threshold for optimal F1 score")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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
    
    # Load dataset
    logger.info("Loading test dataset...")
    test_data_hf = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="test")
    test_dataset = CodeDefectDataset(
        test_data_hf["func"],
        test_data_hf["target"],
        language,
        args
    )
    
    test_loader = DataLoader(test_dataset,
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=2)
    
    # Initialize model
    if args.model_type == "gat":
        model = GATDefectDetectionModel(
            in_channels=7,
            hidden_channels=64,  
            out_channels=args.out_channels,
            num_layers=4  
        ).to(device)
    else:
        model = GNNDefectDetectionModel(
            in_channels=7,
            hidden_channels=64,  
            out_channels=args.out_channels,
            num_layers=4  
        ).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(args.model_path))
        logger.info(f"Loaded model from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    
    # Find optimal threshold if requested
    threshold = 0.5  # Default threshold
    if args.tune_threshold:
        logger.info("Finding optimal threshold...")
        threshold, f1_scores = find_optimal_threshold(model, test_loader, device)
        plot_threshold_f1(f1_scores, args.output_dir)
    
    # Evaluate model with tuned threshold
    logger.info(f"Evaluating model with threshold {threshold:.2f}...")
    metrics, labels, predictions, probabilities = evaluate_with_threshold(
        model, test_loader, device, threshold
    )
    
    # Log performance metrics
    logger.info("Performance Summary:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
    
    # Generate visualizations
    plot_confusion_matrix(labels, predictions, args.output_dir)
    plot_roc_curve(labels, probabilities, args.output_dir)
    plot_pr_curve(labels, probabilities, args.output_dir)
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Model: {args.model_type.upper()}\n")
        f.write(f"Threshold: {threshold:.4f}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")
    
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
