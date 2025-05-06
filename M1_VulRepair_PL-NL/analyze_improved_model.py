import torch
import torch.nn as nn
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
from torch_geometric.loader import DataLoader
from improved_gnn_model import ImprovedGNNModel, build_enhanced_graph_from_code, enrich_node_features_advanced
from train_improved_model import EnhancedCodeDefectDataset

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, test_loader, device):
    """Evaluate model and return detailed metrics and predictions"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)
            logits = model(data)
            
            # Get probabilities and predictions
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(float)
            labels = data.y.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    return metrics, all_labels, all_preds, all_probs

def calculate_metrics(labels, predictions, probabilities):
    """Calculate comprehensive metrics"""
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Handle edge cases
    if len(np.unique(labels)) == 1:
        logger.warning("Only one class present in labels. Some metrics will be set to 0.")
        precision = 0 if np.unique(labels)[0] == 0 else 1
        recall = 0 if np.unique(labels)[0] == 0 else 1
        f1 = 0
    else:
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
    
    # ROC AUC and PR AUC
    try:
        roc_auc = roc_auc_score(labels, probabilities)
    except:
        roc_auc = 0.5
    
    try:
        pr_auc = average_precision_score(labels, probabilities)
    except:
        pr_auc = sum(labels) / len(labels)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def plot_confusion_matrix(labels, predictions, output_dir):
    """Plot and save confusion matrix"""
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
    
    return {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'specificity': specificity,
        'npv': npv
    }

def plot_roc_curve(labels, probabilities, output_dir):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
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
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    logger.info(f"Optimal threshold from ROC curve: {optimal_threshold:.4f}")
    
    return optimal_threshold

def plot_pr_curve(labels, probabilities, output_dir):
    """Plot and save Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
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
    
    # Find optimal threshold for F1 score
    f1_scores = []
    for i in range(len(precision)-1):  # -1 because precision has one more element
        if i < len(thresholds):  # Ensure we don't go out of bounds
            p = precision[i]
            r = recall[i]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_scores.append((f1, thresholds[i]))
    
    if f1_scores:
        optimal_f1, optimal_threshold = max(f1_scores, key=lambda x: x[0])
        logger.info(f"Optimal threshold from PR curve (max F1): {optimal_threshold:.4f}, F1: {optimal_f1:.4f}")
        return optimal_threshold
    return 0.5  # Default if no optimal found

def threshold_analysis(labels, probabilities, output_dir):
    """Analyze different thresholds and their impact on metrics"""
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []
    
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(float)
        metrics = calculate_metrics(labels, predictions, probabilities)
        results.append({
            'threshold': threshold,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Plot metrics vs threshold
    plt.figure(figsize=(12, 8))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(df['threshold'], df[metric], marker='o', label=metric)
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'))
    plt.close()
    
    # Find best threshold for F1
    best_f1_idx = df['f1'].idxmax()
    best_f1_threshold = df.loc[best_f1_idx, 'threshold']
    logger.info(f"Best threshold for F1 from analysis: {best_f1_threshold:.2f}, F1: {df.loc[best_f1_idx, 'f1']:.4f}")
    
    return df, best_f1_threshold

def analyze_error_cases(labels, predictions, probabilities, output_dir):
    """Analyze error cases (false positives and false negatives)"""
    # Create DataFrame with all predictions
    df = pd.DataFrame({
        'true_label': labels,
        'prediction': predictions,
        'probability': probabilities
    })
    
    # Identify error cases
    df['error_type'] = 'correct'
    df.loc[(df['true_label'] == 1) & (df['prediction'] == 0), 'error_type'] = 'false_negative'
    df.loc[(df['true_label'] == 0) & (df['prediction'] == 1), 'error_type'] = 'false_positive'
    
    # Count error types
    error_counts = df['error_type'].value_counts()
    logger.info("Error Analysis:")
    for error_type, count in error_counts.items():
        logger.info(f"  {error_type}: {count}")
    
    # Analyze probability distributions for error cases
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='probability', hue='error_type', bins=20, multiple='stack')
    plt.title('Probability Distribution by Prediction Type')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'error_probability_distribution.png'))
    plt.close()
    
    # Save error cases for further analysis
    error_cases = df[df['error_type'] != 'correct']
    error_cases.to_csv(os.path.join(output_dir, 'error_cases.csv'), index=False)
    
    return error_counts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./improved_gnn_checkpoints/model_best.pt",
                        type=str, help="Path to the trained model")
    parser.add_argument("--hidden_channels", default=256, type=int)
    parser.add_argument("--out_channels", default=128, type=int)
    parser.add_argument("--num_layers", default=3, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--output_dir", default="./improved_analysis_results", type=str)
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
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
    
    # Initialize model
    model = ImprovedGNNModel(
        in_channels=26,  # Updated feature dimension
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        dropout=0.3,
        heads=args.heads
    ).to(device)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(args.model_path))
        logger.info(f"Loaded model from {args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    
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
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=4,
                           pin_memory=True)
    
    # Evaluate model
    metrics, labels, predictions, probabilities = evaluate_model(model, test_loader, device)
    
    # Log performance metrics
    logger.info("Performance Summary:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
    
    # Generate visualizations
    cm_metrics = plot_confusion_matrix(labels, predictions, args.output_dir)
    roc_threshold = plot_roc_curve(labels, probabilities, args.output_dir)
    pr_threshold = plot_pr_curve(labels, probabilities, args.output_dir)
    threshold_df, best_f1_threshold = threshold_analysis(labels, probabilities, args.output_dir)
    error_counts = analyze_error_cases(labels, predictions, probabilities, args.output_dir)
    
    # Save all metrics to a summary file
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write("PERFORMANCE SUMMARY\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX METRICS\n")
        f.write(f"True Positives: {cm_metrics['true_positives']}\n")
        f.write(f"False Positives: {cm_metrics['false_positives']}\n")
        f.write(f"True Negatives: {cm_metrics['true_negatives']}\n")
        f.write(f"False Negatives: {cm_metrics['false_negatives']}\n")
        f.write(f"Specificity: {cm_metrics['specificity']:.4f}\n")
        f.write(f"Negative Predictive Value: {cm_metrics['npv']:.4f}\n\n")
        
        f.write("THRESHOLD ANALYSIS\n")
        f.write(f"Optimal ROC threshold: {roc_threshold:.4f}\n")
        f.write(f"Optimal PR curve threshold: {pr_threshold:.4f}\n")
        f.write(f"Best F1 threshold: {best_f1_threshold:.4f}\n\n")
        
        f.write("ERROR ANALYSIS\n")
        for error_type, count in error_counts.items():
            f.write(f"{error_type}: {count}\n")
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
