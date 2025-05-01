import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import argparse
import logging
import os
from torch_geometric.loader import DataLoader
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

from gnn_main import GNNDefectDetectionModel, CodeDefectDataset
from tree_sitter import Language

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_model_performance(model, test_loader, device, output_dir):
    """Analyze model performance with detailed metrics and visualizations."""
    model.eval()
    all_preds_prob = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            probs = torch.sigmoid(output.squeeze()).cpu().numpy()
            preds = (probs > 0.5).astype(float)

            all_preds_prob.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy())

    # Convert to numpy arrays
    all_preds_prob = np.array(all_preds_prob)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Vulnerable', 'Vulnerable'],
                yticklabels=['Non-Vulnerable', 'Vulnerable'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_preds_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

    # 4. Distribution of prediction probabilities
    plt.figure(figsize=(12, 6))

    # Separate probabilities for actual positives and negatives
    pos_probs = all_preds_prob[all_labels == 1]
    neg_probs = all_preds_prob[all_labels == 0]

    plt.hist(neg_probs, bins=20, alpha=0.5, color='blue', label='Non-Vulnerable')
    plt.hist(pos_probs, bins=20, alpha=0.5, color='red', label='Vulnerable')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()

    # 5. Calculate and save detailed metrics
    accuracy = np.mean(all_preds == all_labels)
    true_positives = np.sum((all_preds == 1) & (all_labels == 1))
    false_positives = np.sum((all_preds == 1) & (all_labels == 0))
    true_negatives = np.sum((all_preds == 0) & (all_labels == 0))
    false_negatives = np.sum((all_preds == 0) & (all_labels == 1))

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Save metrics to file
    with open(os.path.join(output_dir, 'detailed_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"True Negatives: {true_negatives}\n")
        f.write(f"False Positives: {false_positives}\n")
        f.write(f"False Negatives: {false_negatives}\n")
        f.write(f"True Positives: {true_positives}\n")

    logger.info(f"Analysis complete. Results saved to {output_dir}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./gnn_checkpoints/checkpoint-best-f1/model_best.pt",
                        type=str, help="Path to the trained model")
    parser.add_argument("--hidden_channels", default=128, type=int)
    parser.add_argument("--out_channels", default=128, type=int)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--output_dir", default="./analysis_results", type=str)

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load C language for parsing
    try:
        language = Language('/home/hohoanghvy/tree-sitter-container/build/my-languages.so', 'c')
    except Exception as e:
        logger.error(f"Failed to load C language: {str(e)}")
        return

    # Initialize model
    model = GNNDefectDetectionModel(
        in_channels=7,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        num_layers=args.num_layers
    ).to(device)

    # Load model weights
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        logger.info(f"Loaded model from {args.model_path}")
    else:
        logger.error(f"Model not found at {args.model_path}")
        return

    # Load test dataset
    logger.info("Loading test dataset...")
    test_data = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="test")
    test_dataset = CodeDefectDataset(
        test_data["func"],
        test_data["target"],
        language,
        args
    )
    test_loader = DataLoader(test_dataset,
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=8,
                           pin_memory=True)

    # Analyze model performance
    metrics = analyze_model_performance(model, test_loader, device, args.output_dir)

    # Print summary
    logger.info("Performance Summary:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"Detailed results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
