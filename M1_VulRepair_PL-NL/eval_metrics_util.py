import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve

def evaluate_and_plot(y_true, y_pred, y_score=None, model_name='Model'):
    '''
    y_true: true labels
    y_pred: predicted labels (binary)
    y_score: predicted probabilities (optional, for ROC/PR)
    '''
    print(f"\n===== {model_name} Evaluation =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    if y_score is not None:
        roc_auc = roc_auc_score(y_true, y_score)
        print(f"ROC AUC: {roc_auc:.4f}")
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.show()
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        plt.figure()
        plt.plot(recall, precision, label='PR curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.show()
