import re
import csv

log_path = 'comparison_results/comparison.log'
csv_path = 'model_comparison_results.csv'

model_metrics = []
current_model = None
metrics = {}

# Regex patterns for metrics
model_pattern = re.compile(r'- ([A-Za-z0-9_]+):')
acc_pattern = re.compile(r'Accuracy: ([0-9.]+)')
prec_pattern = re.compile(r'Precision: ([0-9.]+)')
rec_pattern = re.compile(r'Recall: ([0-9.]+)')
f1_pattern = re.compile(r'F1: ([0-9.]+)')

with open(log_path, 'r') as f:
    for line in f:
        model_match = model_pattern.search(line)
        if model_match:
            if current_model and metrics:
                model_metrics.append([current_model, metrics.get('Accuracy',''), metrics.get('Precision',''), metrics.get('Recall',''), metrics.get('F1','')])
                metrics = {}
            current_model = model_match.group(1)
        acc_match = acc_pattern.search(line)
        if acc_match:
            metrics['Accuracy'] = acc_match.group(1)
        prec_match = prec_pattern.search(line)
        if prec_match:
            metrics['Precision'] = prec_match.group(1)
        rec_match = rec_pattern.search(line)
        if rec_match:
            metrics['Recall'] = rec_match.group(1)
        f1_match = f1_pattern.search(line)
        if f1_match:
            metrics['F1'] = f1_match.group(1)
    # Add last model
    if current_model and metrics:
        model_metrics.append([current_model, metrics.get('Accuracy',''), metrics.get('Precision',''), metrics.get('Recall',''), metrics.get('F1','')])

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1'])
    writer.writerows(model_metrics)

print(f"Extracted metrics for {len(model_metrics)} models. Results saved to {csv_path}")
