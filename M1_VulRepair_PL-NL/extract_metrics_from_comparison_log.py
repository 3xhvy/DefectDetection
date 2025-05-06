import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to the log file
log_file = '/home/hohoanghvy/PycharmProjects/DefectDetection/comparison_results/comparison.log'
output_csv = '/home/hohoanghvy/PycharmProjects/DefectDetection/M1_VulRepair_PL-NL/model_metrics_summary.csv'
output_html = '/home/hohoanghvy/PycharmProjects/DefectDetection/M1_VulRepair_PL-NL/model_metrics_summary.html'

# Regular expressions to extract model metrics
model_pattern = re.compile(r'INFO - model_comparison - ([A-Za-z0-9_-]+):')
accuracy_pattern = re.compile(r'INFO - model_comparison -\s+Accuracy: ([0-9.]+)')
precision_pattern = re.compile(r'INFO - model_comparison -\s+Precision: ([0-9.]+)')
recall_pattern = re.compile(r'INFO - model_comparison -\s+Recall: ([0-9.]+)')
f1_pattern = re.compile(r'INFO - model_comparison -\s+F1: ([0-9.]+)')

# Store metrics for each model
models = []
current_model = None
metrics = {}

print("Extracting metrics from comparison log...")

with open(log_file, 'r') as f:
    for line in f:
        # Check for model name
        model_match = model_pattern.search(line)
        if model_match:
            # If we found a new model and already have metrics for the previous one, save them
            if current_model and any(metrics):
                models.append({
                    'Model': current_model,
                    'Accuracy': metrics.get('Accuracy', ''),
                    'Precision': metrics.get('Precision', ''),
                    'Recall': metrics.get('Recall', ''),
                    'F1': metrics.get('F1', '')
                })
                metrics = {}
            
            current_model = model_match.group(1)
        
        # Extract metrics
        accuracy_match = accuracy_pattern.search(line)
        if accuracy_match:
            metrics['Accuracy'] = float(accuracy_match.group(1))
            
        precision_match = precision_pattern.search(line)
        if precision_match:
            metrics['Precision'] = float(precision_match.group(1))
            
        recall_match = recall_pattern.search(line)
        if recall_match:
            metrics['Recall'] = float(recall_match.group(1))
            
        f1_match = f1_pattern.search(line)
        if f1_match:
            metrics['F1'] = float(f1_match.group(1))

# Add the last model if there is one
if current_model and any(metrics):
    models.append({
        'Model': current_model,
        'Accuracy': metrics.get('Accuracy', ''),
        'Precision': metrics.get('Precision', ''),
        'Recall': metrics.get('Recall', ''),
        'F1': metrics.get('F1', '')
    })

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(models)

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Metrics saved to {output_csv}")

# Generate HTML report with styling
html = """
<html>
<head>
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 20px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    th, td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    th {
        background-color: #3498db;
        color: white;
    }
    tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    tr:hover {
        background-color: #e5e5e5;
    }
    .best {
        font-weight: bold;
        color: #27ae60;
    }
</style>
</head>
<body>
    <h1>Model Comparison Results</h1>
"""

# Add table to HTML
html += df.to_html(classes='table', index=False)

# Add summary section
html += """
    <h2>Summary</h2>
"""

if not df.empty:
    # Find best model for each metric
    best_accuracy = df.loc[df['Accuracy'].idxmax()] if 'Accuracy' in df.columns and not df['Accuracy'].empty else None
    best_precision = df.loc[df['Precision'].idxmax()] if 'Precision' in df.columns and not df['Precision'].empty else None
    best_recall = df.loc[df['Recall'].idxmax()] if 'Recall' in df.columns and not df['Recall'].empty else None
    best_f1 = df.loc[df['F1'].idxmax()] if 'F1' in df.columns and not df['F1'].empty else None
    
    html += "<ul>"
    if best_accuracy is not None:
        html += f"<li>Best Accuracy: <span class='best'>{best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})</span></li>"
    if best_precision is not None:
        html += f"<li>Best Precision: <span class='best'>{best_precision['Model']} ({best_precision['Precision']:.4f})</span></li>"
    if best_recall is not None:
        html += f"<li>Best Recall: <span class='best'>{best_recall['Model']} ({best_recall['Recall']:.4f})</span></li>"
    if best_f1 is not None:
        html += f"<li>Best F1 Score: <span class='best'>{best_f1['Model']} ({best_f1['F1']:.4f})</span></li>"
    html += "</ul>"
else:
    html += "<p>No metrics found in the log file.</p>"

html += """
</body>
</html>
"""

# Save HTML report
with open(output_html, 'w') as f:
    f.write(html)
print(f"HTML report saved to {output_html}")

# Create visualizations if we have data
if not df.empty:
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart for each metric
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    for i, metric in enumerate(metrics):
        if metric in df.columns and not df[metric].empty:
            plt.subplot(2, 2, i+1)
            sns.barplot(x='Model', y=metric, data=df)
            plt.title(f'{metric} by Model')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)  # Metrics are typically between 0 and 1
    
    plt.tight_layout()
    plt.savefig('/home/hohoanghvy/PycharmProjects/DefectDetection/M1_VulRepair_PL-NL/model_metrics_comparison.png')
    print("Visualization saved to model_metrics_comparison.png")

# Print summary to console
if not df.empty:
    print("\n===== MODEL METRICS SUMMARY =====")
    print(df)
    
    print("\n===== BEST PERFORMING MODELS =====")
    if 'Accuracy' in df.columns and not df['Accuracy'].empty:
        best_model = df.loc[df['Accuracy'].idxmax()]
        print(f"Best Accuracy: {best_model['Model']} ({best_model['Accuracy']:.4f})")
    
    if 'F1' in df.columns and not df['F1'].empty:
        best_model = df.loc[df['F1'].idxmax()]
        print(f"Best F1 Score: {best_model['Model']} ({best_model['F1']:.4f})")
else:
    print("No metrics found in the log file.")
