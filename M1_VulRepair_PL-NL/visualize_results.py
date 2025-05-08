#!/usr/bin/env python3
"""
Visualization Script for Diffusion Model Results

This script generates visualizations for training progress and evaluation metrics.
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import torch

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def extract_losses_from_log(log_file):
    """Extract training and validation losses from log file"""
    train_losses = []
    val_losses = []
    epochs = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # Match training loss
        train_match = re.search(r'Epoch (\d+)/\d+ - Train Loss: (\d+\.\d+)', line)
        if train_match:
            epoch = int(train_match.group(1))
            loss = float(train_match.group(2))
            epochs.append(epoch)
            train_losses.append(loss)
        
        # Match validation loss
        val_match = re.search(r'Epoch (\d+)/\d+ - Validation Loss: (\d+\.\d+)', line)
        if val_match:
            loss = float(val_match.group(2))
            val_losses.append(loss)
    
    return epochs, train_losses, val_losses

def plot_training_curves(log_file, output_dir="./charts"):
    """Plot training and validation loss curves"""
    epochs, train_losses, val_losses = extract_losses_from_log(log_file)
    
    if not epochs:
        print("No training data found in log file")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Diffusion Model Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path)
    print(f"Saved training curves to {output_path}")
    plt.close()
    
    # Plot loss reduction
    if len(train_losses) > 1:
        initial_train_loss = train_losses[0]
        train_loss_reduction = [(initial_train_loss - loss) / initial_train_loss * 100 for loss in train_losses]
        
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, train_loss_reduction, 'g-', label='Training Loss Reduction (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Reduction (%)')
        plt.title('Diffusion Model Training Improvement')
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, 'loss_reduction.png')
        plt.savefig(output_path)
        print(f"Saved loss reduction chart to {output_path}")
        plt.close()

def visualize_evaluation_metrics(results_file, output_dir="./charts"):
    """Visualize evaluation metrics from results file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    avg_results = results.get('average_results', {})
    if not avg_results:
        print("No evaluation results found in file")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Bar chart of average metrics
    metrics = list(avg_results.keys())
    values = list(avg_results.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, values, color=sns.color_palette("viridis", len(metrics)))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)  # Assuming metrics are between 0 and 1
    plt.ylabel('Score')
    plt.title('Diffusion Model Evaluation Metrics')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'evaluation_metrics.png')
    plt.savefig(output_path)
    print(f"Saved evaluation metrics chart to {output_path}")
    plt.close()
    
    # If individual results exist, create a box plot
    individual_results = results.get('individual_results', {})
    if individual_results:
        # Prepare data for box plot
        data = []
        labels = []
        
        for metric, values in individual_results.items():
            if values:  # Check if the list is not empty
                data.append(values)
                labels.append(metric)
        
        if data:  # Check if we have any data to plot
            plt.figure(figsize=(12, 6))
            plt.boxplot(data, labels=labels)
            plt.ylabel('Score')
            plt.title('Distribution of Evaluation Metrics')
            plt.grid(True, axis='y')
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, 'metrics_distribution.png')
            plt.savefig(output_path)
            print(f"Saved metrics distribution chart to {output_path}")
            plt.close()

def visualize_code_diff(original_code, repaired_code, output_dir="./charts"):
    """Visualize the differences between original and repaired code"""
    import difflib
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the diff
    original_lines = original_code.splitlines()
    repaired_lines = repaired_code.splitlines()
    
    diff = list(difflib.unified_diff(original_lines, repaired_lines, lineterm=''))
    
    # Create a colorful HTML diff
    html = ['<!DOCTYPE html>',
            '<html>',
            '<head>',
            '<meta charset="utf-8">',
            '<style>',
            'body { font-family: monospace; white-space: pre; }',
            '.diff { margin: 10px; padding: 10px; border: 1px solid #ccc; }',
            '.removed { background-color: #fdd; color: #900; }',
            '.added { background-color: #dfd; color: #090; }',
            '.info { color: #00c; }',
            '</style>',
            '</head>',
            '<body>',
            '<h1>Code Repair Diff</h1>',
            '<div class="diff">']
    
    for line in diff:
        if line.startswith('+'):
            html.append(f'<div class="added">{line}</div>')
        elif line.startswith('-'):
            html.append(f'<div class="removed">{line}</div>')
        elif line.startswith('@@'):
            html.append(f'<div class="info">{line}</div>')
        else:
            html.append(line)
    
    html.append('</div></body></html>')
    
    # Write the HTML file
    output_path = os.path.join(output_dir, 'code_diff.html')
    with open(output_path, 'w') as f:
        f.write('\n'.join(html))
    
    print(f"Saved code diff visualization to {output_path}")

def visualize_attention_weights(model, code_graph, output_dir="./charts"):
    """Visualize attention weights in the model (if applicable)"""
    # This is a placeholder function - implement if your model has attention mechanisms
    # For GCN-based models without explicit attention, this could visualize node importance
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # For demonstration purposes, let's create a random node importance heatmap
    num_nodes = code_graph.x.shape[0]
    
    # Create a random importance matrix for demonstration
    # In a real implementation, you would extract this from your model
    importance = np.random.rand(num_nodes, num_nodes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(importance, cmap='viridis')
    plt.title('Node Importance Visualization')
    plt.xlabel('Node Index')
    plt.ylabel('Node Index')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'node_importance.png')
    plt.savefig(output_path)
    print(f"Saved node importance visualization to {output_path}")
    plt.close()

def main():
    """Main function to generate all visualizations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations for diffusion model results")
    parser.add_argument("--log_file", type=str, default="diffusion_model_training.log", 
                        help="Path to training log file")
    parser.add_argument("--results_file", type=str, default="evaluation_results.json",
                        help="Path to evaluation results file")
    parser.add_argument("--output_dir", type=str, default="./charts",
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Plot training curves
    if os.path.exists(args.log_file):
        plot_training_curves(args.log_file, args.output_dir)
    else:
        print(f"Log file {args.log_file} not found")
    
    # Visualize evaluation metrics
    if os.path.exists(args.results_file):
        visualize_evaluation_metrics(args.results_file, args.output_dir)
    else:
        print(f"Results file {args.results_file} not found")
    
    print(f"All visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
