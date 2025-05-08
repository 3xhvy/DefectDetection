"""
Graph Metrics Analyzer for Vulnerability Detection Models

This module provides functions to extract and analyze various graph metrics from:
1. GNN models (structure, propagation, centrality)
2. Diffusion models (diffusion parameters, graph evolution)

Usage:
    python graph_metrics_analyzer.py --model_type gnn --sample_idx 0
    python graph_metrics_analyzer.py --model_type diffusion --sample_idx 0
"""

import argparse
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import logging
import os
import pandas as pd
from tqdm import tqdm

# Import project modules
from gnn_main import (
    build_graph_from_code,
    CodeDefectDataset,
    GNNDefectDetectionModel,
    set_seed
)
from train_diffusion_model import (
    CodeDiffusionModel,
    filter_large_graphs
)
from graph_diffusion_model import GraphDiffusionModel

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def extract_basic_graph_metrics(graph_data):
    """Extract basic metrics from a PyG graph"""
    # Convert to NetworkX for analysis
    g = to_networkx(graph_data)
    
    metrics = {}
    
    # Basic graph properties
    metrics['num_nodes'] = g.number_of_nodes()
    metrics['num_edges'] = g.number_of_edges()
    metrics['density'] = nx.density(g)
    
    return metrics

def extract_centrality_metrics(graph_data):
    """Extract centrality metrics from a PyG graph"""
    # Convert to NetworkX for analysis
    g = to_networkx(graph_data)
    
    metrics = {}
    
    # Centrality measures
    try:
        # Degree centrality
        degree_centrality = nx.degree_centrality(g)
        metrics['avg_degree_centrality'] = sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0
        metrics['max_degree_centrality'] = max(degree_centrality.values()) if degree_centrality else 0
        
        # Betweenness centrality (how often a node lies on shortest paths)
        if g.number_of_nodes() > 1:
            betweenness_centrality = nx.betweenness_centrality(g)
            metrics['avg_betweenness_centrality'] = sum(betweenness_centrality.values()) / len(betweenness_centrality) if betweenness_centrality else 0
            metrics['max_betweenness_centrality'] = max(betweenness_centrality.values()) if betweenness_centrality else 0
        else:
            metrics['avg_betweenness_centrality'] = 0
            metrics['max_betweenness_centrality'] = 0
            
        # Closeness centrality (how close a node is to all other nodes)
        if nx.is_connected(g.to_undirected()) and g.number_of_nodes() > 1:
            closeness_centrality = nx.closeness_centrality(g)
            metrics['avg_closeness_centrality'] = sum(closeness_centrality.values()) / len(closeness_centrality) if closeness_centrality else 0
            metrics['max_closeness_centrality'] = max(closeness_centrality.values()) if closeness_centrality else 0
        else:
            metrics['avg_closeness_centrality'] = 0
            metrics['max_closeness_centrality'] = 0
    except Exception as e:
        logger.warning(f"Error calculating centrality metrics: {str(e)}")
        metrics['avg_degree_centrality'] = 0
        metrics['max_degree_centrality'] = 0
        metrics['avg_betweenness_centrality'] = 0
        metrics['max_betweenness_centrality'] = 0
        metrics['avg_closeness_centrality'] = 0
        metrics['max_closeness_centrality'] = 0
    
    return metrics

def extract_propagation_metrics(graph_data):
    """Extract propagation and connectivity metrics from a PyG graph"""
    # Convert to NetworkX for analysis
    g = to_networkx(graph_data)
    
    metrics = {}
    
    # Path length and connectivity
    try:
        if nx.is_connected(g.to_undirected()) and g.number_of_nodes() > 1:
            metrics['avg_path_length'] = nx.average_shortest_path_length(g.to_undirected())
            metrics['diameter'] = nx.diameter(g.to_undirected())
            
            # Eccentricity (maximum distance from each node to any other node)
            eccentricity = nx.eccentricity(g.to_undirected())
            metrics['avg_eccentricity'] = sum(eccentricity.values()) / len(eccentricity)
            
            # Center nodes (nodes with minimum eccentricity)
            center_nodes = nx.center(g.to_undirected())
            metrics['center_nodes'] = center_nodes
            metrics['num_center_nodes'] = len(center_nodes)
        else:
            metrics['avg_path_length'] = 0
            metrics['diameter'] = 0
            metrics['avg_eccentricity'] = 0
            metrics['center_nodes'] = []
            metrics['num_center_nodes'] = 0
    except Exception as e:
        logger.warning(f"Error calculating propagation metrics: {str(e)}")
        metrics['avg_path_length'] = 0
        metrics['diameter'] = 0
        metrics['avg_eccentricity'] = 0
        metrics['center_nodes'] = []
        metrics['num_center_nodes'] = 0
    
    # Clustering coefficient (measure of graph "tightness")
    try:
        metrics['clustering_coefficient'] = nx.average_clustering(g.to_undirected())
    except Exception as e:
        logger.warning(f"Error calculating clustering coefficient: {str(e)}")
        metrics['clustering_coefficient'] = 0
    
    return metrics

def extract_similarity_metrics(graph1, graph2):
    """Calculate similarity between two PyG graphs"""
    # Convert to NetworkX
    g1 = to_networkx(graph1)
    g2 = to_networkx(graph2)
    
    metrics = {}
    
    # Graph Edit Distance (smaller = more similar)
    try:
        if g1.number_of_nodes() < 10 and g2.number_of_nodes() < 10:  # Only for small graphs
            edit_distance = nx.graph_edit_distance(g1, g2)
            metrics['edit_distance'] = edit_distance
        else:
            metrics['edit_distance'] = "N/A - Graphs too large"
    except Exception as e:
        logger.warning(f"Error calculating edit distance: {str(e)}")
        metrics['edit_distance'] = float('inf')
    
    # Jaccard similarity of edges
    try:
        g1_edges = set(g1.edges())
        g2_edges = set(g2.edges())
        
        if len(g1_edges.union(g2_edges)) > 0:
            jaccard = len(g1_edges.intersection(g2_edges)) / len(g1_edges.union(g2_edges))
        else:
            jaccard = 0
            
        metrics['jaccard_similarity'] = jaccard
    except Exception as e:
        logger.warning(f"Error calculating Jaccard similarity: {str(e)}")
        metrics['jaccard_similarity'] = 0
    
    # Node overlap ratio
    try:
        g1_nodes = set(g1.nodes())
        g2_nodes = set(g2.nodes())
        
        if len(g1_nodes.union(g2_nodes)) > 0:
            node_overlap = len(g1_nodes.intersection(g2_nodes)) / len(g1_nodes.union(g2_nodes))
        else:
            node_overlap = 0
            
        metrics['node_overlap'] = node_overlap
    except Exception as e:
        logger.warning(f"Error calculating node overlap: {str(e)}")
        metrics['node_overlap'] = 0
        
    return metrics

def extract_diffusion_parameters(model):
    """Extract diffusion parameters from a diffusion model"""
    metrics = {}
    
    # Extract noise schedule parameters
    metrics['num_timesteps'] = model.num_timesteps
    
    # Get first, middle and last values of key parameters
    metrics['betas_start'] = model.betas[0].item()
    metrics['betas_mid'] = model.betas[model.num_timesteps // 2].item()
    metrics['betas_end'] = model.betas[-1].item()
    
    metrics['alphas_start'] = model.alphas[0].item()
    metrics['alphas_mid'] = model.alphas[model.num_timesteps // 2].item()
    metrics['alphas_end'] = model.alphas[-1].item()
    
    metrics['alphas_cumprod_start'] = model.alphas_cumprod[0].item()
    metrics['alphas_cumprod_mid'] = model.alphas_cumprod[model.num_timesteps // 2].item()
    metrics['alphas_cumprod_end'] = model.alphas_cumprod[-1].item()
    
    return metrics

def visualize_graph(graph_data, title="Graph Visualization"):
    """Visualize a PyG graph"""
    g = to_networkx(graph_data)
    
    plt.figure(figsize=(10, 8))
    
    # Use spring layout for node positioning
    pos = nx.spring_layout(g)
    
    # Draw nodes
    nx.draw_networkx_nodes(g, pos, node_size=50, node_color='skyblue')
    
    # Draw edges
    nx.draw_networkx_edges(g, pos, width=1, alpha=0.5)
    
    # Add labels if graph is small enough
    if g.number_of_nodes() < 50:
        nx.draw_networkx_labels(g, pos, font_size=8)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("graph_visualizations", exist_ok=True)
    plt.savefig(f"graph_visualizations/{title.replace(' ', '_')}.png")
    plt.close()

def analyze_gnn_graph(dataset, sample_idx=0):
    """Analyze a GNN graph from the dataset"""
    if sample_idx >= len(dataset):
        logger.error(f"Sample index {sample_idx} out of range (dataset size: {len(dataset)})")
        return
    
    # Get the sample graph
    sample = dataset[sample_idx]
    logger.info(f"Analyzing GNN graph sample {sample_idx}")
    
    # Extract metrics
    basic_metrics = extract_basic_graph_metrics(sample)
    centrality_metrics = extract_centrality_metrics(sample)
    propagation_metrics = extract_propagation_metrics(sample)
    
    # Combine metrics
    all_metrics = {**basic_metrics, **centrality_metrics, **propagation_metrics}
    
    # Print metrics
    logger.info("GNN Graph Metrics:")
    for key, value in all_metrics.items():
        if key != 'center_nodes':  # Skip printing the list of center nodes
            logger.info(f"  {key}: {value}")
    
    # Visualize the graph
    visualize_graph(sample, f"GNN_Graph_Sample_{sample_idx}")
    
    return all_metrics

def analyze_diffusion_graph(model, num_nodes=50, timesteps=None):
    """Analyze a diffusion graph at different timesteps"""
    logger.info("Analyzing diffusion graph")
    
    # Create a simple edge index for demonstration
    edge_index = torch.tensor([[i, i+1] for i in range(num_nodes-1)], dtype=torch.long).t().contiguous()
    
    # Extract diffusion parameters
    diffusion_params = extract_diffusion_parameters(model)
    
    # Print diffusion parameters
    logger.info("Diffusion Model Parameters:")
    for key, value in diffusion_params.items():
        logger.info(f"  {key}: {value}")
    
    # If timesteps not provided, use a default sequence
    if timesteps is None:
        timesteps = [0, model.num_timesteps // 4, model.num_timesteps // 2, 
                    3 * model.num_timesteps // 4, model.num_timesteps - 1]
    
    # Sample graphs at different timesteps
    metrics_at_timesteps = {}
    
    # Start with random noise
    x_T = torch.randn(num_nodes, model.node_dim)
    
    # For each timestep, denoise and analyze
    for t in reversed(range(model.num_timesteps)):
        # Skip timesteps not in our list
        if t not in timesteps:
            continue
            
        # Denoise to this timestep
        t_tensor = torch.full((num_nodes,), t, dtype=torch.long)
        x_t = model.reverse_diffusion_step(x_T, t_tensor, edge_index)
        
        # Create a PyG graph
        graph_data = Data(x=x_t, edge_index=edge_index)
        
        # Extract metrics
        basic_metrics = extract_basic_graph_metrics(graph_data)
        centrality_metrics = extract_centrality_metrics(graph_data)
        propagation_metrics = extract_propagation_metrics(graph_data)
        
        # Combine metrics
        all_metrics = {**basic_metrics, **centrality_metrics, **propagation_metrics}
        metrics_at_timesteps[t] = all_metrics
        
        # Visualize the graph
        visualize_graph(graph_data, f"Diffusion_Graph_Timestep_{t}")
    
    # Compare first and last timestep graphs
    if len(timesteps) >= 2:
        first_t = min(timesteps)
        last_t = max(timesteps)
        
        # Create graphs for comparison
        first_graph = Data(x=torch.randn(num_nodes, model.node_dim), edge_index=edge_index)
        last_graph = Data(x=model.sample(num_nodes, edge_index), edge_index=edge_index)
        
        # Calculate similarity
        similarity = extract_similarity_metrics(first_graph, last_graph)
        
        logger.info(f"Similarity between timestep {first_t} and {last_t}:")
        for key, value in similarity.items():
            logger.info(f"  {key}: {value}")
    
    return metrics_at_timesteps, diffusion_params

def main():
    parser = argparse.ArgumentParser(description="Graph Metrics Analyzer")
    parser.add_argument("--model_type", type=str, choices=["gnn", "diffusion"], required=True,
                        help="Type of model to analyze (gnn or diffusion)")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of sample to analyze (for GNN)")
    parser.add_argument("--num_nodes", type=int, default=50,
                        help="Number of nodes for diffusion graph generation")
    parser.add_argument("--output_dir", type=str, default="./graph_metrics",
                        help="Directory to save output metrics")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model_type == "gnn":
        # Load a sample dataset
        from datasets import load_dataset
        
        # Load a small subset of the dataset for analysis
        dataset = load_dataset("code_x_glue_cc_defect_detection")
        train_data = dataset["train"]
        
        # Convert to list format
        code_samples = train_data["func"][:100]  # Limit to 100 samples
        labels = train_data["target"][:100]
        
        # Create the dataset
        from tree_sitter import Language
        try:
            language = Language('build/languages.so', 'c')
        except:
            # Fallback
            logger.warning("Could not load C language, using Python instead")
            language = Language('build/languages.so', 'python')
        
        # Create a simple args object for the dataset
        class Args:
            def __init__(self):
                self.memory_efficient = True
        
        gnn_dataset = CodeDefectDataset(code_samples, labels, language, Args())
        
        # Analyze GNN graph
        metrics = analyze_gnn_graph(gnn_dataset, args.sample_idx)
        
        # Save metrics to CSV
        pd.DataFrame([metrics]).to_csv(f"{args.output_dir}/gnn_graph_metrics_sample_{args.sample_idx}.csv", index=False)
        
    elif args.model_type == "diffusion":
        # Create a simple diffusion model
        node_dim = 64  # Example dimension
        model = GraphDiffusionModel(node_dim=node_dim)
        
        # Analyze diffusion graph
        metrics, params = analyze_diffusion_graph(model, num_nodes=args.num_nodes)
        
        # Save metrics to CSV
        for timestep, timestep_metrics in metrics.items():
            pd.DataFrame([timestep_metrics]).to_csv(
                f"{args.output_dir}/diffusion_graph_metrics_timestep_{timestep}.csv", index=False)
        
        # Save diffusion parameters
        pd.DataFrame([params]).to_csv(f"{args.output_dir}/diffusion_parameters.csv", index=False)

if __name__ == "__main__":
    main()
