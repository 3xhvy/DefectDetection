import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import logging
from torch_geometric.data import DataLoader
from improved_gnn_model import ImprovedGNNModel, MemoryEfficientTrainer, FocalLoss, build_enhanced_graph_from_code, enrich_node_features_advanced
from tree_sitter import Language, Parser
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path, in_channels, device):
    """Load a trained model from disk"""
    try:
        # Create model with memory-efficient settings
        model = ImprovedGNNModel(
            in_channels=in_channels,
            hidden_channels=64,  # Reduced from 128
            out_channels=32,     # Reduced from 64
            num_layers=2,
            dropout=0.3,
            heads=2,
            use_checkpointing=True,
            use_mixed_precision=True,
            memory_efficient=True
        ).to(device)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def prepare_code_for_inference(code_snippet, language_path):
    """Convert code to graph data for model input"""
    try:
        # Load tree-sitter language
        LANGUAGE = Language(language_path, 'c')
        
        # Build graph from code
        graph_data = build_enhanced_graph_from_code(code_snippet, LANGUAGE)
        
        # Enrich with features
        graph_data = enrich_node_features_advanced(graph_data, code_snippet)
        
        return graph_data
    except Exception as e:
        logger.error(f"Error preparing code: {str(e)}")
        raise

def predict_vulnerability(model, graph_data, device):
    """Predict if code is vulnerable using the model"""
    model.eval()
    
    # Move data to device
    graph_data = graph_data.to(device)
    
    # Run inference with memory optimization
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = model(graph_data)
    
    # Get prediction (1 = vulnerable, 0 = not vulnerable)
    probability = torch.sigmoid(output).item()
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def optimize_cuda_memory():
    """Apply CUDA memory optimizations"""
    # Empty CUDA cache
    torch.cuda.empty_cache()
    
    # Set memory fraction to use
    import gc
    gc.collect()
    
    # Enable memory-efficient attention if PyTorch version supports it
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except:
        logger.info("Memory efficient SDP not available in this PyTorch version")
    
    # Set to deterministic mode for consistent memory usage
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info("Applied CUDA memory optimizations")

def main():
    parser = argparse.ArgumentParser(description='Run local inference with optimized GNN model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--code_file', type=str, required=True, help='Path to code file for vulnerability detection')
    parser.add_argument('--language_path', type=str, required=True, help='Path to tree-sitter language file')
    parser.add_argument('--in_channels', type=int, default=26, help='Number of input channels (feature dimensions)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device index to use')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Apply memory optimizations for CUDA
    if torch.cuda.is_available():
        optimize_cuda_memory()
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    # Load model
    model = load_model(args.model_path, args.in_channels, device)
    
    # Read code file
    try:
        with open(args.code_file, 'r') as f:
            code_snippet = f.read()
    except Exception as e:
        logger.error(f"Error reading code file: {str(e)}")
        return
    
    # Prepare code for inference
    start_time = time.time()
    graph_data = prepare_code_for_inference(code_snippet, args.language_path)
    prep_time = time.time() - start_time
    
    # Run prediction
    start_time = time.time()
    prediction, probability = predict_vulnerability(model, graph_data, device)
    inference_time = time.time() - start_time
    
    # Display results
    logger.info(f"Code preparation time: {prep_time:.4f} seconds")
    logger.info(f"Inference time: {inference_time:.4f} seconds")
    logger.info(f"Prediction: {'Vulnerable' if prediction == 1 else 'Not Vulnerable'}")
    logger.info(f"Confidence: {probability:.4f}")

if __name__ == "__main__":
    main()
