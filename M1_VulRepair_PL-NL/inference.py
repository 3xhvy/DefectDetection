import torch
import argparse
import logging
from tree_sitter import Language
import os
from gnn_main import GNNDefectDetectionModel, build_graph_from_code

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_vulnerability(code_snippet, model, language, device):
    """Predict if a code snippet contains vulnerabilities."""
    # Convert code to graph
    graph_data = build_graph_from_code(code_snippet, language)
    
    # Create node features (same as in training)
    node_features = []
    for i, node_type in enumerate(graph_data.type):
        node_type_hash = hash(node_type) % 1000
        
        # Enhanced feature vector with multiple dimensions
        feature = [
            node_type_hash / 1000.0,  # Normalized hash value
            float(len(node_type)) / 50.0,  # Normalized length of type name
            float(graph_data.start_byte[i]) / 10000.0,  # Normalized position
            float(graph_data.end_byte[i] - graph_data.start_byte[i]) / 1000.0,  # Normalized size
            1.0 if "expr" in node_type else 0.0,  # Is expression
            1.0 if "decl" in node_type else 0.0,  # Is declaration
            1.0 if "stmt" in node_type else 0.0,  # Is statement
        ]
        
        node_features.append(feature)
    
    graph_data.x = torch.tensor(node_features, dtype=torch.float)
    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
    
    # Move to device
    graph_data = graph_data.to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(graph_data)
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./gnn_checkpoints/checkpoint-best-f1/model_best.pt", 
                        type=str, help="Path to the trained model")
    parser.add_argument("--hidden_channels", default=128, type=int)
    parser.add_argument("--out_channels", default=128, type=int)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--input_file", type=str, help="Path to file containing code to analyze")
    parser.add_argument("--code_snippet", type=str, help="Direct code snippet to analyze")
    
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
    
    # Get code to analyze
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            return
    elif args.code_snippet:
        code = args.code_snippet
    else:
        logger.error("Please provide either --input_file or --code_snippet")
        return
    
    # Make prediction
    prediction, probability = predict_vulnerability(code, model, language, device)
    
    # Output results
    logger.info("Analysis Results:")
    logger.info(f"Vulnerability Prediction: {'Vulnerable' if prediction == 1 else 'Not Vulnerable'}")
    logger.info(f"Confidence: {probability:.4f}")
    
    # Print detailed explanation based on confidence
    if prediction == 1:
        if probability > 0.9:
            logger.info("HIGH CONFIDENCE: This code is very likely to contain vulnerabilities.")
        elif probability > 0.7:
            logger.info("MEDIUM CONFIDENCE: This code likely contains vulnerabilities.")
        else:
            logger.info("LOW CONFIDENCE: This code might contain vulnerabilities.")
    else:
        if probability < 0.1:
            logger.info("HIGH CONFIDENCE: This code is very likely to be safe.")
        elif probability < 0.3:
            logger.info("MEDIUM CONFIDENCE: This code is likely to be safe.")
        else:
            logger.info("LOW CONFIDENCE: This code is probably safe, but review recommended.")

if __name__ == "__main__":
    main()
