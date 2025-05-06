import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Import your model classes
from sequence_models import BiLSTMModel
from graph_models import GCNModel, GraphSAGEModel
from gnn_main import GATDefectDetectionModel
from improved_gnn_model import ImprovedGNNModel

# Model checkpoints
MODEL_PATHS = {
    "BiLSTM": "best_BiLSTM.pt",
    "GCN": "best_GCNModel.pt",
    "GraphSAGE": "best_GraphSAGEModel.pt", 
    "GAT": "best_GATDefectDetectionModel.pt",
    "ImprovedGNN": "best_ImprovedGNNModel.pt"
}

# Model parameters (based on your model definitions)
MODEL_PARAMS = {
    "BiLSTM": {
        "vocab_size": 100000,
        "embedding_dim": 128,
        "hidden_dim": 64,
        "output_dim": 1,
        "n_layers": 2,
        "dropout": 0.2
    },
    "GCN": {
        "in_channels": 128,
        "hidden_channels": 64,
        "out_channels": 32,
        "num_layers": 2,
        "dropout": 0.2
    },
    "GraphSAGE": {
        "in_channels": 128,
        "hidden_channels": 64,
        "out_channels": 32,
        "num_layers": 2,
        "dropout": 0.2
    },
    "GAT": {
        "in_channels": 128,
        "hidden_channels": 64,
        "out_channels": 32,
        "num_layers": 2,
        "dropout": 0.2
    },
    "ImprovedGNN": {
        "in_channels": 128,
        "hidden_channels": 64,
        "out_channels": 32,
        "num_layers": 2,
        "dropout": 0.2
    }
}

def load_model(model_name, device="cuda"):
    """Load a model by name"""
    print(f"\n===== Loading {model_name} model =====")
    
    try:
        # Get model parameters
        params = MODEL_PARAMS[model_name]
        
        # Initialize the appropriate model class with correct parameters
        if model_name == "BiLSTM":
            model = BiLSTMModel(
                vocab_size=params["vocab_size"],
                embedding_dim=params["embedding_dim"],
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                n_layers=params["n_layers"],
                dropout=params["dropout"]
            )
        elif model_name == "GCN":
            model = GCNModel(
                in_channels=params["in_channels"],
                hidden_channels=params["hidden_channels"],
                out_channels=params["out_channels"],
                num_layers=params["num_layers"],
                dropout=params["dropout"]
            )
        elif model_name == "GraphSAGE":
            model = GraphSAGEModel(
                in_channels=params["in_channels"],
                hidden_channels=params["hidden_channels"],
                out_channels=params["out_channels"],
                num_layers=params["num_layers"],
                dropout=params["dropout"]
            )
        elif model_name == "GAT":
            model = GATDefectDetectionModel(
                in_channels=params["in_channels"],
                hidden_channels=params["hidden_channels"],
                out_channels=params["out_channels"]
            )
        elif model_name == "ImprovedGNN":
            model = ImprovedGNNModel(
                in_channels=params["in_channels"],
                hidden_channels=params["hidden_channels"]
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load the model weights
        model_path = MODEL_PATHS[model_name]
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        print(f"{model_name} model loaded successfully!")
        return model
    
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_model_summary(model_name, model):
    """Print a summary of the model architecture"""
    if model is None:
        return
    
    print(f"\n===== {model_name} Model Summary =====")
    print(f"Model type: {type(model).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print model structure
    print("\nModel structure:")
    print(model)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # List available models
    print("\nAvailable models to evaluate:")
    for i, model_name in enumerate(MODEL_PATHS.keys()):
        print(f"{i+1}. {model_name}")
    
    # Get user choice
    choice = input("\nEnter model number to evaluate (or 'all' for all models): ")
    
    if choice.lower() == 'all':
        models_to_evaluate = list(MODEL_PATHS.keys())
    else:
        try:
            idx = int(choice) - 1
            models_list = list(MODEL_PATHS.keys())
            if 0 <= idx < len(models_list):
                models_to_evaluate = [models_list[idx]]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(models_list)}")
                return
        except ValueError:
            print("Invalid input. Please enter a number or 'all'")
            return
    
    # Evaluate each selected model
    for model_name in models_to_evaluate:
        model = load_model(model_name, device)
        if model:
            print_model_summary(model_name, model)
            print(f"\n{model_name} model loaded and ready for inference!")
            print("To run inference, you would need to:")
            print("1. Prepare your test data in the appropriate format")
            print("2. Run model.forward() on your test data")
            print("3. Process the outputs to get predictions")
            print("4. Compare predictions with ground truth to compute metrics")
            
            print("\nModel checkpoint information:")
            print(f"- Path: {MODEL_PATHS[model_name]}")
            print(f"- Size: {os.path.getsize(MODEL_PATHS[model_name]) / (1024*1024):.2f} MB")
    
    print("\n===== EVALUATION SUMMARY =====")
    print("Models loaded successfully:")
    for model_name in models_to_evaluate:
        print(f"- {model_name}")
    
    print("\nTo perform detailed evaluation with test data, you would need to:")
    print("1. Load your test dataset")
    print("2. Process it into the format expected by each model")
    print("3. Run inference and compute metrics")
    
    print("\nFor a full evaluation script that handles these steps, you may need to:")
    print("1. Identify the correct data processing functions in your codebase")
    print("2. Create model-specific data preparation functions")
    print("3. Implement evaluation loops for each model type")

if __name__ == "__main__":
    import os
    main()
