import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from eval_metrics_util import evaluate_and_plot
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, TensorDataset

# Import your model classes from the correct files
from sequence_models import BiLSTMModel
from graph_models import GCNModel, GraphSAGEModel
from gnn_main import GATDefectDetectionModel
from improved_gnn_model import ImprovedGNNModel

# Map model names to (class, checkpoint, data_type)
model_info = [
    ("BiLSTM", BiLSTMModel, "best_BiLSTM.pt", "sequence"),
    ("GCN", GCNModel, "best_GCNModel.pt", "graph"),
    ("GraphSAGE", GraphSAGEModel, "best_GraphSAGEModel.pt", "graph"),
    ("ImprovedGNN", ImprovedGNNModel, "best_ImprovedGNNModel.pt", "graph"),
    ("GAT", GATDefectDetectionModel, "best_GATDefectDetectionModel.pt", "graph"),
]

def load_test_data():
    """Load the test dataset"""
    print("Loading test dataset...")
    test_data = load_dataset("google/code_x_glue_cc_defect_detection", split="test")
    print(f"Test dataset loaded with {len(test_data)} samples")
    return test_data

def prepare_sequence_data(test_data, batch_size=128):
    """Prepare data for sequence models (BiLSTM)"""
    from sequence_models import tokenize_and_pad
    
    # Tokenize and prepare data
    print("Preparing sequence data...")
    inputs, labels = tokenize_and_pad(test_data)
    
    # Create DataLoader
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

def prepare_graph_data(test_data, batch_size=128):
    """Prepare data for graph models (GCN, GraphSAGE, GAT)"""
    from gnn_main import code_to_graph
    
    print("Preparing graph data...")
    graph_data = []
    
    for i, example in enumerate(test_data):
        if i % 100 == 0:
            print(f"Processing graph {i}/{len(test_data)}")
        
        code = example['func']
        label = example['target']
        
        # Convert code to graph
        try:
            graph = code_to_graph(code)
            if graph is not None:
                # Create PyG Data object
                data = Data(
                    x=torch.tensor(graph['node_features'], dtype=torch.float),
                    edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
                    y=torch.tensor(label, dtype=torch.float)
                )
                graph_data.append(data)
        except Exception as e:
            print(f"Error processing graph {i}: {e}")
    
    # Create DataLoader
    dataloader = DataLoader(graph_data, batch_size=batch_size, shuffle=False, 
                           collate_fn=lambda data_list: Batch.from_data_list(data_list))
    
    return dataloader

def evaluate_model(model_class, checkpoint_path, model_name, data_type, device="cuda"):
    print(f"\n===== Evaluating {model_name} =====")
    try:
        # Load test data
        test_data = load_test_data()
        
        # Prepare data based on model type
        if data_type == "sequence":
            test_loader = prepare_sequence_data(test_data)
        elif data_type == "graph":
            test_loader = prepare_graph_data(test_data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Instantiate model with correct parameters based on model type
        if model_name == "BiLSTM":
            # Assuming BiLSTM needs vocab_size and embedding_dim
            vocab_size = 100000  # Adjust based on your actual vocab size
            embedding_dim = 128  # Adjust based on your model
            model = model_class(vocab_size=vocab_size, embedding_dim=embedding_dim)
        elif model_name == "ImprovedGNN":
            # Assuming ImprovedGNN has specific init parameters
            model = model_class(hidden_channels=64)  # Adjust parameters as needed
        else:
            # For other models, use default initialization
            model = model_class()
        
        # Load model weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Run predictions
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                # Adjust unpacking based on data type
                if data_type == "sequence":
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                elif data_type == "graph":
                    # For graph data, adjust based on your dataloader format
                    data = batch.to(device)
                    labels = data.y
                    outputs = model(data)
                
                # Process outputs
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                # Store results
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)
                y_score.extend(probs)
        
        # Evaluate and plot results
        evaluate_and_plot(y_true, y_pred, y_score, model_name=model_name)
        
        # Save results to CSV
        results_file = f"{model_name}_evaluation_results.csv"
        with open(results_file, 'w') as f:
            f.write("true_label,predicted_label,prediction_score\n")
            for true, pred, score in zip(y_true, y_pred, y_score):
                f.write(f"{true},{pred},{score}\n")
        print(f"Results saved to {results_file}")
        
        return {
            "model_name": model_name,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score
        }
    
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Ask which model to evaluate
    print("Available models to evaluate:")
    for i, (name, _, _, _) in enumerate(model_info):
        print(f"{i+1}. {name}")
    
    choice = input("Enter model number to evaluate (or 'all' for all models): ")
    
    if choice.lower() == 'all':
        models_to_evaluate = model_info
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_info):
                models_to_evaluate = [model_info[idx]]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(model_info)}")
                exit(1)
        except ValueError:
            print("Invalid input. Please enter a number or 'all'")
            exit(1)
    
    results = {}
    for model_name, model_class, ckpt, data_type in models_to_evaluate:
        try:
            print(f"\nAttempting to evaluate {model_name} model...")
            result = evaluate_model(model_class, ckpt, model_name, data_type, device)
            if result:
                results[model_name] = result
        except Exception as e:
            print(f"Could not evaluate {model_name}: {e}")
    
    # Summarize all results
    print("\n===== EVALUATION SUMMARY =====")
    for model_name in results:
        print(f"{model_name}: Evaluated successfully")
