import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
import networkx as nx
import numpy as np
from tree_sitter import Language, Parser
import argparse
import logging
import os
import random
from tqdm import tqdm
import pandas as pd
import datasets
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# 1. Build the Graph from AST
def build_graph_from_code(source_code, language):
    """Convert source code to graph using AST (Abstract Syntax Tree)."""
    parser = Parser()
    try:
        parser.set_language(language)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node

        graph = nx.Graph()
        node_count = 0

        def traverse(node, parent=None):
            nonlocal node_count
            graph.add_node(node_count,
                          type=node.type,
                          start_byte=node.start_byte,
                          end_byte=node.end_byte)
            if parent is not None:
                graph.add_edge(parent, node_count)
            current_node = node_count
            node_count += 1

            for child in node.children:
                traverse(child, current_node)

        traverse(root_node)
        return from_networkx(graph)
    except Exception as e:
        logger.warning(f"AST parsing failed: {str(e)}")
        # Return minimal fallback graph
        graph = nx.Graph()
        graph.add_node(0, type="fallback", start_byte=0, end_byte=0)
        return from_networkx(graph)

# 2. GNN Model
class GNNDefectDetectionModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GNNDefectDetectionModel, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        # Last GCN layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))

        # Classification layers
        self.fc1 = nn.Linear(out_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = nn.Linear(hidden_channels // 2, 1)
        self.dropout = nn.Dropout(0.2)  # Increased dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolution layers with batch normalization
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropout(x)

        # Last convolution layer
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        x = torch.relu(x)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Classification layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x

# 3. Dataset Class
class CodeDefectDataset(Dataset):
    def __init__(self, code_samples, labels, language, args):
        super().__init__()
        self.code_samples = code_samples
        self.labels = [self._clean_label(l) for l in labels]  # Clean labels
        self.language = language
        self.args = args

        # Print class distribution for debugging
        label_counts = {}
        for label in self.labels:
            label_counts[float(label)] = label_counts.get(float(label), 0) + 1
        logger.info(f"Dataset created with label distribution: {label_counts}")

    def _clean_label(self, label):
        """Convert label to 0 or 1, handling special cases"""
        if isinstance(label, (int, float)):
            return float(label)
        # Handle string labels
        label = str(label).strip().lower()
        if label in ('1', 'true', 't', 'yes', 'y', 'positive'):
            return 1.0
        elif label in ('0', 'false', 'f', 'no', 'n', 'negative'):
            return 0.0
        else:
            return 0.0  # Default to negative class for unparseable labels

    def len(self):
        return len(self.code_samples)

    def get(self, idx):
        code = self.code_samples[idx]
        try:
            graph_data = build_graph_from_code(code, self.language)

            # Create more informative node features
            node_features = []
            for i, node_type in enumerate(graph_data.type):
                # Create a more informative feature vector
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
            graph_data.y = torch.tensor([float(self.labels[idx])], dtype=torch.float)
            return graph_data
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {str(e)}")
            # Fallback graph
            graph = nx.Graph()
            graph.add_node(0, type="fallback", start_byte=0, end_byte=0)
            graph_data = from_networkx(graph)
            graph_data.x = torch.tensor([[0.0]], dtype=torch.float)
            graph_data.y = torch.tensor([float(self.labels[idx])], dtype=torch.float)
            return graph_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args, model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")

    # Debug: Count predictions
    correct_predictions = 0
    total_samples = 0
    label_counts = {0: 0, 1: 0}

    # For gradient accumulation
    optimizer.zero_grad()
    accumulated_steps = 0

    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)

        # Count labels in this batch
        for label in batch.y.squeeze().cpu().numpy():
            label_counts[int(label)] += 1

        output = model(batch)
        loss = criterion(output.squeeze(), batch.y.squeeze())

        # Scale loss by gradient accumulation steps
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        # Debug: Count correct predictions
        preds = (output.squeeze() > 0.5).float()
        correct_predictions += (preds == batch.y.squeeze()).sum().item()
        total_samples += batch.y.size(0)

        # Update total loss (use the unscaled loss for reporting)
        batch_loss = loss.item() * args.gradient_accumulation_steps
        total_loss += batch_loss
        progress_bar.set_postfix({'loss': batch_loss})

        # Perform optimizer step after accumulating gradients
        accumulated_steps += 1
        if accumulated_steps == args.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            accumulated_steps = 0

    # Handle any remaining gradients
    if accumulated_steps > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    # Debug: Print training statistics
    print(f"Training Statistics:")
    print(f"Label distribution: {label_counts}")
    print(f"Training accuracy: {correct_predictions/total_samples:.4f}")
    print(f"Average loss: {total_loss/len(train_loader):.6f}")

    return total_loss / len(train_loader)



def evaluate(args, model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Debug: Print model parameters
    print("Model Parameter Stats:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean_val = param.data.mean().item()
            # Check if tensor has more than 1 element before calculating std
            if param.numel() > 1:
                std_val = param.data.std().item()
                print(f"  {name}: mean={mean_val:.4f}, std={std_val:.4f}")
            else:
                print(f"  {name}: mean={mean_val:.4f}, std=N/A (single value)")

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output.squeeze(), batch.y.squeeze())
            total_loss += loss.item()

            preds = (output.squeeze() > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # Calculate precision, recall, and F1 score
    true_positives = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    false_positives = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    false_negatives = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)

    # Avoid division by zero
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Print confusion matrix-like info
    true_negatives = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)

    logger.info("Evaluation Statistics:")
    logger.info("  True Negative (0,0): %d", true_negatives)
    logger.info("  False Positive (0,1): %d", false_positives)
    logger.info("  False Negative (1,0): %d", false_negatives)
    logger.info("  True Positive (1,1): %d", true_positives)
    logger.info("  Label distribution: 0s=%d, 1s=%d", all_labels.count(0), all_labels.count(1))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return total_loss / len(eval_loader), metrics


def main():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--hidden_channels", default=256, type=int)
    parser.add_argument("--out_channels", default=128, type=int)
    parser.add_argument("--num_layers", default=4, type=int)  # Increased from 3

    # Training parameters
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--train_batch_size", default=64, type=int)  # Increased from 32
    parser.add_argument("--eval_batch_size", default=64, type=int)  # Increased from 32
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", default=1e-3, type=float)  # Increased from 2e-4
    parser.add_argument("--weight_decay", default=0.001, type=float)  # Decreased from 0.01
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--epochs", default=30, type=int)  # Increased from 10
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./gnn_checkpoints", type=str)
    parser.add_argument("--balance_dataset", action='store_true', help="Balance dataset by undersampling majority class")
    parser.add_argument("--class_weights", action='store_true', help="Use class weights in loss function")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',
                      level=logging.INFO)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info("device: %s", device)

    # Set seed
    set_seed(args.seed)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load C language for parsing
    try:
        language = Language('/home/hohoanghvy/tree-sitter-container/build/my-languages.so', 'c')
    except:
        logger.error("Failed to load C language. Make sure to build tree-sitter-c first.")
        return

    # Training
    if args.do_train:
        # Load dataset - following the same approach as in vulrepair_main.py
        logger.info("Loading training dataset...")
        train_data_whole = datasets.load_dataset("google/code_x_glue_cc_defect_detection", split="train")
        df = pd.DataFrame({"code": train_data_whole["func"], "label": train_data_whole["target"]})

        logger.info("Dataset before split:")
        logger.info("Total samples: %d", len(df))

        # Use the same test split ratio as in vulrepair_main.py
        train_data, val_data = train_test_split(df, test_size=0.1238, random_state=args.seed)

        # Create datasets
        train_dataset = CodeDefectDataset(
            train_data["code"].tolist(),
            train_data["label"].tolist(),
            language,
            args
        )
        eval_dataset = CodeDefectDataset(
            val_data["code"].tolist(),
            val_data["label"].tolist(),
            language,
            args
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset,
                                batch_size=args.train_batch_size,
                                shuffle=True,
                                num_workers=8,  # Increased from 4
                                pin_memory=True,  # Enable faster data transfer to GPU
                                prefetch_factor=2,  # Prefetch 2 batches per worker
                                persistent_workers=True)  # Keep workers alive between epochs
        eval_loader = DataLoader(eval_dataset,
                               batch_size=args.eval_batch_size,
                               shuffle=False,
                               num_workers=8,  # Increased from 4
                               pin_memory=True,  # Enable faster data transfer to GPU
                               prefetch_factor=2)  # Prefetch 2 batches per worker

        # Initialize model
        model = GNNDefectDetectionModel(
            in_channels=7,  # Feature dimension from node features
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            num_layers=args.num_layers
        ).to(device)

        # Initialize optimizer and loss
        optimizer = optim.AdamW(model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)

        # Use class weights if there's imbalance
        class_counts = train_data['label'].value_counts().to_dict()
        if len(class_counts) > 1:  # Only if we have both classes
            total_samples = len(train_data)
            weight_0 = total_samples / (2 * class_counts.get(0, 1))
            weight_1 = total_samples / (2 * class_counts.get(1, 1))
            pos_weight = torch.tensor([weight_1 / weight_0]).to(device)
            logger.info("Using class weights - pos_weight: %.4f", pos_weight.item())
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # Training loop
        best_accuracy = 0
        best_f1 = 0
        for epoch in range(args.epochs):
            log.info(f"----EPOCH {epoch+1}-----------------------------------")
            # Train
            train_loss = train(args, model, train_loader, optimizer, criterion, device)

            # Evaluate
            eval_loss, eval_metrics = evaluate(args, model, eval_loader, criterion, device)
            eval_accuracy = eval_metrics['accuracy']
            eval_f1 = eval_metrics['f1']

            logger.info(f"Epoch {epoch+1}/{args.epochs}:")
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Eval Loss: {eval_loss:.4f}")
            logger.info(f"Eval Accuracy: {eval_accuracy:.4f}")
            logger.info(f"Eval Precision: {eval_metrics['precision']:.4f}")
            logger.info(f"Eval Recall: {eval_metrics['recall']:.4f}")
            logger.info(f"Eval F1: {eval_f1:.4f}")

            # Save best model based on F1 score (better for imbalanced datasets)
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                best_accuracy = eval_accuracy
                logger.info("  " + "*" * 20)
                logger.info("  Best F1: %.4f", best_f1)
                logger.info("  " + "*" * 20)
                checkpoint_prefix = 'checkpoint-best-f1'
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt"))
                logger.info("Saving model checkpoint to %s", output_dir)

    # Testing
    if args.do_test:
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
                               batch_size=args.eval_batch_size,
                               shuffle=False,
                               num_workers=8,  # Increased from 4
                               pin_memory=True,  # Enable faster data transfer to GPU
                               prefetch_factor=2)  # Prefetch 2 batches per worker

        # Load best model
        model = GNNDefectDetectionModel(
            in_channels=7,
            hidden_channels=128,  # Changed from 256 to match saved model
            out_channels=128,
            num_layers=args.num_layers
        ).to(device)

        # Load model weights
        model_path = os.path.join(args.output_dir, 'checkpoint-best-f1/model_best.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            logger.info("Loaded model from %s", model_path)
        else:
            logger.error("No saved model found at %s", model_path)
            return

        # Evaluate on test set
        criterion = nn.BCEWithLogitsLoss()
        test_loss, test_metrics = evaluate(args, model, test_loader, criterion, device)
        logger.info(f"Test Results:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
