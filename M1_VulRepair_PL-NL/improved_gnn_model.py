import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import from_networkx, to_networkx
import networkx as nx
import numpy as np
from tree_sitter import Language, Parser
import os
import logging

logger = logging.getLogger(__name__)

# 1. Improved Graph Construction from AST
def build_enhanced_graph_from_code(source_code, language):
    """Convert source code to a more informative graph using AST with additional edges."""
    parser = Parser()
    try:
        parser.set_language(language)
        tree = parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node

        # Create a directed graph to capture more structural information
        graph = nx.DiGraph()
        node_count = 0
        node_map = {}  # Map tree-sitter nodes to graph indices

        # First pass: add all nodes
        def add_nodes(node, parent_idx=None):
            nonlocal node_count
            current_idx = node_count
            node_map[node_count] = current_idx  # Use node_count as key instead of node.id
            
            # Add node with more detailed attributes
            graph.add_node(
                current_idx,
                type=node.type,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_point=(node.start_point[0], node.start_point[1]),
                end_point=(node.end_point[0], node.end_point[1]),
                text=source_code[node.start_byte:node.end_byte] if node.start_byte < len(source_code) else "",
                is_named=node.is_named,
                depth=0 if parent_idx is None else graph.nodes[parent_idx]['depth'] + 1
            )
            
            # Add parent-child edge if parent exists
            if parent_idx is not None:
                graph.add_edge(parent_idx, current_idx, edge_type='parent-child')
            
            node_count += 1
            
            # Process children
            for child in node.children:
                add_nodes(child, current_idx)
        
        # Add all nodes first
        add_nodes(root_node)
        
        # Second pass: add additional edges for data flow
        # Add sibling edges (nodes with same parent)
        for node in graph.nodes():
            predecessors = list(graph.predecessors(node))
            if predecessors:
                parent = predecessors[0]
                siblings = [n for n in graph.successors(parent) if n != node]
                for sibling in siblings:
                    graph.add_edge(node, sibling, edge_type='sibling')
        
        # Add next-token edges (sequential relationship)
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: (x[1]['start_byte'], x[1]['end_byte']))
        for i in range(len(sorted_nodes) - 1):
            graph.add_edge(sorted_nodes[i][0], sorted_nodes[i+1][0], edge_type='next-token')
        
        # Convert to PyTorch Geometric data
        pyg_graph = from_networkx(graph)
        return pyg_graph
    
    except Exception as e:
        logger.warning(f"AST parsing failed: {str(e)}")
        # Return minimal fallback graph
        graph = nx.DiGraph()
        graph.add_node(0, type="fallback", start_byte=0, end_byte=0, 
                     start_point=(0,0), end_point=(0,0), text="", is_named=False, depth=0)
        return from_networkx(graph)

# 2. Enhanced Feature Engineering
def enrich_node_features_advanced(graph_data, code_snippet=None):
    """Create more sophisticated node features for better code representation."""
    # Process existing features
    if hasattr(graph_data, 'type'):
        if isinstance(graph_data.type, (np.ndarray, torch.Tensor)):
            node_types = graph_data.type.tolist()
        else:
            node_types = list(graph_data.type)
    else:
        node_types = []
    
    # Create node features
    node_features = []
    for i, node_type in enumerate(node_types):
        # Basic features
        node_type_hash = hash(node_type) % 1000
        
        # Node type one-hot encoding
        unique_types = ['identifier', 'function_definition', 'call_expression', 'argument_list', 
                       'binary_expression', 'if_statement', 'for_statement', 'while_statement',
                       'assignment_expression', 'return_statement', 'expression_statement']
        type_one_hot = [1.0 if t in node_type else 0.0 for t in unique_types]
        
        # Position and size features
        start_byte = float(graph_data.start_byte[i]) if hasattr(graph_data, 'start_byte') else 0
        end_byte = float(graph_data.end_byte[i]) if hasattr(graph_data, 'end_byte') else 0
        size = end_byte - start_byte
        
        # Text-based features if available
        text_length = 0
        has_number = 0
        has_string = 0
        if hasattr(graph_data, 'text') and i < len(graph_data.text):
            text = graph_data.text[i]
            text_length = len(text) / 100.0  # Normalized text length
            has_number = 1.0 if any(c.isdigit() for c in text) else 0.0
            has_string = 1.0 if '"' in text or "'" in text else 0.0
        
        # Depth feature
        depth = float(graph_data.depth[i]) / 10.0 if hasattr(graph_data, 'depth') else 0.0
        
        # Combine all features
        feature = [
            node_type_hash / 1000.0,  # Normalized hash value
            float(len(node_type)) / 50.0,  # Normalized length of type name
            start_byte / 10000.0,  # Normalized position
            size / 1000.0,  # Normalized size
            depth,  # Normalized depth in AST
            1.0 if "expr" in node_type else 0.0,  # Is expression
            1.0 if "decl" in node_type else 0.0,  # Is declaration
            1.0 if "stmt" in node_type else 0.0,  # Is statement
            1.0 if "func" in node_type else 0.0,  # Is function related
            1.0 if "call" in node_type else 0.0,  # Is function call
            1.0 if "if" in node_type else 0.0,  # Is conditional
            1.0 if "loop" in node_type or "for" in node_type or "while" in node_type else 0.0,  # Is loop
            text_length,
            has_number,
            has_string
        ]
        
        # Add type one-hot encoding
        feature.extend(type_one_hot)
        
        node_features.append(feature)
    
    # Convert to tensor
    if node_features:
        graph_data.x = torch.tensor(node_features, dtype=torch.float)
    else:
        # Fallback for empty features
        graph_data.x = torch.zeros((1, 26), dtype=torch.float)
    
    return graph_data

# 3. Improved GNN Model with Multi-head Attention and Residual Connections
class ImprovedGNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32, num_layers=2, dropout=0.3, heads=2, 
                 use_checkpointing=True, use_mixed_precision=True, memory_efficient=True):
        super(ImprovedGNNModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing
        self.use_mixed_precision = use_mixed_precision
        self.memory_efficient = memory_efficient
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Multi-head GAT layers with residual connections
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, add_self_loops=memory_efficient))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, add_self_loops=memory_efficient))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        self.convs.append(GATConv(hidden_channels, out_channels // heads, heads=heads, add_self_loops=memory_efficient))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        # Classification layers with skip connections
        self.fc1 = nn.Linear(out_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = nn.Linear(hidden_channels // 2, 1)
        
        # Batch normalization for classification layers
        self.bn_fc1 = nn.BatchNorm1d(hidden_channels)
        self.bn_fc2 = nn.BatchNorm1d(hidden_channels // 2)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def _conv_op(self, conv, x, edge_index):
        """Wrapper for convolution operation with gradient checkpointing support"""
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(conv, x, edge_index, use_reentrant=False)
        else:
            return conv(x, edge_index)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Use mixed precision if enabled
        dtype = torch.float16 if self.use_mixed_precision and self.training else torch.float32
        
        # Input projection
        x = F.relu(self.input_proj(x))
        
        # Graph convolution layers with residual connections
        for i in range(self.num_layers):
            identity = x
            
            # Apply convolution with optional checkpointing
            if self.use_mixed_precision and self.training:
                with torch.amp.autocast('cuda'):
                    x = self._conv_op(self.convs[i], x, edge_index)
            else:
                x = self._conv_op(self.convs[i], x, edge_index)
                
            x = self.batch_norms[i](x)
            
            # Apply residual connection if shapes match
            if i < self.num_layers - 1 and x.size(-1) == identity.size(-1):
                x = x + identity
                
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Memory-efficient pooling
        if self.memory_efficient:
            x = global_mean_pool(x, batch)
        else:
            # Use multiple pooling methods and concatenate for better performance
            # but more memory intensive
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
            # Adjust the first FC layer to handle the concatenated features
            if not hasattr(self, '_adjusted_fc1'):
                self.fc1 = nn.Linear(x.size(1), self.fc1.out_features)
                self._adjusted_fc1 = True
        
        # Classification layers with batch normalization
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        x = self.fc3(x)
        
        return x.squeeze(-1)

# 4. Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Add a new class for memory-efficient training
class MemoryEfficientTrainer:
    def __init__(self, model, optimizer, loss_fn, device, use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
        
    def train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        if self.use_mixed_precision:
            with torch.amp.autocast('cuda'):
                out = self.model(batch)
                loss = self.loss_fn(out, batch.y.float())
            
            # Scale gradients and optimize
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y.float())
            loss.backward()
            self.optimizer.step()
            
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            
            if self.use_mixed_precision:
                with torch.amp.autocast('cuda'):
                    out = self.model(batch)
                    loss = self.loss_fn(out, batch.y.float())
            else:
                out = self.model(batch)
                loss = self.loss_fn(out, batch.y.float())
                
            total_loss += loss.item() * batch.num_graphs
            pred = (out > 0).float()
            correct += int((pred == batch.y).sum())
            total += batch.y.size(0)
            
        return total_loss / total, correct / total
