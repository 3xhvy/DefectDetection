import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool
import logging

logger = logging.getLogger(__name__)

class GCNModel(nn.Module):
    """Graph Convolutional Network model for code vulnerability detection"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super(GCNModel, self).__init__()
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        # Classification layers
        self.fc1 = nn.Linear(out_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = nn.Linear(hidden_channels // 2, 1)
        
        # Batch normalization for classification layers
        self.bn_fc1 = nn.BatchNorm1d(hidden_channels)
        self.bn_fc2 = nn.BatchNorm1d(hidden_channels // 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = F.relu(self.input_proj(x))
        
        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification layers with batch normalization
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x.squeeze(-1)

class GraphSAGEModel(nn.Module):
    """GraphSAGE model for code vulnerability detection"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
        # Classification layers
        self.fc1 = nn.Linear(out_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = nn.Linear(hidden_channels // 2, 1)
        
        # Batch normalization for classification layers
        self.bn_fc1 = nn.BatchNorm1d(hidden_channels)
        self.bn_fc2 = nn.BatchNorm1d(hidden_channels // 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = F.relu(self.input_proj(x))
        
        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification layers with batch normalization
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x.squeeze(-1)

class CombinedGNNModel(nn.Module):
    """Combined GNN model that uses multiple GNN layers in parallel"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super(CombinedGNNModel, self).__init__()
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GCN branch
        self.gcn_convs = nn.ModuleList()
        self.gcn_batch_norms = nn.ModuleList()
        
        # GraphSAGE branch
        self.sage_convs = nn.ModuleList()
        self.sage_batch_norms = nn.ModuleList()
        
        # First layer
        self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
        self.gcn_batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        self.sage_convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.sage_batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.gcn_batch_norms.append(nn.BatchNorm1d(hidden_channels))
            
            self.sage_convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.sage_batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Last layer
        self.gcn_convs.append(GCNConv(hidden_channels, out_channels))
        self.gcn_batch_norms.append(nn.BatchNorm1d(out_channels))
        
        self.sage_convs.append(SAGEConv(hidden_channels, out_channels))
        self.sage_batch_norms.append(nn.BatchNorm1d(out_channels))
        
        # Classification layers
        self.fc1 = nn.Linear(out_channels * 2, hidden_channels)  # *2 because we concatenate GCN and GraphSAGE outputs
        self.fc2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc3 = nn.Linear(hidden_channels // 2, 1)
        
        # Batch normalization for classification layers
        self.bn_fc1 = nn.BatchNorm1d(hidden_channels)
        self.bn_fc2 = nn.BatchNorm1d(hidden_channels // 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = F.relu(self.input_proj(x))
        
        # GCN branch
        x_gcn = x
        for i in range(self.num_layers):
            x_gcn = self.gcn_convs[i](x_gcn, edge_index)
            x_gcn = self.gcn_batch_norms[i](x_gcn)
            x_gcn = F.relu(x_gcn)
            x_gcn = self.dropout(x_gcn)
        
        # GraphSAGE branch
        x_sage = x
        for i in range(self.num_layers):
            x_sage = self.sage_convs[i](x_sage, edge_index)
            x_sage = self.sage_batch_norms[i](x_sage)
            x_sage = F.relu(x_sage)
            x_sage = self.dropout(x_sage)
        
        # Global pooling for both branches
        x_gcn = global_mean_pool(x_gcn, batch)
        x_sage = global_mean_pool(x_sage, batch)
        
        # Concatenate the outputs from both branches
        x = torch.cat([x_gcn, x_sage], dim=1)
        
        # Classification layers with batch normalization
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x.squeeze(-1)
