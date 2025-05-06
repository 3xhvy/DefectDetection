import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.utils import to_networkx
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def extract_ast_features(code_samples, language):
    """
    Extract features from AST for traditional ML models.
    
    Args:
        code_samples: List of code strings
        language: Tree-sitter language object
        
    Returns:
        numpy array of features for each code sample
    """
    from gnn_main import build_graph_from_code
    
    logger.info("Extracting AST features for traditional ML models...")
    features = []
    
    for code in tqdm(code_samples, desc="Extracting AST features"):
        try:
            # Build graph from code
            graph_data = build_graph_from_code(code, language)
            
            # Convert to networkx graph for feature extraction
            g = to_networkx(graph_data)
            
            # Basic graph statistics
            num_nodes = len(g.nodes())
            num_edges = len(g.edges())
            density = nx.density(g)
            
            # Node type statistics
            node_types = [d.get('type', '') for _, d in g.nodes(data=True)]
            unique_types = set(node_types)
            
            # Count different node types
            expr_count = sum(1 for t in node_types if 'expr' in t)
            decl_count = sum(1 for t in node_types if 'decl' in t)
            stmt_count = sum(1 for t in node_types if 'stmt' in t)
            func_count = sum(1 for t in node_types if 'func' in t)
            call_count = sum(1 for t in node_types if 'call' in t)
            if_count = sum(1 for t in node_types if 'if' in t)
            loop_count = sum(1 for t in node_types if any(loop in t for loop in ['for', 'while', 'loop']))
            
            # Structural features
            try:
                avg_degree = sum(dict(g.degree()).values()) / num_nodes if num_nodes > 0 else 0
                max_degree = max(dict(g.degree()).values()) if num_nodes > 0 else 0
            except:
                avg_degree = 0
                max_degree = 0
                
            # Try to compute centrality measures
            try:
                if num_nodes > 0:
                    centrality = nx.degree_centrality(g)
                    avg_centrality = sum(centrality.values()) / len(centrality)
                    max_centrality = max(centrality.values())
                else:
                    avg_centrality = 0
                    max_centrality = 0
            except:
                avg_centrality = 0
                max_centrality = 0
            
            # Try to compute path length
            try:
                if num_nodes > 1:
                    # Convert to undirected for path calculations if needed
                    if not nx.is_connected(g.to_undirected()):
                        avg_path_length = 0
                    else:
                        avg_path_length = nx.average_shortest_path_length(g.to_undirected())
                else:
                    avg_path_length = 0
            except:
                avg_path_length = 0
            
            # Depth features
            node_depths = {}
            root_nodes = [n for n in g.nodes() if g.in_degree(n) == 0]
            
            max_depth = 0
            avg_depth = 0
            
            if root_nodes:
                # BFS to calculate depth
                for root in root_nodes:
                    visited = {root: 0}  # node: depth
                    queue = [root]
                    
                    while queue:
                        node = queue.pop(0)
                        depth = visited[node]
                        
                        for neighbor in g.neighbors(node):
                            if neighbor not in visited:
                                visited[neighbor] = depth + 1
                                queue.append(neighbor)
                
                depths = list(visited.values())
                max_depth = max(depths)
                avg_depth = sum(depths) / len(depths)
            
            # Code complexity approximations
            cyclomatic_complexity = 1 + if_count + loop_count
            
            # Combine all features
            feature_vector = [
                num_nodes,                # Number of nodes
                num_edges,                # Number of edges
                density,                  # Graph density
                len(unique_types),        # Number of unique node types
                expr_count,               # Expression count
                decl_count,               # Declaration count
                stmt_count,               # Statement count
                func_count,               # Function-related count
                call_count,               # Function call count
                if_count,                 # If statement count
                loop_count,               # Loop count
                avg_degree,               # Average node degree
                max_degree,               # Maximum node degree
                avg_centrality,           # Average centrality
                max_centrality,           # Maximum centrality
                avg_path_length,          # Average path length
                max_depth,                # Maximum depth in AST
                avg_depth,                # Average depth in AST
                cyclomatic_complexity,    # Cyclomatic complexity approximation
                
                # Ratios and derived features
                expr_count / num_nodes if num_nodes > 0 else 0,
                decl_count / num_nodes if num_nodes > 0 else 0,
                stmt_count / num_nodes if num_nodes > 0 else 0,
                func_count / num_nodes if num_nodes > 0 else 0,
                call_count / num_nodes if num_nodes > 0 else 0,
                if_count / num_nodes if num_nodes > 0 else 0,
                loop_count / num_nodes if num_nodes > 0 else 0,
                
                # Additional derived features
                (if_count + loop_count) / num_nodes if num_nodes > 0 else 0,
                num_edges / num_nodes if num_nodes > 0 else 0,
            ]
            
            features.append(feature_vector)
        except Exception as e:
            logger.warning(f"Error extracting features: {str(e)}")
            # Add a default feature vector
            features.append([0] * 30)  # Match the feature dimension
    
    return np.array(features)

class LogisticRegressionModel:
    """Wrapper for scikit-learn's Logistic Regression model"""
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class RandomForestModel:
    """Wrapper for scikit-learn's Random Forest model"""
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def feature_importances(self):
        """Get feature importances from the model"""
        return self.model.feature_importances_
