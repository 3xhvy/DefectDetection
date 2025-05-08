#!/usr/bin/env python3
"""
Code Graph Visualization Tool

This script visualizes the AST (Abstract Syntax Tree) graph representation 
of code used in the vulnerability detection GNN model.
"""

import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tree_sitter import Language, Parser

# Import functions from the project if available
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gnn_main import build_graph_from_code, enrich_node_features
    IMPORT_SUCCESS = True
except ImportError:
    print("Warning: Could not import functions from gnn_main.py")
    print("Will use standalone implementations instead.")
    IMPORT_SUCCESS = False

# Standalone implementation of build_graph_from_code
def build_detailed_graph(source_code, language):
    """Convert source code to graph using AST with detailed node information"""
    parser = Parser()
    parser.set_language(language)
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node
    
    graph = nx.Graph()
    node_count = 0
    node_map = {}  # To store original nodes for reference
    
    def traverse(node, parent=None):
        nonlocal node_count
        # Extract code snippet for this node
        start, end = node.start_byte, node.end_byte
        code_snippet = source_code[start:end]
        if len(code_snippet) > 20:  # Truncate long snippets
            code_snippet = code_snippet[:17] + "..."
        
        # Add node with detailed information
        graph.add_node(node_count,
                      type=node.type,
                      start_byte=node.start_byte,
                      end_byte=node.end_byte,
                      code=code_snippet)
        
        # Store original node for reference
        node_map[node_count] = node
        
        if parent is not None:
            graph.add_edge(parent, node_count, relationship="parent-child")
        
        current_node = node_count
        node_count += 1
        
        for child in node.children:
            traverse(child, current_node)
    
    traverse(root_node)
    return graph, node_map

def visualize_ast_graph(graph, node_map, max_nodes=30, output_file=None):
    """Visualize AST graph with node types and code snippets"""
    # If graph is too large, create a simplified version
    if len(graph.nodes) > max_nodes:
        print(f"Graph is too large ({len(graph.nodes)} nodes). Showing only the first {max_nodes} nodes.")
        # Get a subgraph of the first max_nodes nodes
        nodes = list(graph.nodes)[:max_nodes]
        graph = graph.subgraph(nodes)
    
    plt.figure(figsize=(20, 16))
    
    # Create a layout for the graph
    pos = nx.spring_layout(graph, seed=42, k=0.8)
    
    # Create node labels with type and code snippet
    node_labels = {}
    for node in graph.nodes:
        node_type = graph.nodes[node]['type']
        code = graph.nodes[node].get('code', '')
        if code and len(code) > 0:
            label = f"{node}: {node_type}\n'{code}'"
        else:
            label = f"{node}: {node_type}"
        node_labels[node] = label
    
    # Draw the graph
    nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.7, edge_color='gray')
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10, font_family='sans-serif')
    
    plt.title("Abstract Syntax Tree (AST) Graph Representation", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_file}")
    else:
        plt.show()
    
    return node_labels

def explain_node_types(graph):
    """Print explanations for node types in the graph"""
    # Create a dictionary explaining common node types in C AST
    node_type_explanations = {
        "translation_unit": "The root node representing the entire source file",
        "function_definition": "A function declaration with its implementation",
        "parameter_list": "List of parameters for a function",
        "parameter_declaration": "Declaration of a single parameter",
        "compound_statement": "A block of code enclosed in braces {}",
        "declaration": "Variable or type declaration",
        "init_declarator": "Declaration with initialization",
        "declarator": "The name and type being declared",
        "array_declarator": "Declaration of an array",
        "primitive_type": "Basic types like int, char, etc.",
        "call_expression": "Function call",
        "argument_list": "Arguments passed to a function",
        "string_literal": "String constant in quotes",
        "identifier": "Name of a variable, function, etc.",
        "expression_statement": "Statement containing an expression",
        "binary_expression": "Expression with two operands and an operator",
        "parenthesized_expression": "Expression in parentheses",
        "number_literal": "Numeric constant"
    }
    
    # Print explanations for node types in our graph
    print("\nNode Type Explanations in the Context of Vulnerability Detection:\n")
    node_types_in_graph = set(nx.get_node_attributes(graph, 'type').values())
    
    for node_type in sorted(node_types_in_graph):
        explanation = node_type_explanations.get(node_type, "Custom or specialized syntax element")
        print(f"- {node_type}: {explanation}")
        
        # Add vulnerability relevance for certain node types
        if node_type == "call_expression":
            print("  * VULNERABILITY RELEVANCE: Critical for detecting unsafe function calls like strcpy, gets, etc.")
        elif node_type == "array_declarator":
            print("  * VULNERABILITY RELEVANCE: Important for buffer overflow detection (array size vs usage)")
        elif node_type == "string_literal":
            print("  * VULNERABILITY RELEVANCE: Can indicate hardcoded credentials or format string vulnerabilities")
        elif node_type == "binary_expression":
            print("  * VULNERABILITY RELEVANCE: May reveal integer overflow issues or improper comparisons")
    
    print("\nEdge Meaning:\n")
    print("- Edges represent parent-child relationships in the AST")
    print("- Parent nodes represent containing or higher-level syntax elements")
    print("- Child nodes represent nested or component elements")
    print("- The graph structure captures the syntactic structure of the code")
    print("- GNN models use this structure to understand code context and detect patterns associated with vulnerabilities")

def explain_gnn_process():
    """Explain how the GNN uses the graph representation to detect vulnerabilities"""
    print("\nGNN Vulnerability Detection Process:\n")
    print("1. Code Parsing: Source code is parsed into an Abstract Syntax Tree (AST) using tree-sitter")
    print("2. Graph Construction: The AST is converted to a graph where:")
    print("   - Nodes represent syntax elements (function definitions, declarations, expressions, etc.)")
    print("   - Edges represent the hierarchical structure of the code")
    print("   - Node features capture the type and context of each syntax element")
    print("\n3. Feature Engineering:")
    print("   - Each node gets a feature vector encoding its type and properties")
    print("   - Position, size, and semantic role (expression/declaration/statement) are captured")
    print("   - One-hot encoding of node types adds categorical information")
    print("\n4. Graph Neural Network Processing:")
    print("   - GNN layers aggregate information from neighboring nodes")
    print("   - This captures patterns across the code structure")
    print("   - Multiple GNN layers allow information to flow across the entire graph")
    print("\n5. Global Pooling:")
    print("   - Node features are aggregated to create a single graph-level representation")
    print("   - This represents the entire code snippet's vulnerability profile")
    print("\n6. Classification:")
    print("   - Fully connected layers process the graph representation")
    print("   - Final output is a vulnerability score (0-1)")
    print("   - Scores above a threshold (typically 0.5) indicate potential vulnerabilities")
    print("\nKey Vulnerability Patterns Detected:")
    print("- Unsafe function calls (strcpy, gets, etc.)")
    print("- Buffer size mismatches")
    print("- Missing bounds checks")
    print("- Improper input validation")
    print("- Integer overflow/underflow conditions")
    print("- Format string vulnerabilities")
    print("- Null pointer dereferences")

def main():
    """Main function to demonstrate graph visualization"""
    # Check if tree-sitter language is available
    language_path = None
    for potential_path in [
        os.path.join(os.path.dirname(os.getcwd()), 'build', 'languages.so'),
        os.path.join(os.getcwd(), 'build', 'languages.so'),
        os.path.join(os.getcwd(), 'languages.so'),
        os.path.expanduser('~/tree-sitter-container/build/my-languages.so'),
        os.path.expanduser('~/PycharmProjects/GNN-CodeXGLUE/Code-Code/code-to-code-trans/evaluator/CodeBLEU/parser/my-languages.so'),
        os.path.expanduser('~/defect_detection/lib/python3.12/site-packages/tree_sitter_languages/languages.so')
    ]:
        if os.path.exists(potential_path):
            language_path = potential_path
            print(f"Found language file at: {language_path}")
            break
    
    if not language_path:
        print("Could not find tree-sitter language file. Please specify the path manually.")
        return
    
    try:
        # Load the C language for demonstration
        C_LANGUAGE = Language(language_path, 'c')
        print("Successfully loaded C language")
    except Exception as e:
        print(f"Error loading language: {e}")
        return
    
    # Sample vulnerable C code (buffer overflow)
    sample_code = """
    void copy_data(char *user_input) {
        char buffer[10];
        strcpy(buffer, user_input);  // Vulnerable: no bounds checking
        printf("Buffer contains: %s\\n", buffer);
    }
    """
    
    print("\nSample Code:")
    print(sample_code)
    
    # Build the graph
    try:
        detailed_graph, node_map = build_detailed_graph(sample_code, C_LANGUAGE)
        print(f"\nCreated graph with {len(detailed_graph.nodes)} nodes and {len(detailed_graph.edges)} edges")
    except Exception as e:
        print(f"Error building graph: {e}")
        return
    
    # Visualize the graph
    try:
        output_file = "code_graph_visualization.png"
        node_labels = visualize_ast_graph(detailed_graph, node_map, output_file=output_file)
    except Exception as e:
        print(f"Error visualizing graph: {e}")
    
    # Explain node types
    explain_node_types(detailed_graph)
    
    # Explain GNN process
    explain_gnn_process()
    
    print("\nVisualization complete! Check the output file for the graph image.")

if __name__ == "__main__":
    main()
