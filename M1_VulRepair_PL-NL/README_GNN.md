# GNN-based Code Vulnerability Detection

## Overview
This project implements a Graph Neural Network (GNN) approach for detecting vulnerabilities in source code. The system converts code into Abstract Syntax Tree (AST) graphs and uses Graph Convolutional Networks (GCNs) to learn structural patterns associated with vulnerabilities.

## Key Features
- Converts source code to AST graphs using tree-sitter
- Uses GCN layers to process code structure
- Binary classification: vulnerable (1) vs non-vulnerable (0)
- Trained on the Google Code-X-Glue dataset
- Advantages: structure-aware, context understanding, format invariant

## Installation

### Prerequisites
- Python 3.6+
- PyTorch
- PyTorch Geometric
- Tree-sitter
- Hugging Face Datasets

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd DefectDetection

# Install dependencies
pip install torch torch-geometric networkx numpy tree-sitter tqdm pandas datasets scikit-learn matplotlib seaborn

# Build tree-sitter for C language
git clone https://github.com/tree-sitter/tree-sitter-c
cd tree-sitter-c
mkdir -p build
cc -fPIC -c -I. src/parser.c
cc -fPIC -shared src/parser.o -o build/my-languages.so
```

## Usage

### Training
To train the model from scratch:
```bash
python gnn_main.py --do_train --epochs 30 --train_batch_size 64 --gradient_accumulation_steps 4
```

Key parameters:
- `--do_train`: Enable training mode
- `--epochs`: Number of training epochs
- `--train_batch_size`: Batch size for training
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients (effectively increases batch size)
- `--hidden_channels`: Size of hidden layers (default: 256)
- `--out_channels`: Size of output features (default: 128)
- `--num_layers`: Number of GCN layers (default: 4)
- `--learning_rate`: Learning rate for optimization (default: 1e-3)

### Testing
To evaluate a trained model:
```bash
python gnn_main.py --do_test
```

### Inference
To analyze a specific code snippet:
```bash
python inference.py --input_file /path/to/code.c
# OR
python inference.py --code_snippet "int main() { int x = 5; return x; }"
```

### Analysis
To generate detailed performance metrics and visualizations:
```bash
python analyze_model.py --output_dir ./analysis_results
```

## Model Architecture
The GNN model consists of:
- Multiple GCN layers for message passing
- Batch normalization after each layer
- Global mean pooling to aggregate node features
- Fully connected layers for classification
- Dropout for regularization

## Workflow
1. **Data Preparation**: Code snippets are converted to AST graphs
2. **Feature Engineering**: Node features are created based on node types and positions
3. **Training**: The model is trained using binary cross-entropy loss with class weights
4. **Evaluation**: Performance is measured using accuracy, precision, recall, and F1 score
5. **Inference**: The trained model can analyze new code snippets for vulnerabilities

## Results Interpretation
The evaluation produces several metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of true positives among predicted positives
- **Recall**: Proportion of true positives among actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve
- **PR AUC**: Area under the Precision-Recall curve

## Next Steps
After training the model, consider:
1. Analyzing model performance with detailed metrics
2. Creating a custom inference pipeline for real-world code
3. Experimenting with different model architectures
4. Integrating the model into a code review tool
5. Exploring transfer learning to other programming languages

## Troubleshooting
If you encounter perfect model performance (100% accuracy):
1. Suspect overfitting or data leakage
2. Implement stricter data splitting
3. Add regularization techniques
4. Use cross-validation
5. Simplify the model or augment the dataset

## License
[Your license information]

## Acknowledgments
- Google Code-X-Glue dataset
- PyTorch Geometric library
- Tree-sitter parser
