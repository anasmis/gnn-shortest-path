# GNN Shortest Path Solver

This project implements a Graph Neural Network (GNN) to solve the shortest path problem in graphs. The model uses PyTorch Geometric and can be trained on both CPU and GPU.

## Features

- GNN-based shortest path prediction
- Support for both path and distance prediction
- GPU acceleration with CUDA
- Training with curriculum learning
- Comparison with traditional algorithms (Dijkstra, Bellman-Ford)

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- CUDA (optional, for GPU acceleration)

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/gnn-shortest-path.git
cd gnn-shortest-path

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Local Training

```python
from src.train import train_and_evaluate

# Start training
train_and_evaluate(resume_training=False)
```

### Google Colab Training

To run the training in Google Colab:

1. Create a new Colab notebook
2. Select GPU runtime (Runtime > Change runtime type > GPU)
3. Run the following commands:

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/gnn-shortest-path.git
%cd gnn-shortest-path

# Run the training script
!python colab_train.py
```

The script will automatically:
- Install all required dependencies
- Verify GPU availability
- Start the training process
- Save the trained model

## Project Structure

```
gnn-shortest-path/
├── src/
│   ├── models/
│   │   └── gnn_model.py
│   ├── algorithms/
│   │   └── traditional.py
│   └── train.py
├── requirements.txt
├── README.md
└── colab_train.py
```

## License

MIT License 