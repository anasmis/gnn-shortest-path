# GNN Shortest Path Solver: Mathematical Documentation
Author: Anas AIT ALI

## 1. Problem Formulation

### 1.1 Shortest Path Problem
Given a directed graph G = (V, E) with:
- V: set of vertices (nodes)
- E: set of edges
- w: E → ℝ⁺: weight function assigning non-negative weights to edges

The shortest path problem aims to find a path P = (v₁, v₂, ..., vₖ) between source node s and target node t that minimizes:
```
min ∑ w(e) for e ∈ P
```

### 1.2 Graph Neural Network Approach
Our GNN model learns to approximate the shortest path function:
```
f: (G, s, t) → (P, d)
```
where:
- P: predicted path
- d: predicted distance

## 2. Model Architecture

### 2.1 Graph Representation
For a graph G with n nodes, we represent:
- Node features: X ∈ ℝ^(n×d_in)
- Edge features: E ∈ ℝ^(m×d_edge)
- Adjacency matrix: A ∈ {0,1}^(n×n)

### 2.2 GNN Layers
The model uses multiple Graph Convolutional Layers (GCN) with the following propagation rule:

For layer l:
```
H^(l+1) = σ(D^(-1/2)ÃD^(-1/2)H^(l)W^(l))
```
where:
- Ã = A + I (adjacency matrix with self-loops)
- D: degree matrix
- H^(l): node features at layer l
- W^(l): learnable weight matrix
- σ: activation function (ReLU)

### 2.3 Position Encoding
We enhance node features with position encoding:
```
X_enhanced = [X || P]
```
where:
- P ∈ ℝ^(n×2): position encoding matrix
- ||: concatenation operator

### 2.4 Skip Connections
To mitigate vanishing gradients, we implement skip connections:
```
H^(l+1) = H^(l+1) + W_skip * H^(l)
```
where W_skip is a learnable transformation matrix.

## 3. Training Process

### 3.1 Loss Functions
The model optimizes two objectives:

1. Distance Prediction Loss (Huber Loss):
```
L_dist = 1/n ∑ huber(d_pred - d_true)
```

2. Path Prediction Loss (Binary Cross-Entropy):
```
L_path = -1/n ∑ [y_true * log(y_pred) + (1-y_true) * log(1-y_pred)]
```

Total Loss:
```
L_total = L_dist + λ * L_path
```
where λ is a balancing parameter.

### 3.2 Curriculum Learning
We implement curriculum learning with three difficulty levels:

1. Easy: Dense graphs with uniform weights
   - Edge density: 0.3-0.8
   - Weight range: [0.5, 5.0]

2. Medium: Balanced graphs
   - Edge density: 0.3
   - Weight range: [0.1, 10.0]

3. Hard: Sparse graphs with varied weights
   - Edge density: 0.1-0.3
   - Weight range: [0.1, 20.0]

## 4. Model Components

### 4.1 GCN Layers
The model uses 4 GCN layers with:
- Hidden dimension: 512
- Batch normalization
- Dropout rate: 0.2
- ReLU activation

### 4.2 Prediction Heads
1. Path Prediction:
```
P_path = sigmoid(MLP(H^(L)))
```
where:
- H^(L): final layer output
- MLP: 3-layer neural network

2. Distance Prediction:
```
P_dist = MLP(H^(L))
```

## 5. Performance Metrics

### 5.1 Distance Accuracy
A prediction is considered correct if:
```
|d_pred - d_true|/d_true < 0.1
```

### 5.2 Path Accuracy
Path prediction is correct if:
```
P_pred ∩ P_true = P_true
```
where P_pred and P_true are the predicted and true paths.

## 6. Implementation Details

### 6.1 Data Generation
For each graph:
- Number of nodes: n ∈ [20, 100]
- Edge density: p ∈ [0.1, 0.8]
- Weight distribution: w ~ U[a, b]

### 6.2 Training Configuration
- Batch size: 256
- Learning rate: 0.001
- Optimizer: AdamW
- Weight decay: 0.01
- Scheduler: CosineAnnealingWarmRestarts

### 6.3 GPU Utilization
- CUDA memory fraction: 0.8
- Batch normalization
- Gradient accumulation

## 7. Comparison with Traditional Algorithms

### 7.1 Dijkstra's Algorithm
Time complexity: O(|E| + |V|log|V|)
Space complexity: O(|V|)

### 7.2 Bellman-Ford Algorithm
Time complexity: O(|V||E|)
Space complexity: O(|V|)

### 7.3 GNN Model
- Training time: O(batch_size * num_epochs * |V|)
- Inference time: O(|V|)
- Memory complexity: O(|V| * hidden_dim)

## 8. Future Improvements

1. Attention Mechanisms:
```
Attention(Q, K, V) = softmax(QK^T/√d_k)V
```

2. Graph Pooling:
```
H_pooled = max_pool(H)
```

3. Multi-task Learning:
```
L_total = ∑ λ_i * L_i
```

## 9. Performance Visualization

### 9.1 Training Metrics
We visualize the following metrics during training:
1. Loss Curves:
   - Total loss: L_total
   - Distance loss: L_dist
   - Path loss: L_path

2. Accuracy Metrics:
   - Distance accuracy over epochs
   - Path accuracy over epochs
   - Relative error distribution

### 9.2 Graph Visualization
For each prediction, we visualize:
1. Input Graph:
   - Node positions using force-directed layout
   - Edge weights represented by thickness
   - Source and target nodes highlighted

2. Predicted Path:
   - True shortest path in green
   - Predicted path in blue
   - Incorrect predictions in red

3. Attention Weights:
   - Node attention scores
   - Edge attention scores
   - Path probability heatmap

### 9.3 Performance Analysis
We generate:
1. Confusion Matrices:
   - Path prediction accuracy
   - Distance prediction accuracy

2. Error Analysis:
   - Error distribution by graph size
   - Error distribution by edge density
   - Error distribution by weight range

3. Comparative Analysis:
   - GNN vs Dijkstra performance
   - GNN vs Bellman-Ford performance
   - Training time comparison

### 9.4 Visualization Tools
We use:
- NetworkX for graph visualization
- Matplotlib for metric plots
- Seaborn for statistical visualizations
- Plotly for interactive plots

## References

1. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
2. Veličković, P., et al. (2017). Graph attention networks.
3. Hamilton, W. L., et al. (2017). Inductive representation learning on large graphs.

## License

MIT License

Copyright (c) 2024 Anas AIT ALI

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 