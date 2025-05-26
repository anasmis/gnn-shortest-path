"""
Visualization tools for the GNN Shortest Path model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Tuple, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PerformanceVisualizer:
    def __init__(self, save_dir: str = "visualizations"):
        """Initialize the visualizer."""
        self.save_dir = save_dir
        self.metrics_history = {
            'total_loss': [],
            'dist_loss': [],
            'path_loss': [],
            'dist_accuracy': [],
            'path_accuracy': [],
            'relative_error': []
        }
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics history."""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def plot_training_curves(self):
        """Plot training metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.metrics_history['total_loss'], label='Total Loss')
        axes[0, 0].plot(self.metrics_history['dist_loss'], label='Distance Loss')
        axes[0, 0].plot(self.metrics_history['path_loss'], label='Path Loss')
        axes[0, 0].set_title('Training Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy curves
        axes[0, 1].plot(self.metrics_history['dist_accuracy'], label='Distance Accuracy')
        axes[0, 1].plot(self.metrics_history['path_accuracy'], label='Path Accuracy')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        
        # Error distribution
        sns.histplot(self.metrics_history['relative_error'], ax=axes[1, 0])
        axes[1, 0].set_title('Relative Error Distribution')
        axes[1, 0].set_xlabel('Relative Error')
        axes[1, 0].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_curves.png")
        plt.close()
    
    def visualize_graph(self, 
                       adj_matrix: torch.Tensor,
                       true_path: List[int],
                       pred_path: List[int],
                       source: int,
                       target: int):
        """Visualize the graph with true and predicted paths."""
        G = nx.from_numpy_array(adj_matrix.numpy())
        pos = nx.spring_layout(G)
        
        plt.figure(figsize=(12, 8))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=500)
        
        # Highlight source and target
        nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color='green', node_size=700)
        nx.draw_networkx_nodes(G, pos, nodelist=[target], node_color='red', node_size=700)
        
        # Draw true path
        true_edges = list(zip(true_path[:-1], true_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=true_edges, edge_color='green', width=2)
        
        # Draw predicted path
        pred_edges = list(zip(pred_path[:-1], pred_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=pred_edges, edge_color='blue', width=2, style='dashed')
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title('Graph with True (Green) and Predicted (Blue) Paths')
        plt.axis('off')
        plt.savefig(f"{self.save_dir}/graph_visualization.png")
        plt.close()
    
    def plot_error_analysis(self, 
                           errors: List[float],
                           graph_sizes: List[int],
                           densities: List[float]):
        """Plot error analysis by different graph properties."""
        fig = make_subplots(rows=1, cols=3,
                           subplot_titles=('By Graph Size', 'By Edge Density', 'Error Distribution'))
        
        # Error by graph size
        fig.add_trace(
            go.Scatter(x=graph_sizes, y=errors, mode='markers'),
            row=1, col=1
        )
        
        # Error by density
        fig.add_trace(
            go.Scatter(x=densities, y=errors, mode='markers'),
            row=1, col=2
        )
        
        # Error distribution
        fig.add_trace(
            go.Histogram(x=errors),
            row=1, col=3
        )
        
        fig.update_layout(height=500, showlegend=False)
        fig.write_html(f"{self.save_dir}/error_analysis.html")
    
    def plot_comparison(self,
                       gnn_times: List[float],
                       dijkstra_times: List[float],
                       bf_times: List[float]):
        """Plot performance comparison with traditional algorithms."""
        plt.figure(figsize=(10, 6))
        
        data = [gnn_times, dijkstra_times, bf_times]
        labels = ['GNN', 'Dijkstra', 'Bellman-Ford']
        
        plt.boxplot(data, labels=labels)
        plt.title('Execution Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')
        
        plt.savefig(f"{self.save_dir}/algorithm_comparison.png")
        plt.close()
    
    def plot_attention_weights(self,
                             attention_weights: torch.Tensor,
                             node_labels: List[str]):
        """Plot attention weights as a heatmap."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(attention_weights.numpy(),
                   xticklabels=node_labels,
                   yticklabels=node_labels,
                   cmap='viridis')
        plt.title('Attention Weights Heatmap')
        plt.savefig(f"{self.save_dir}/attention_weights.png")
        plt.close()

def visualize_batch_results(model_outputs: List[Tuple],
                          true_paths: List[List[int]],
                          true_distances: List[float],
                          graph_sizes: List[int],
                          densities: List[float]):
    """Visualize results for a batch of predictions."""
    visualizer = PerformanceVisualizer()
    
    # Calculate metrics
    metrics = {
        'total_loss': [],
        'dist_loss': [],
        'path_loss': [],
        'dist_accuracy': [],
        'path_accuracy': [],
        'relative_error': []
    }
    
    for (pred_path, pred_dist), true_path, true_dist in zip(model_outputs, true_paths, true_distances):
        # Update metrics
        metrics['relative_error'].append(abs(pred_dist - true_dist) / true_dist)
        metrics['path_accuracy'].append(1.0 if set(pred_path) == set(true_path) else 0.0)
    
    # Plot visualizations
    visualizer.plot_training_curves()
    visualizer.plot_error_analysis(
        metrics['relative_error'],
        graph_sizes,
        densities
    )
    
    return metrics 