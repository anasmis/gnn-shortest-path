"""
Script de déploiement pour le modèle GNN de plus court chemin.
Génère un graphe, trouve le chemin avec le GNN et visualise les résultats.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional
import time

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gnn_model import ShortestPathGNN, create_graph_data
from src.algorithms.traditional import Dijkstra, BellmanFord

class GraphGenerator:
    """Générateur de graphes pour le déploiement."""
    
    @staticmethod
    def generate_random_graph(n_nodes: int = 20, density: float = 0.3, 
                            weight_range: Tuple[float, float] = (1.0, 10.0)) -> Tuple[torch.Tensor, Dict]:
        """Génère un graphe aléatoire connexe."""
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        
        # Créer un graphe aléatoire
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.random.random() < density:
                    weight = np.random.uniform(*weight_range)
                    adj_matrix[i, j] = weight
                    adj_matrix[j, i] = weight
        
        # S'assurer que le graphe est connexe
        for i in range(n_nodes-1):
            if adj_matrix[i, i+1] == 0:
                weight = np.random.uniform(*weight_range)
                adj_matrix[i, i+1] = weight
                adj_matrix[i+1, i] = weight
        
        # Convertir en dictionnaire d'adjacence
        adj_dict = {}
        for i in range(n_nodes):
            adj_dict[i] = []
            for j in range(n_nodes):
                if adj_matrix[i, j] > 0:
                    adj_dict[i].append((j, float(adj_matrix[i, j])))
        
        return adj_matrix, adj_dict
    
    @staticmethod
    def generate_grid_graph(rows: int = 5, cols: int = 5, 
                           weight_range: Tuple[float, float] = (1.0, 5.0)) -> Tuple[torch.Tensor, Dict]:
        """Génère un graphe en grille."""
        n_nodes = rows * cols
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        
        def get_node_id(r, c):
            return r * cols + c
        
        # Créer les connexions de la grille
        for r in range(rows):
            for c in range(cols):
                current = get_node_id(r, c)
                
                # Connexion droite
                if c < cols - 1:
                    right = get_node_id(r, c + 1)
                    weight = np.random.uniform(*weight_range)
                    adj_matrix[current, right] = weight
                    adj_matrix[right, current] = weight
                
                # Connexion bas
                if r < rows - 1:
                    down = get_node_id(r + 1, c)
                    weight = np.random.uniform(*weight_range)
                    adj_matrix[current, down] = weight
                    adj_matrix[down, current] = weight
        
        # Convertir en dictionnaire d'adjacence
        adj_dict = {}
        for i in range(n_nodes):
            adj_dict[i] = []
            for j in range(n_nodes):
                if adj_matrix[i, j] > 0:
                    adj_dict[i].append((j, float(adj_matrix[i, j])))
        
        return adj_matrix, adj_dict

class GNNPathFinder:
    """Classe pour utiliser le modèle GNN entraîné."""
    
    def __init__(self, model_path: str = "best_gnn_shortest_path.pt", device: str = "auto"):
        """Initialise le finder avec le modèle pré-entraîné."""
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Charge le modèle pré-entraîné."""
        if not os.path.exists(self.model_path):
            print(f"❌ Modèle non trouvé: {self.model_path}")
            print("🏃 Entraînement rapide du modèle...")
            self.train_quick_model()
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Créer le modèle avec les bonnes dimensions
            input_dim = 1
            hidden_dim = checkpoint.get('hidden_dim', 128)
            
            self.model = ShortestPathGNN(input_dim=input_dim, hidden_dim=hidden_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Modèle chargé avec succès!")
            print(f"   Accuracy: {checkpoint.get('accuracy', 'N/A'):.2f}%")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {str(e)}")
            print("🏃 Entraînement rapide du modèle...")
            self.train_quick_model()
    
    def train_quick_model(self):
        """Entraîne rapidement un modèle si aucun n'est disponible."""
        try:
            from src.train import train_and_evaluate_fast
            
            def save_callback(model, optimizer, epoch, accuracy, total_loss, avg_error):
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'avg_error': avg_error,
                    'hidden_dim': 128,
                }, self.model_path)
            
            self.model, accuracy = train_and_evaluate_fast(
                resume_training=False,
                save_callback=save_callback,
                device_choice=str(self.device),
                batch_size=32,
                hidden_dim=128,
                num_epochs=50,
                target_accuracy=80
            )
            
            print(f"✅ Modèle entraîné avec une accuracy de {accuracy:.2f}%")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement: {str(e)}")
            raise
    
    def find_path(self, adj_matrix: torch.Tensor, source: int, target: int, 
                  threshold: float = 0.3) -> Tuple[List[int], float, torch.Tensor]:
        """Trouve le plus court chemin avec le GNN."""
        if self.model is None:
            raise ValueError("Modèle non chargé!")
        
        n_nodes = adj_matrix.size(0)
        
        # Créer les caractéristiques des nœuds
        node_features = torch.ones(n_nodes, 1)
        
        # Créer l'encodage de position
        pos_encoding = torch.zeros(n_nodes, 2)
        pos_encoding[source, 0] = 1.0  # Source
        pos_encoding[target, 1] = 1.0   # Target
        
        # Créer l'objet Data
        data = create_graph_data(adj_matrix, node_features, pos_encoding)
        data = data.to(self.device)
        
        # Prédiction
        with torch.no_grad():
            path_probs, pred_distances = self.model(data)
        
        # Extraire le chemin
        path = self.extract_path_from_probs(
            path_probs.cpu(), data.edge_index.cpu(), source, target, threshold
        )
        
        # Distance prédite pour le nœud cible
        predicted_distance = pred_distances[target].cpu().item()
        
        return path, predicted_distance, path_probs.cpu()
    
    def extract_path_from_probs(self, path_probs: torch.Tensor, edge_index: torch.Tensor,
                               source: int, target: int, threshold: float = 0.3) -> List[int]:
        """Extrait le chemin à partir des probabilités."""
        # Obtenir les nœuds candidats
        candidate_nodes = torch.where(path_probs > threshold)[0].tolist()
        
        # S'assurer que source et target sont inclus
        if source not in candidate_nodes:
            candidate_nodes.append(source)
        if target not in candidate_nodes:
            candidate_nodes.append(target)
        
        if len(candidate_nodes) < 2:
            return [source, target]
        
        # Créer un graphe des nœuds candidats
        G = nx.Graph()
        
        # Ajouter les arêtes entre les nœuds candidats
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u in candidate_nodes and v in candidate_nodes:
                # Poids basé sur les probabilités
                weight = 1.0 / (path_probs[u] * path_probs[v] + 1e-6)
                G.add_edge(u, v, weight=weight)
        
        # Trouver le plus court chemin dans le sous-graphe
        try:
            path = nx.shortest_path(G, source, target, weight='weight')
            return path
        except nx.NetworkXNoPath:
            # Fallback: chemin direct si possible
            return [source, target]

class PathVisualizer:
    """Classe pour visualiser les résultats."""
    
    @staticmethod
    def visualize_paths(adj_matrix: torch.Tensor, gnn_path: List[int], 
                       dijkstra_path: List[int], source: int, target: int,
                       path_probs: Optional[torch.Tensor] = None,
                       save_path: str = "path_comparison.png"):
        """Visualise la comparaison des chemins."""
        
        # Créer le graphe NetworkX
        G = nx.from_numpy_array(adj_matrix.numpy())
        
        # Layout pour la visualisation
        pos = nx.spring_layout(G, seed=42)
        
        # Créer la figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Graphique 1: Chemin GNN
        ax1.set_title('GNN Predicted Path', fontsize=14, fontweight='bold')
        
        # Dessiner les arêtes
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3, edge_color='gray')
        
        # Dessiner tous les nœuds
        node_colors = ['lightblue'] * len(G.nodes())
        node_sizes = [300] * len(G.nodes())
        
        # Colorer selon les probabilités si disponibles
        if path_probs is not None:
            for i, prob in enumerate(path_probs):
                intensity = min(prob.item(), 1.0)
                node_colors[i] = plt.cm.Reds(intensity)
                node_sizes[i] = 300 + 200 * intensity  # Taille proportionnelle à la probabilité
        
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, node_size=node_sizes)
        
        # Mettre en évidence source et target
        nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=[source], 
                              node_color='green', node_size=500, label='Source')
        nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=[target], 
                              node_color='red', node_size=500, label='Target')
        
        # Dessiner le chemin GNN
        if len(gnn_path) > 1:
            gnn_edges = list(zip(gnn_path[:-1], gnn_path[1:]))
            nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=gnn_edges, 
                                  edge_color='blue', width=3, alpha=0.8, label='GNN Path')
        
        # Ajouter les labels des nœuds
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8)
        ax1.legend()
        ax1.axis('off')
        
        # Graphique 2: Chemin Dijkstra
        ax2.set_title('Dijkstra Optimal Path', fontsize=14, fontweight='bold')
        
        # Dessiner les arêtes
        nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.3, edge_color='gray')
        
        # Dessiner tous les nœuds
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='lightblue', node_size=300)
        
        # Mettre en évidence source et target
        nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=[source], 
                              node_color='green', node_size=500, label='Source')
        nx.draw_networkx_nodes(G, pos, ax=ax2, nodelist=[target], 
                              node_color='red', node_size=500, label='Target')
        
        # Dessiner le chemin Dijkstra
        if len(dijkstra_path) > 1:
            dijkstra_edges = list(zip(dijkstra_path[:-1], dijkstra_path[1:]))
            nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=dijkstra_edges, 
                                  edge_color='orange', width=3, alpha=0.8, label='Dijkstra Path')
        
        # Ajouter les labels des nœuds
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8)
        ax2.legend()
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualisation sauvegardée: {save_path}")

def deploy_gnn_pathfinder(n_nodes: int = 20, density: float = 0.3, 
                         graph_type: str = "random", model_path: str = "best_gnn_shortest_path.pt"):
    """Fonction principale de déploiement."""
    
    print("🚀 Déploiement du GNN Shortest Path Finder")
    print("=" * 50)
    
    # 1. Générer le graphe
    print(f"📊 Génération d'un graphe {graph_type} ({n_nodes} nœuds, densité={density})")
    
    if graph_type == "random":
        adj_matrix, adj_dict = GraphGenerator.generate_random_graph(n_nodes, density)
    elif graph_type == "grid":
        rows = int(np.sqrt(n_nodes))
        cols = rows
        adj_matrix, adj_dict = GraphGenerator.generate_grid_graph(rows, cols)
    else:
        raise ValueError("Type de graphe non supporté")
    
    # 2. Choisir source et target aléatoirement
    source, target = random.sample(range(n_nodes), 2)
    print(f"🎯 Source: {source}, Target: {target}")
    
    # 3. Initialiser le path finder
    print("🤖 Chargement du modèle GNN...")
    gnn_finder = GNNPathFinder(model_path)
    
    # 4. Trouver le chemin avec le GNN
    print("🔍 Recherche du chemin avec le GNN...")
    start_time = time.time()
    gnn_path, gnn_distance, path_probs = gnn_finder.find_path(adj_matrix, source, target)
    gnn_time = time.time() - start_time
    
    # 5. Trouver le chemin optimal avec Dijkstra
    print("🔍 Recherche du chemin optimal avec Dijkstra...")
    start_time = time.time()
    dijkstra_path, dijkstra_distance = Dijkstra.shortest_path(adj_dict, source, target)
    dijkstra_time = time.time() - start_time
    
    # 6. Afficher les résultats
    print("\n📈 RÉSULTATS")
    print("=" * 30)
    print(f"GNN Path:      {gnn_path}")
    print(f"GNN Distance:  {gnn_distance:.3f}")
    print(f"GNN Time:      {gnn_time*1000:.2f}ms")
    print()
    print(f"Dijkstra Path: {dijkstra_path}")
    print(f"Dijkstra Dist: {dijkstra_distance:.3f}")
    print(f"Dijkstra Time: {dijkstra_time*1000:.2f}ms")
    print()
    
    # Calculer l'erreur
    if dijkstra_distance != float('inf') and dijkstra_distance > 0:
        error = abs(gnn_distance - dijkstra_distance) / dijkstra_distance * 100
        print(f"Erreur relative: {error:.2f}%")
        
        # Vérifier si les chemins sont identiques
        path_match = set(gnn_path) == set(dijkstra_path)
        print(f"Chemins identiques: {'✅' if path_match else '❌'}")
    
    # 7. Visualiser les résultats
    print("\n🎨 Génération de la visualisation...")
    PathVisualizer.visualize_paths(
        adj_matrix, gnn_path, dijkstra_path, source, target, path_probs
    )
    
    return {
        'graph': adj_matrix,
        'adj_dict': adj_dict,
        'source': source,
        'target': target,
        'gnn_path': gnn_path,
        'gnn_distance': gnn_distance,
        'gnn_time': gnn_time,
        'dijkstra_path': dijkstra_path,
        'dijkstra_distance': dijkstra_distance,
        'dijkstra_time': dijkstra_time,
        'path_probs': path_probs
    }

def main():
    """Fonction principale avec arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description='Deploy GNN Shortest Path Finder')
    parser.add_argument('--nodes', type=int, default=20, 
                       help='Number of nodes in the graph')
    parser.add_argument('--density', type=float, default=0.3,
                       help='Edge density of the graph')
    parser.add_argument('--graph-type', choices=['random', 'grid'], default='random',
                       help='Type of graph to generate')
    parser.add_argument('--model-path', type=str, default='best_gnn_shortest_path.pt',
                       help='Path to the trained model')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Définir la seed si spécifiée
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Déployer le système
    results = deploy_gnn_pathfinder(
        n_nodes=args.nodes,
        density=args.density,
        graph_type=args.graph_type,
        model_path=args.model_path
    )
    
    return results

if __name__ == "__main__":
    results = main()