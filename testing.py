"""
Script de visualisation et comparaison entre le modèle GNN et l'algorithme de Dijkstra.
Génère 5 graphes différents, visualise les chemins prédits et évalue les performances.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random
import os
import sys
from typing import List, Tuple, Dict
import seaborn as sns

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gnn_model import ShortestPathGNN, create_graph_data
from src.algorithms.traditional import Dijkstra

class GraphPathVisualizer:
    """Classe pour visualiser et comparer les chemins prédits par GNN et Dijkstra."""
    
    def __init__(self, model_path="best_gnn_shortest_path.pt"):
        """
        Initialise le visualisateur.
        
        Args:
            model_path: Chemin vers le modèle GNN entraîné
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = model_path
        self.results = {
            'gnn_times': [],
            'dijkstra_times': [],
            'gnn_distances': [],
            'dijkstra_distances': [],
            'gnn_paths': [],
            'dijkstra_paths': [],
            'accuracies': [],
            'relative_errors': []
        }
        
    def load_model(self, input_dim=1, hidden_dim=128):
        """Charge le modèle GNN pré-entraîné."""
        try:
            self.model = ShortestPathGNN(input_dim=input_dim, hidden_dim=hidden_dim)
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Modèle chargé depuis {self.model_path}")
                return True
            else:
                print(f"❌ Modèle non trouvé: {self.model_path}")
                return False
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return False
    
    def generate_test_graph(self, n_nodes, density=0.4, seed=None):
        """
        Génère un graphe de test.
        
        Args:
            n_nodes: Nombre de nœuds
            density: Densité du graphe
            seed: Graine pour la reproductibilité
            
        Returns:
            tuple: (adj_matrix, adj_dict, pos_layout)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Créer matrice d'adjacence
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        weight_range = (1.0, 10.0)
        
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.random.random() < density:
                    weight = np.random.uniform(*weight_range)
                    adj_matrix[i, j] = weight
                    adj_matrix[j, i] = weight
        
        # S'assurer de la connexité
        for i in range(n_nodes-1):
            if adj_matrix[i, i+1] == 0:
                weight = np.random.uniform(*weight_range)
                adj_matrix[i, i+1] = weight
                adj_matrix[i+1, i] = weight
        
        # Créer dictionnaire d'adjacence
        adj_dict = {}
        for i in range(n_nodes):
            adj_dict[i] = []
            for j in range(n_nodes):
                if adj_matrix[i, j] > 0:
                    adj_dict[i].append((j, float(adj_matrix[i, j])))
        
        # Créer layout pour visualisation
        G = nx.from_numpy_array(adj_matrix.numpy())
        pos_layout = nx.spring_layout(G, seed=seed)
        
        return adj_matrix, adj_dict, pos_layout
    
    def create_position_encoding(self, n_nodes, src, tgt):
        """Crée un encodage de position pour les nœuds source et cible."""
        pos_encoding = torch.zeros(n_nodes, 2)
        pos_encoding[src, 0] = 1.0
        pos_encoding[tgt, 1] = 1.0
        return pos_encoding
    
    def predict_gnn_path(self, adj_matrix, src, tgt, threshold=0.3):
        """
        Predicts the path with GNN and computes the actual sum of edge weights.
        
        Args:
            adj_matrix: Adjacency matrix (n_nodes x n_nodes)
            src: Source node
            tgt: Target node
            threshold: Probability threshold for path nodes
        
        Returns:
            tuple: (path, actual_path_distance, execution_time)
        """
        if self.model is None:
            return [], float('inf'), 0.0

        start_time = time.time()
        n_nodes = adj_matrix.size(0)
        
        try:
            # Forward pass: Get GNN predictions
            node_features = torch.ones(n_nodes, 1)
            pos_encoding = self.create_position_encoding(n_nodes, src, tgt)
            data = create_graph_data(adj_matrix, node_features, pos_encoding)
            data = data.to(self.device)
            
            with torch.no_grad():
                path_probs, pred_distances = self.model(data)
                
                # Reconstruct path from probabilities
                path_nodes = torch.where(path_probs > threshold)[0].tolist()
                if src not in path_nodes:
                    path_nodes.append(src)
                if tgt not in path_nodes:
                    path_nodes.append(tgt)
                
                path = self.reconstruct_path_from_probs(
                    path_probs, data.edge_index, src, tgt, path_nodes
                )
                
                # Compute ACTUAL distance by summing edge weights along the path
                actual_distance = 0.0
                if len(path) > 1:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        actual_distance += adj_matrix[u, v].item()
                else:
                    actual_distance = float('inf')  # No valid path
                
        except Exception as e:
            print(f"❌ GNN Error: {e}")
            path, actual_distance = [], float('inf')
        
        execution_time = time.time() - start_time
        return path, actual_distance, execution_time
    
    def reconstruct_path_from_probs(self, path_probs, edge_index, src, tgt, candidate_nodes):
        """Reconstruit le chemin à partir des probabilités."""
        try:
            # Créer un graphe avec seulement les nœuds candidats
            G = nx.Graph()
            
            # Ajouter les arêtes entre les nœuds candidats
            for i in range(edge_index.size(1)):
                u, v = edge_index[0, i].item(), edge_index[1, i].item()
                if u in candidate_nodes and v in candidate_nodes:
                    # Poids basé sur les probabilités
                    weight = 1.0 / (path_probs[u].item() * path_probs[v].item() + 1e-6)
                    G.add_edge(u, v, weight=weight)
            
            # Trouver le plus court chemin dans ce sous-graphe
            if G.has_node(src) and G.has_node(tgt):
                try:
                    path = nx.shortest_path(G, src, tgt, weight='weight')
                    return path
                except nx.NetworkXNoPath:
                    return [src, tgt] if src != tgt else [src]
            else:
                return [src, tgt] if src != tgt else [src]
                
        except Exception:
            return [src, tgt] if src != tgt else [src]
    
    def compare_paths(self, graph_id, adj_matrix, adj_dict, src, tgt, pos_layout):
        """
        Compare les chemins GNN et Dijkstra pour un graphe donné.
        
        Args:
            graph_id: ID du graphe
            adj_matrix: Matrice d'adjacence
            adj_dict: Dictionnaire d'adjacence
            src: Nœud source
            tgt: Nœud cible
            pos_layout: Layout pour la visualisation
            
        Returns:
            dict: Résultats de la comparaison
        """
        print(f"🔍 Analyse Graphe {graph_id}: {src} → {tgt}")
        
        # Prédiction GNN
        gnn_path, gnn_dist, gnn_time = self.predict_gnn_path(adj_matrix, src, tgt)
        
        # Prédiction Dijkstra
        start_time = time.time()
        dijkstra_path, dijkstra_dist = Dijkstra.shortest_path(adj_dict, src, tgt)
        dijkstra_time = time.time() - start_time
        
        # Calculer la précision
        accuracy = 0
        if dijkstra_dist != float('inf') and gnn_dist != float('inf'):
            relative_error = abs(gnn_dist - dijkstra_dist) / dijkstra_dist
            accuracy = 1 - min(relative_error, 1.0)  # Précision basée sur l'erreur relative
        
        # Stocker les résultats
        results = {
            'graph_id': graph_id,
            'src': src,
            'tgt': tgt,
            'gnn_path': gnn_path,
            'dijkstra_path': dijkstra_path,
            'gnn_distance': gnn_dist,
            'dijkstra_distance': dijkstra_dist,
            'gnn_time': gnn_time,
            'dijkstra_time': dijkstra_time,
            'accuracy': accuracy,
            'relative_error': abs(gnn_dist - dijkstra_dist) / max(dijkstra_dist, 1e-6)
        }
        
        # Mettre à jour les résultats globaux
        self.results['gnn_times'].append(gnn_time)
        self.results['dijkstra_times'].append(dijkstra_time)
        self.results['gnn_distances'].append(gnn_dist)
        self.results['dijkstra_distances'].append(dijkstra_dist)
        self.results['gnn_paths'].append(gnn_path)
        self.results['dijkstra_paths'].append(dijkstra_path)
        self.results['accuracies'].append(accuracy)
        self.results['relative_errors'].append(results['relative_error'])
        
        print(f"  GNN: {gnn_path} (dist={gnn_dist:.2f}, time={gnn_time*1000:.2f}ms)")
        print(f"  Dijkstra: {dijkstra_path} (dist={dijkstra_dist:.2f}, time={dijkstra_time*1000:.2f}ms)")
        print(f"  Précision: {accuracy*100:.1f}%")
        
        return results
    
    def visualize_graph_comparison(self, results, adj_matrix, pos_layout):
        """Visualise la comparaison pour un graphe."""
        plt.figure(figsize=(15, 5))
        
        # Créer le graphe NetworkX
        G = nx.from_numpy_array(adj_matrix.numpy())
        
        # Graphe original
        plt.subplot(1, 3, 1)
        nx.draw_networkx_edges(G, pos_layout, alpha=0.3)
        nx.draw_networkx_nodes(G, pos_layout, node_color='lightgray', node_size=300)
        nx.draw_networkx_nodes(G, pos_layout, nodelist=[results['src']], 
                              node_color='green', node_size=500, label='Source')
        nx.draw_networkx_nodes(G, pos_layout, nodelist=[results['tgt']], 
                              node_color='red', node_size=500, label='Cible')
        nx.draw_networkx_labels(G, pos_layout, font_size=8)
        plt.title(f"Graphe {results['graph_id']}\n{results['src']} → {results['tgt']}")
        plt.axis('off')
        
        # Chemin Dijkstra
        plt.subplot(1, 3, 2)
        nx.draw_networkx_edges(G, pos_layout, alpha=0.2)
        nx.draw_networkx_nodes(G, pos_layout, node_color='lightgray', node_size=300)
        
        if len(results['dijkstra_path']) > 1:
            dijkstra_edges = list(zip(results['dijkstra_path'][:-1], results['dijkstra_path'][1:]))
            nx.draw_networkx_edges(G, pos_layout, edgelist=dijkstra_edges, 
                                 edge_color='blue', width=3)
        
        nx.draw_networkx_nodes(G, pos_layout, nodelist=[results['src']], 
                              node_color='green', node_size=500)
        nx.draw_networkx_nodes(G, pos_layout, nodelist=[results['tgt']], 
                              node_color='red', node_size=500)
        nx.draw_networkx_labels(G, pos_layout, font_size=8)
        plt.title(f"Dijkstra\nDist: {results['dijkstra_distance']:.2f}\nTemps: {results['dijkstra_time']*1000:.2f}ms")
        plt.axis('off')
        
        # Chemin GNN
        plt.subplot(1, 3, 3)
        nx.draw_networkx_edges(G, pos_layout, alpha=0.2)
        nx.draw_networkx_nodes(G, pos_layout, node_color='lightgray', node_size=300)
        
        if len(results['gnn_path']) > 1:
            gnn_edges = list(zip(results['gnn_path'][:-1], results['gnn_path'][1:]))
            nx.draw_networkx_edges(G, pos_layout, edgelist=gnn_edges, 
                                 edge_color='orange', width=3, style='dashed')
        
        nx.draw_networkx_nodes(G, pos_layout, nodelist=[results['src']], 
                              node_color='green', node_size=500)
        nx.draw_networkx_nodes(G, pos_layout, nodelist=[results['tgt']], 
                              node_color='red', node_size=500)
        nx.draw_networkx_labels(G, pos_layout, font_size=8)
        plt.title(f"GNN\nDist: {results['gnn_distance']:.2f}\nTemps: {results['gnn_time']*1000:.2f}ms\nPrécision: {results['accuracy']*100:.1f}%")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"graph_comparison_{results['graph_id']}.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_performance_summary(self):
        """Trace un résumé des performances."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Temps d'exécution
        axes[0, 0].bar(['GNN', 'Dijkstra'], 
                      [np.mean(self.results['gnn_times'])*1000, 
                       np.mean(self.results['dijkstra_times'])*1000],
                      color=['orange', 'blue'], alpha=0.7)
        axes[0, 0].set_title('Temps d\'exécution moyen')
        axes[0, 0].set_ylabel('Temps (ms)')
        
        # Précision
        axes[0, 1].hist(self.results['accuracies'], bins=10, alpha=0.7, color='green')
        axes[0, 1].axvline(np.mean(self.results['accuracies']), color='red', linestyle='--', 
                          label=f'Moyenne: {np.mean(self.results["accuracies"])*100:.1f}%')
        axes[0, 1].set_title('Distribution de la précision')
        axes[0, 1].set_xlabel('Précision')
        axes[0, 1].set_ylabel('Fréquence')
        axes[0, 1].legend()
        
        # Erreur relative
        axes[1, 0].hist(self.results['relative_errors'], bins=10, alpha=0.7, color='red')
        axes[1, 0].axvline(np.mean(self.results['relative_errors']), color='blue', linestyle='--',
                          label=f'Moyenne: {np.mean(self.results["relative_errors"])*100:.1f}%')
        axes[1, 0].set_title('Distribution de l\'erreur relative')
        axes[1, 0].set_xlabel('Erreur relative')
        axes[1, 0].set_ylabel('Fréquence')
        axes[1, 0].legend()
        
        # Comparaison des distances
        axes[1, 1].scatter(self.results['dijkstra_distances'], self.results['gnn_distances'], 
                          alpha=0.7, color='purple')
        min_dist = min(min(self.results['dijkstra_distances']), min(self.results['gnn_distances']))
        max_dist = max(max(self.results['dijkstra_distances']), max(self.results['gnn_distances']))
        axes[1, 1].plot([min_dist, max_dist], [min_dist, max_dist], 'r--', alpha=0.8)
        axes[1, 1].set_title('Distances: GNN vs Dijkstra')
        axes[1, 1].set_xlabel('Distance Dijkstra')
        axes[1, 1].set_ylabel('Distance GNN')
        
        plt.tight_layout()
        plt.savefig("performance_summary.png", dpi=150, bbox_inches='tight')
        plt.show()
    
    def print_statistics(self):
        """Affiche les statistiques finales."""
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DES PERFORMANCES")
        print("="*60)
        
        print(f"🔥 Temps d'exécution moyen:")
        print(f"   GNN:      {np.mean(self.results['gnn_times'])*1000:.2f} ± {np.std(self.results['gnn_times'])*1000:.2f} ms")
        print(f"   Dijkstra: {np.mean(self.results['dijkstra_times'])*1000:.2f} ± {np.std(self.results['dijkstra_times'])*1000:.2f} ms")
        print(f"   Ratio:    {np.mean(self.results['gnn_times'])/np.mean(self.results['dijkstra_times']):.2f}x")
        
        print(f"\n🎯 Précision:")
        print(f"   Moyenne:  {np.mean(self.results['accuracies'])*100:.1f}%")
        print(f"   Médiane:  {np.median(self.results['accuracies'])*100:.1f}%")
        print(f"   Min/Max:  {np.min(self.results['accuracies'])*100:.1f}% / {np.max(self.results['accuracies'])*100:.1f}%")
        
        print(f"\n📏 Erreur relative:")
        print(f"   Moyenne:  {np.mean(self.results['relative_errors'])*100:.1f}%")
        print(f"   Médiane:  {np.median(self.results['relative_errors'])*100:.1f}%")
        print(f"   Max:      {np.max(self.results['relative_errors'])*100:.1f}%")
        
        print("="*60)

def main():
    """Fonction principale pour exécuter la comparaison."""
    print("🚀 Démarrage de la comparaison GNN vs Dijkstra")
    
    # Initialiser le visualisateur
    visualizer = GraphPathVisualizer()
    
    # Charger le modèle
    if not visualizer.load_model():
        print("❌ Impossible de charger le modèle. Assurez-vous d'avoir entraîné le modèle d'abord.")
        return
    
    # Configuration des graphes de test
    test_configs = [
        {'n_nodes': random.randint(30, 50), 'density': 0.01*random.randint(3, 4), 'seed': random.randint(1, 100000)} for _ in range(5)
    ]
    
    print(f"📈 Test sur {len(test_configs)} graphes différents\n")
    
    # Analyser chaque graphe
    for i, config in enumerate(test_configs, 1):
        print(f"🔬 === GRAPHE {i} ===")
        
        # Générer le graphe
        adj_matrix, adj_dict, pos_layout = visualizer.generate_test_graph(**config)
        n_nodes = config['n_nodes']
        
        # Choisir source et cible aléatoirement
        src, tgt = random.sample(range(n_nodes), 2)
        
        # Comparer les algorithmes
        results = visualizer.compare_paths(i, adj_matrix, adj_dict, src, tgt, pos_layout)
        
        # Visualiser la comparaison
        visualizer.visualize_graph_comparison(results, adj_matrix, pos_layout)
        
        print()
    
    # Afficher le résumé des performances
    visualizer.plot_performance_summary()
    visualizer.print_statistics()
    
    print("\n✅ Comparaison terminée!")
    print("📁 Les visualisations ont été sauvegardées dans le répertoire courant.")

if __name__ == "__main__":
    main()