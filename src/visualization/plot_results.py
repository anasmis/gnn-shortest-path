"""
Script de visualisation des résultats de comparaison entre le modèle GNN et les algorithmes traditionnels.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_performance_comparison(gnn_times, dijkstra_times, bf_times, gnn_distances, dijkstra_distances, bf_distances):
    """
    Trace les temps d'exécution et les distances prédites pour chaque méthode.
    
    Args:
        gnn_times (list): Liste des temps d'exécution du GNN
        dijkstra_times (list): Liste des temps d'exécution de Dijkstra
        bf_times (list): Liste des temps d'exécution de Bellman-Ford
        gnn_distances (list): Liste des distances prédites par le GNN
        dijkstra_distances (list): Liste des distances calculées par Dijkstra
        bf_distances (list): Liste des distances calculées par Bellman-Ford
    """
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Tracer les temps d'exécution
    ax1.bar(['GNN', 'Dijkstra', 'Bellman-Ford'], [np.mean(gnn_times), np.mean(dijkstra_times), np.mean(bf_times)])
    ax1.set_title('Temps d\'exécution moyen')
    ax1.set_ylabel('Temps (s)')
    
    # Tracer les distances prédites
    ax2.bar(['GNN', 'Dijkstra', 'Bellman-Ford'], [np.mean(gnn_distances), np.mean(dijkstra_distances), np.mean(bf_distances)])
    ax2.set_title('Distances prédites moyennes')
    ax2.set_ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Exemple de données (à remplacer par les résultats réels)
    gnn_times = [0.1, 0.2, 0.15]
    dijkstra_times = [0.05, 0.06, 0.07]
    bf_times = [0.08, 0.09, 0.1]
    gnn_distances = [10, 11, 12]
    dijkstra_distances = [10, 10, 10]
    bf_distances = [10, 10, 10]
    
    plot_performance_comparison(gnn_times, dijkstra_times, bf_times, gnn_distances, dijkstra_distances, bf_distances) 