"""
Exemple d'utilisation du système de déploiement GNN Shortest Path.
"""

import sys
import os

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deploy_gnn import deploy_gnn_pathfinder, GNNPathFinder, GraphGenerator, PathVisualizer
import torch
import random
import numpy as np

def example_basic_usage():
    """Exemple d'utilisation basique."""
    print("🔥 EXEMPLE 1: Utilisation basique")
    print("=" * 40)
    
    # Déploiement simple avec paramètres par défaut
    results = deploy_gnn_pathfinder(
        n_nodes=15,
        density=0.4,
        graph_type="random"
    )
    
    print(f"✅ Déploiement terminé!")
    print(f"   - Chemin GNN: {len(results['gnn_path'])} nœuds")
    print(f"   - Temps GNN: {results['gnn_time']*1000:.2f}ms")
    
    return results

def example_custom_graph():
    """Exemple avec un graphe personnalisé."""
    print("\n🔥 EXEMPLE 2: Graphe personnalisé")
    print("=" * 40)
    
    # Créer un graphe en grille
    results = deploy_gnn_pathfinder(
        n_nodes=25,  # 5x5 grille
        density=0.3,
        graph_type="grid"
    )
    
    return results

def example_manual_usage():
    """Exemple d'utilisation manuelle étape par étape."""
    print("\n🔥 EXEMPLE 3: Utilisation manuelle")
    print("=" * 40)
    
    # 1. Générer un graphe
    print("📊 Génération du graphe...")
    adj_matrix, adj_dict = GraphGenerator.generate_random_graph(
        n_nodes=12, density=0.5
    )
    
    # 2. Choisir source et target
    source, target = 0, 11
    print(f"🎯 Source: {source}, Target: {target}")
    
    # 3. Initialiser le finder
    print("🤖 Initialisation du GNN...")
    finder = GNNPathFinder()
    
    # 4. Trouver le chemin
    print("🔍 Recherche du chemin...")
    gnn_path, gnn_distance, path_probs = finder.find_path(
        adj_matrix, source, target, threshold=0.2
    )
    
    # 5. Comparer avec Dijkstra
    from src.algorithms.traditional import Dijkstra
    dijkstra_path, dijkstra_distance = Dijkstra.shortest_path(
        adj_dict, source, target
    )
    
    # 6. Afficher les résultats
    print(f"GNN: {gnn_path} (distance: {gnn_distance:.3f})")
    print(f"Dijkstra: {dijkstra_path} (distance: {dijkstra_distance:.3f})")
    
    # 7. Visualiser
    PathVisualizer.visualize_paths(
        adj_matrix, gnn_path, dijkstra_path, source, target, path_probs,
        save_path="manual_example.png"
    )
    
    return {
        'gnn_path': gnn_path,
        'dijkstra_path': dijkstra_path,
        'gnn_distance': gnn_distance,
        'dijkstra_distance': dijkstra_distance
    }

def example_batch_testing():
    """Exemple de test sur plusieurs graphes."""
    print("\n🔥 EXEMPLE 4: Test batch")
    print("=" * 40)
    
    finder = GNNPathFinder()
    results = []
    
    for i in range(5):
        print(f"Test {i+1}/5...")
        
        # Générer un graphe aléatoire
        n_nodes = random.randint(10, 20)
        density = random.uniform(0.2, 0.5)
        
        adj_matrix, adj_dict = GraphGenerator.generate_random_graph(
            n_nodes, density
        )
        
        # Source et target aléatoires
        source, target = random.sample(range(n_nodes), 2)
        
        # Prédiction GNN
        gnn_path, gnn_distance, _ = finder.find_path(adj_matrix, source, target)
        
        # Vérité terrain (Dijkstra)
        from src.algorithms.traditional import Dijkstra
        dijkstra_path, dijkstra_distance = Dijkstra.shortest_path(
            adj_dict, source, target
        )
        
        # Calculer l'erreur
        if dijkstra_distance != float('inf') and dijkstra_distance > 0:
            error = abs(gnn_distance - dijkstra_distance) / dijkstra_distance * 100
        else:
            error = float('inf')
        
        results.append({
            'graph_size': n_nodes,
            'density': density,
            'error': error,
            'gnn_distance': gnn_distance,
            'dijkstra_distance': dijkstra_distance
        })
        
        print(f"   Nœuds: {n_nodes}, Erreur: {error:.2f}%")
    
    # Statistiques
    valid_errors = [r['error'] for r in results if r['error'] != float('inf')]
    if valid_errors:
        avg_error = np.mean(valid_errors)
        max_error = np.max(valid_errors)
        print(f"\n📊 Statistiques:")
        print(f"   Erreur moyenne: {avg_error:.2f}%")
        print(f"   Erreur maximale: {max_error:.2f}%")
    
    return results

def example_different_thresholds():
    """Exemple avec différents seuils de probabilité."""
    print("\n🔥 EXEMPLE 5: Différents seuils")
    print("=" * 40)
    
    # Générer un graphe fixe
    torch.manual_seed(42)
    np.random.seed(42)
    
    adj_matrix, adj_dict = GraphGenerator.generate_random_graph(15, 0.4)
    source, target = 0, 14
    
    finder = GNNPathFinder()
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    print(f"Source: {source}, Target: {target}")
    
    # Référence Dijkstra
    from src.algorithms.traditional import Dijkstra
    dijkstra_path, dijkstra_distance = Dijkstra.shortest_path(adj_dict, source, target)
    print(f"Dijkstra: {dijkstra_path} (distance: {dijkstra_distance:.3f})")
    
    print("\nRésultats GNN avec différents seuils:")
    for threshold in thresholds:
        gnn_path, gnn_distance, _ = finder.find_path(
            adj_matrix, source, target, threshold=threshold
        )
        error = abs(gnn_distance - dijkstra_distance) / dijkstra_distance * 100
        print(f"  Seuil {threshold}: {gnn_path} (distance: {gnn_distance:.3f}, erreur: {error:.2f}%)")

if __name__ == "__main__":
    print("🚀 EXEMPLES D'UTILISATION DU GNN SHORTEST PATH")
    print("=" * 60)
    
    # Définir une seed pour la reproductibilité
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Exemple 1: Utilisation basique
        example_basic_usage()
        
        # Exemple 2: Graphe personnalisé
        example_custom_graph()
        
        # Exemple 3: Utilisation manuelle
        example_manual_usage()
        
        # Exemple 4: Test batch
        example_batch_testing()
        
        # Exemple 5: Différents seuils
        example_different_thresholds()
        
        print("\n✅ Tous les exemples terminés avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()