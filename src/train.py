"""
Script d'entraînement et d'évaluation du modèle GNN avec comparaison des performances.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm
import sys
import os
import random
import argparse
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gnn_model import ShortestPathGNN, create_graph_data
from src.algorithms.traditional import Dijkstra, BellmanFord

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GNN for shortest path prediction')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cuda', 'cpu'],
                      help='Device to use for training (auto: use CUDA if available)')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from last checkpoint')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='Batch size for training')
    parser.add_argument('--hidden-dim', type=int, default=512,
                      help='Hidden dimension of the GNN')
    parser.add_argument('--num-epochs', type=int, default=1000,
                      help='Number of training epochs')
    parser.add_argument('--target-accuracy', type=float, default=0.85,
                      help='Target accuracy to stop training')
    return parser.parse_args()

def generate_random_graph(n_nodes, density=0.3, difficulty='easy'):
    """
    Génère un graphe aléatoire avec différents niveaux de difficulté.
    
    Args:
        n_nodes (int): Nombre de nœuds
        density (float): Densité du graphe (0-1)
        difficulty (str): Niveau de difficulté ('easy', 'medium', 'hard')
    """
    adj_matrix = torch.zeros(n_nodes, n_nodes)
    
    if difficulty == 'easy':
        # Graphes plus denses avec poids plus uniformes
        density = min(density * 1.5, 0.8)
        weight_range = (0.5, 5.0)
    elif difficulty == 'medium':
        # Densité moyenne avec poids variés
        weight_range = (0.1, 10.0)
    else:  # hard
        # Graphes plus clairsemés avec poids très variés
        density = max(density * 0.7, 0.1)
        weight_range = (0.1, 20.0)
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if np.random.random() < density:
                weight = np.random.uniform(*weight_range)
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight
    
    adj_dict = {}
    for i in range(n_nodes):
        adj_dict[i] = []
        for j in range(n_nodes):
            if adj_matrix[i, j] > 0:
                adj_dict[i].append((j, float(adj_matrix[i, j])))
    
    return adj_matrix, adj_dict

def create_position_encoding(n_nodes, src, tgt):
    """Crée un encodage de position pour les nœuds source et cible."""
    pos_encoding = torch.zeros(n_nodes, 2)
    # Encodage one-hot pour source et cible
    pos_encoding[src, 0] = 1.0
    pos_encoding[tgt, 1] = 1.0
    return pos_encoding

def create_dataset(num_graphs=100, n_nodes=20, density=0.3, num_pairs=10, difficulty='easy'):
    """Crée un dataset avec différents niveaux de difficulté."""
    dataset = []
    for _ in range(num_graphs):
        adj_matrix, adj_dict = generate_random_graph(n_nodes, density, difficulty)
        for _ in range(num_pairs):
            src, tgt = random.sample(range(n_nodes), 2)
            try:
                _, dist = Dijkstra.shortest_path(adj_dict, src, tgt)
                if dist < float('inf'):
                    # Créer les caractéristiques des nœuds
                    node_features = torch.ones(n_nodes, 1)
                    # Ajouter l'encodage de position
                    pos_encoding = create_position_encoding(n_nodes, src, tgt)
                    data = create_graph_data(adj_matrix, node_features, pos_encoding)
                    dataset.append((data, src, tgt, dist))
            except Exception:
                continue
    return dataset

def collate_batch(batch):
    datas, srcs, tgts, dists = zip(*batch)
    return list(datas), list(srcs), list(tgts), torch.tensor(dists, dtype=torch.float32)

def train_and_evaluate(resume_training=False, save_callback=None, device_choice='auto', batch_size=256, hidden_dim=512, num_epochs=1000, target_accuracy=0.85):
    # Configuration
    input_dim = 1  # Dimension des caractéristiques d'entrée
    hidden_dim = hidden_dim
    learning_rate = 0.0005  # Reduced learning rate for more stable training
    num_epochs = num_epochs
    batch_size = batch_size
    target_accuracy = target_accuracy
    
    # Dataset parameters - adjusted for better initial training
    num_graphs = 1000  # Reduced for faster initial training
    n_nodes = 50  # Reduced for easier initial learning
    density = 0.3  # Graph density
    num_pairs = 30  # Reduced for faster training
    
    # Device selection
    if device_choice == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_choice)
    
    print(f"\nDevice configuration:")
    print(f"Selected device: {device}")
    if device.type == 'cuda':
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        # Configurer CUDA pour utiliser plus de mémoire
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        # Forcer l'utilisation de la mémoire GPU
        torch.cuda.set_per_process_memory_fraction(0.8)  # Utiliser 80% de la mémoire GPU
    
    # Créer le modèle
    model = ShortestPathGNN(input_dim=input_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    
    def print_gpu_memory():
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(0) / 1024**2
            reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    
    print_gpu_memory()
    
    # Optimiseur et scheduler - adjusted for better convergence
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)  # Increased weight decay
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)  # Adjusted scheduler
    dist_criterion = nn.HuberLoss()  # Pour la distance
    path_criterion = nn.BCELoss()  # Pour les probabilités de chemin
    
    # Variables pour le suivi de l'entraînement
    start_epoch = 0
    best_accuracy = 0
    best_loss = float('inf')
    patience = 10  # Early stopping patience
    no_improve_epochs = 0
    
    # Charger le dernier checkpoint si disponible
    if resume_training and os.path.exists("best_gnn_shortest_path.pt"):
        print("\nTentative de chargement du dernier checkpoint...")
        try:
            checkpoint = torch.load("best_gnn_shortest_path.pt", map_location=device)
            if all(key in checkpoint['model_state_dict'] for key in model.state_dict().keys()):
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_accuracy = checkpoint['accuracy']
                print(f"Checkpoint chargé avec succès - Epoch: {start_epoch}, Meilleure précision: {best_accuracy:.2f}%")
            else:
                print("Architecture du modèle modifiée, démarrage d'un nouvel entraînement...")
                if os.path.exists("best_gnn_shortest_path.pt"):
                    os.rename("best_gnn_shortest_path.pt", "best_gnn_shortest_path_old.pt")
        except Exception as e:
            print(f"Erreur lors du chargement du checkpoint: {str(e)}")
            print("Démarrage d'un nouvel entraînement...")
            if os.path.exists("best_gnn_shortest_path.pt"):
                os.rename("best_gnn_shortest_path.pt", "best_gnn_shortest_path_old.pt")
    
    # Créer les datasets avec progression de difficulté
    print("\nCréation des datasets...")
    easy_dataset = create_dataset(num_graphs//2, n_nodes, density, num_pairs, 'easy')
    medium_dataset = create_dataset(num_graphs//4, n_nodes, density, num_pairs, 'medium')
    hard_dataset = create_dataset(num_graphs//4, n_nodes, density, num_pairs, 'hard')
    
    # Combiner les datasets
    dataset = easy_dataset + medium_dataset + hard_dataset
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_set, val_set = dataset[:split], dataset[split:]
    
    print(f"Taille du dataset d'entraînement: {len(train_set)}")
    print(f"Taille du dataset de validation: {len(val_set)}")
    print_gpu_memory()
    
    # Entraînement
    model.train()
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        random.shuffle(train_set)
        total_loss = 0
        model.train()
        
        # Barre de progression pour chaque epoch
        pbar = tqdm(range(0, len(train_set), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i in pbar:
            batch = train_set[i:i+batch_size]
            datas, srcs, tgts, dists = collate_batch(batch)
            
            # Déplacer les données vers GPU
            datas = [data.to(device) for data in datas]
            dists = dists.to(device)
            
            optimizer.zero_grad()
            batch_loss = 0
            
            for data, src, tgt, dist in zip(datas, srcs, tgts, dists):
                path_probs, pred_dist = model(data)
                
                # Calculer la perte de distance
                dist_loss = dist_criterion(pred_dist[tgt], dist)
                
                # Calculer la perte de chemin
                path_labels = torch.zeros_like(path_probs)
                try:
                    # Obtenir le vrai chemin avec Dijkstra
                    adj_dict = {i: [] for i in range(data.x.size(0))}
                    for j in range(data.edge_index.size(1)):
                        u, v = data.edge_index[0, j].item(), data.edge_index[1, j].item()
                        w = data.edge_weight[j].item()
                        adj_dict[u].append((v, w))
                    
                    true_path, _ = Dijkstra.shortest_path(adj_dict, src, tgt)
                    if true_path:
                        path_labels[true_path] = 1.0
                except:
                    path_labels[src] = 1.0
                    path_labels[tgt] = 1.0
                
                path_loss = path_criterion(path_probs, path_labels)
                
                # Perte totale avec pondération
                loss = dist_loss + 0.5 * path_loss  # Reduced path loss weight
                loss.backward()
                batch_loss += loss.item()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += batch_loss
            
            # Mettre à jour la barre de progression
            pbar.set_postfix({'loss': f'{batch_loss/len(batch):.4f}'})
            
            # Afficher l'utilisation de la mémoire GPU périodiquement
            if i % (batch_size * 10) == 0:
                print_gpu_memory()
        
        scheduler.step()
        avg_loss = total_loss / len(train_set)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        print_gpu_memory()
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        # Évaluation sur le validation set
        model.eval()
        correct = 0
        total_error = 0
        path_correct = 0
        with torch.no_grad():
            for data, src, tgt, dist in val_set:
                data = data.to(device)
                path_probs, pred_dist = model(data)
                
                # Évaluer la distance
                pred = pred_dist[tgt].item()
                if abs(pred - dist) / dist < 0.1:  # Prédiction à 10% près
                    correct += 1
                total_error += abs(pred - dist) / dist
                
                # Évaluer le chemin
                pred_path = model.get_path(path_probs, data.edge_index)
                try:
                    # Obtenir le vrai chemin
                    adj_dict = {i: [] for i in range(data.x.size(0))}
                    for j in range(data.edge_index.size(1)):
                        u, v = data.edge_index[0, j].item(), data.edge_index[1, j].item()
                        w = data.edge_weight[j].item()
                        adj_dict[u].append((v, w))
                    
                    true_path, _ = Dijkstra.shortest_path(adj_dict, src, tgt)
                    if true_path and set(pred_path) == set(true_path):
                        path_correct += 1
                except:
                    pass
        
        accuracy = correct / len(val_set) * 100
        path_accuracy = path_correct / len(val_set) * 100
        avg_error = total_error / len(val_set) * 100
        print(f"Validation Distance Accuracy: {accuracy:.2f}%")
        print(f"Validation Path Accuracy: {path_accuracy:.2f}%")
        print(f"Average Relative Error: {avg_error:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Sauvegarder le meilleur modèle
            if save_callback:
                save_callback(model, optimizer, epoch, accuracy, path_accuracy, avg_error)
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'path_accuracy': path_accuracy,
                    'avg_error': avg_error,
                }, "best_gnn_shortest_path.pt")
                print(f"New best model saved with accuracy: {accuracy:.2f}%")
        
        if accuracy >= target_accuracy:
            print("Accuracy atteinte, arrêt de l'entraînement.")
            break
    
    # Sauvegarder le modèle final
    if save_callback:
        save_callback(model, optimizer, num_epochs, accuracy, path_accuracy, avg_error)
    else:
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'path_accuracy': path_accuracy,
            'avg_error': avg_error,
        }, "final_gnn_shortest_path.pt")
    
    training_time = time.time() - start_time
    print(f"\nTemps d'entraînement total: {training_time:.2f} secondes")
    print(f"Meilleure précision atteinte: {best_accuracy:.2f}%")
    print(f"Meilleure précision de chemin: {path_accuracy:.2f}%")
    
    # Comparaison avec Dijkstra et Bellman-Ford
    print("\nComparaison des performances:")
    dijkstra_times, bf_times = [], []
    for data, src, tgt, dist in val_set:
        adj_matrix = data.edge_weight.new_zeros((n_nodes, n_nodes))
        edge_idx = data.edge_index
        adj_matrix[edge_idx[0], edge_idx[1]] = data.edge_weight
        adj_dict = {i: [] for i in range(n_nodes)}
        for i in range(edge_idx.size(1)):
            u, v = edge_idx[0, i].item(), edge_idx[1, i].item()
            w = data.edge_weight[i].item()
            adj_dict[u].append((v, w))
        
        # Dijkstra
        start = time.time()
        _, dijkstra_dist = Dijkstra.shortest_path(adj_dict, src, tgt)
        dijkstra_times.append(time.time() - start)
        
        # Bellman-Ford
        start = time.time()
        _, bf_dist = BellmanFord.shortest_path(adj_dict, src, tgt)
        bf_times.append(time.time() - start)
    
    print(f"Temps moyen Dijkstra: {np.mean(dijkstra_times):.6f}s")
    print(f"Temps moyen Bellman-Ford: {np.mean(bf_times):.6f}s")
    
    if device.type == 'cuda':
        print(f"\nGPU Memory final:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(
        resume_training=args.resume,
        device_choice=args.device,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        target_accuracy=args.target_accuracy
    ) 