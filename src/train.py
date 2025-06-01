"""
Script d'entraînement rapide et d'évaluation du modèle GNN avec comparaison des performances.
Optimisé pour un entraînement très court.
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
from torch.optim.lr_scheduler import StepLR

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gnn_model import ShortestPathGNN, create_graph_data
from src.algorithms.traditional import Dijkstra, BellmanFord

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GNN for shortest path prediction (fast version)')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cuda', 'cpu'],
                      help='Device to use for training (auto: use CUDA if available)')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from last checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,  # Reduced
                      help='Batch size for training')
    parser.add_argument('--hidden-dim', type=int, default=128,  # Reduced
                      help='Hidden dimension of the GNN')
    parser.add_argument('--num-epochs', type=int, default=100,  # Much reduced
                      help='Number of training epochs')
    parser.add_argument('--target-accuracy', type=float, default=85,  # Lower target
                      help='Target accuracy to stop training')
    return parser.parse_args()

def generate_simple_graph(n_nodes, density=0.4):
    """
    Génère un graphe simple pour un entraînement rapide.
    """
    adj_matrix = torch.zeros(n_nodes, n_nodes)
    
    # Densité plus élevée pour faciliter l'apprentissage
    weight_range = (1.0, 5.0)  # Poids simples
    
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

def create_fast_dataset(num_graphs=50, n_nodes=15, density=0.4, num_pairs=5):
    """Crée un petit dataset pour un entraînement rapide."""
    dataset = []
    for _ in range(num_graphs):
        adj_matrix, adj_dict = generate_simple_graph(n_nodes, density)
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

def train_and_evaluate_fast(resume_training=False, save_callback=None, device_choice='auto', 
                           batch_size=64, hidden_dim=128, num_epochs=50, target_accuracy=90):
    # Configuration simplifiée pour entraînement rapide
    input_dim = 1
    learning_rate = 0.005  # Taux d'apprentissage plus élevé
    
    # Paramètres de dataset réduits
    num_graphs = 1000  # Très réduit
    n_nodes = 30      # Très réduit
    density = 0.15     # Densité 
    num_pairs = 10     # Très réduit
    
    # Device selection
    if device_choice == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_choice)
    
    print(f"🚀 Mode entraînement rapide activé!")
    print(f"Device: {device}")
    print(f"Graphes: {num_graphs}, Nœuds: {n_nodes}, Paires: {num_pairs}")
    print(f"Epochs max: {num_epochs}, Target accuracy: {target_accuracy}")
    
    # Créer le modèle simplifié
    model = ShortestPathGNN(input_dim=input_dim, hidden_dim=hidden_dim)
    model = model.to(device)
    
    # Optimiseur simplifié
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)  # Réduction plus agressive
    criterion = nn.MSELoss()  # Critère simplifié
    
    # Variables pour le suivi
    start_epoch = 0
    best_accuracy = 0
    patience = 10  # Patience réduite
    no_improve_epochs = 0
    
    # Charger le checkpoint si disponible
    if resume_training and os.path.exists("best_gnn_shortest_path.pt"):
        print("Chargement du checkpoint...")
        try:
            checkpoint = torch.load("best_gnn_shortest_path.pt", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['accuracy']
            print(f"✅ Checkpoint chargé - Epoch: {start_epoch}, Accuracy: {best_accuracy:.2f}%")
        except Exception as e:
            print(f"❌ Erreur checkpoint: {str(e)}")
    
    # Créer le dataset simplifié
    print("Création du dataset rapide...")
    dataset = create_fast_dataset(num_graphs, n_nodes, density, num_pairs)
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_set, val_set = dataset[:split], dataset[split:]
    
    print(f"📊 Dataset: {len(train_set)} train, {len(val_set)} val")
    
    # Entraînement rapide
    model.train()
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        random.shuffle(train_set)
        total_loss = 0
        model.train()
        
        # Barre de progression simplifiée
        num_batches = len(train_set) // batch_size + (1 if len(train_set) % batch_size != 0 else 0)
        pbar = tqdm(range(0, len(train_set), batch_size), 
                   desc=f"Epoch {epoch+1:2d}/{num_epochs}", 
                   leave=False)
        
        for i in pbar:
            batch = train_set[i:i+batch_size]
            if len(batch) == 0:
                continue
                
            datas, srcs, tgts, dists = collate_batch(batch)
            
            # Déplacer vers GPU
            datas = [data.to(device) for data in datas]
            dists = dists.to(device)
            
            optimizer.zero_grad()
            batch_loss = 0
            
            for data, src, tgt, dist in zip(datas, srcs, tgts, dists):
                try:
                    path_probs, pred_dist = model(data)
                    # Perte simplifiée - seulement sur la distance
                    loss = criterion(pred_dist[tgt], dist)
                    loss.backward()
                    batch_loss += loss.item()
                except Exception as e:
                    continue
            
            # Gradient clipping léger
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += batch_loss
            
            pbar.set_postfix({'loss': f'{batch_loss/len(batch):.3f}'})
        
        scheduler.step()
        avg_loss = total_loss / max(len(train_set), 1)
        
        # Évaluation rapide
        model.eval()
        correct = 0
        total_error = 0
        
        with torch.no_grad():
            for data, src, tgt, dist in val_set:
                try:
                    data = data.to(device)
                    path_probs, pred_dist = model(data)
                    pred = 0.0
                    if len(path) > 1:
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i + 1]
                            pred += adj_matrix[u, v].item()
                    else:
                        pred = float('inf')  # No valid path
                    
                    # Tolérance plus large pour convergence rapide
                    if abs(pred - dist) / max(dist, 1e-6) < 0.15:  # 15% de tolérance
                        correct += 1
                    total_error += abs(pred - dist) / max(dist, 1e-6)
                except Exception:
                    continue
        
        accuracy = correct / max(len(val_set), 1) * 100
        avg_error = total_error / max(len(val_set), 1) * 100
        
        print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Acc={accuracy:.1f}%, Err={avg_error:.1f}%")
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improve_epochs = 0
            
            # Sauvegarder le meilleur modèle
            if save_callback:
                save_callback(model, optimizer, epoch, accuracy, 0, avg_error)
            else:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'avg_error': avg_error,
                }, "best_gnn_shortest_path.pt")
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= patience:
            print(f"🛑 Early stopping après {epoch + 1} epochs")
            break
            
        if accuracy >= target_accuracy:
            print(f"🎯 Target accuracy atteinte!")
            break
    
    training_time = time.time() - start_time
    print(f"\n✅ Entraînement terminé en {training_time:.1f}s")
    print(f"🏆 Meilleure accuracy: {best_accuracy:.2f}%")
    
    # Comparaison rapide avec les algorithmes traditionnels
    print("\n⚡ Comparaison des performances:")
    sample_size = min(10, len(val_set))  # Échantillon réduit
    dijkstra_times, bf_times = [], []
    
    for data, src, tgt, dist in val_set[:sample_size]:
        adj_dict = {i: [] for i in range(n_nodes)}
        edge_idx = data.edge_index
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
    
    print(f"Dijkstra: {np.mean(dijkstra_times)*1000:.2f}ms")
    print(f"Bellman-Ford: {np.mean(bf_times)*1000:.2f}ms")
    
    return model, best_accuracy

if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate_fast(
        resume_training=args.resume,
        device_choice=args.device,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_epochs=args.num_epochs,
        target_accuracy=args.target_accuracy
    )