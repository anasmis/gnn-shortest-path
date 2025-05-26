"""
Module contenant l'implémentation du modèle GNN pour la résolution du problème du plus court chemin.
Utilise PyTorch Geometric pour des opérations graphiques efficaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

class ShortestPathGNN(nn.Module):
    """
    Modèle GNN pour la prédiction du plus court chemin.
    
    Architecture:
    - GCN layers avec skip connections
    - Path prediction
    - Position encoding
    """
    
    def __init__(self, input_dim, hidden_dim=256):
        """
        Initialisation du modèle.
        
        Args:
            input_dim (int): Dimension des caractéristiques d'entrée
            hidden_dim (int): Dimension des couches cachées
        """
        super(ShortestPathGNN, self).__init__()
        
        # Position encoding
        self.pos_encoder = nn.Linear(2, hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # GCN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)  # Ajout d'une couche
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)  # Ajout d'une couche
        
        # Skip connections
        self.skip1 = nn.Linear(hidden_dim, hidden_dim)
        self.skip2 = nn.Linear(hidden_dim, hidden_dim)
        self.skip3 = nn.Linear(hidden_dim, hidden_dim)  # Ajout d'une connexion
        
        # Path prediction layers
        self.path_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Distance prediction layer
        self.dist_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, data: Data):
        """
        Propagation avant du modèle.
        
        Args:
            data (Data): Objet PyTorch Geometric contenant:
                - x: Caractéristiques des nœuds
                - edge_index: Indices des arêtes
                - edge_weight: Poids des arêtes
                - batch: Indices de batch
                - pos: Position encoding (optionnel)
            
        Returns:
            tuple: (path_probs, distance) où:
                - path_probs: Probabilités de chaque nœud d'être dans le chemin
                - distance: Distance prédite
        """
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        # Ajouter des self-loops
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
        
        # Position encoding et projection
        if hasattr(data, 'pos'):
            pos_enc = self.pos_encoder(data.pos)
            x = torch.cat([x, pos_enc], dim=1)
            x = self.input_proj(x)
        
        # Première couche GCN
        identity = x
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x + self.skip1(identity)
        
        # Deuxième couche GCN
        identity = x
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x + self.skip2(identity)
        
        # Troisième couche GCN
        identity = x
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x + self.skip3(identity)
        
        # Quatrième couche GCN
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Prédictions
        path_probs = torch.sigmoid(self.path_predictor(x)).squeeze(-1)  # Probabilités de chemin
        distance = self.dist_predictor(x).squeeze(-1)  # Distances
        
        return path_probs, distance
    
    def get_path(self, path_probs, edge_index, threshold=0.5):
        """
        Reconstruit le chemin à partir des probabilités.
        
        Args:
            path_probs (torch.Tensor): Probabilités de chaque nœud
            edge_index (torch.Tensor): Indices des arêtes
            threshold (float): Seuil pour considérer un nœud dans le chemin
            
        Returns:
            list: Liste des nœuds dans le chemin
        """
        # Obtenir les nœuds avec probabilité > threshold
        path_nodes = torch.where(path_probs > threshold)[0].tolist()
        
        # Reconstruire le chemin en utilisant les arêtes
        path = []
        if path_nodes:
            # Commencer par le nœud avec la plus haute probabilité
            current = max(path_nodes, key=lambda x: path_probs[x])
            path.append(current)
            
            # Ajouter les nœuds connectés dans l'ordre des probabilités
            while len(path) < len(path_nodes):
                neighbors = []
                for i in range(edge_index.size(1)):
                    if edge_index[0, i] == current and edge_index[1, i] in path_nodes:
                        neighbors.append(edge_index[1, i].item())
                    elif edge_index[1, i] == current and edge_index[0, i] in path_nodes:
                        neighbors.append(edge_index[0, i].item())
                
                if not neighbors:
                    break
                    
                # Choisir le voisin avec la plus haute probabilité
                current = max(neighbors, key=lambda x: path_probs[x])
                if current not in path:
                    path.append(current)
        
        return path

def create_graph_data(adj_matrix, node_features=None, pos_encoding=None):
    """
    Crée un objet Data de PyTorch Geometric avec position encoding.
    
    Args:
        adj_matrix (torch.Tensor): Matrice d'adjacence [N, N]
        node_features (torch.Tensor, optional): Caractéristiques des nœuds [N, F]
        pos_encoding (torch.Tensor, optional): Position encoding [N, 2]
        
    Returns:
        Data: Objet PyTorch Geometric
    """
    # Convertir la matrice d'adjacence en indices d'arêtes
    edge_index = torch.nonzero(adj_matrix).t()
    edge_weight = adj_matrix[edge_index[0], edge_index[1]]
    
    # Si pas de caractéristiques de nœuds, utiliser des ones
    if node_features is None:
        node_features = torch.ones(adj_matrix.size(0), 1)
    
    # Créer l'objet Data
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight
    )
    
    # Ajouter position encoding si disponible
    if pos_encoding is not None:
        data.pos = pos_encoding
    
    return data 