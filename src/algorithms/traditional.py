"""
Module contenant les implémentations des algorithmes traditionnels de plus court chemin.
"""

import heapq
from typing import Dict, List, Tuple, Set

class Dijkstra:
    """
    Implémentation de l'algorithme de Dijkstra pour trouver le plus court chemin.
    """
    
    @staticmethod
    def shortest_path(adj_dict: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> Tuple[List[int], float]:
        """
        Implémentation de l'algorithme de Dijkstra.
        
        Args:
            adj_dict: Dictionnaire d'adjacence {nœud: [(voisin, poids), ...]}
            start: Nœud de départ
            end: Nœud d'arrivée
            
        Returns:
            Tuple contenant:
            - Liste des nœuds dans le chemin le plus court
            - Distance totale du chemin
        """
        # Initialisation
        distances = {node: float('inf') for node in adj_dict}
        distances[start] = 0
        previous = {node: None for node in adj_dict}
        visited = set()
        
        # File de priorité
        pq = [(0, start)]
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current == end:
                break
                
            if current in visited:
                continue
                
            visited.add(current)
            
            # Explorer les voisins
            for neighbor, weight in adj_dict[current]:
                if neighbor in visited:
                    continue
                    
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruire le chemin
        path = []
        if distances[end] != float('inf'):
            current = end
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()
        
        return path, distances[end]

class BellmanFord:
    """
    Implémentation de l'algorithme de Bellman-Ford pour trouver le plus court chemin.
    """
    
    @staticmethod
    def shortest_path(adj_dict: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> Tuple[List[int], float]:
        """
        Implémentation de l'algorithme de Bellman-Ford.
        
        Args:
            adj_dict: Dictionnaire d'adjacence {nœud: [(voisin, poids), ...]}
            start: Nœud de départ
            end: Nœud d'arrivée
            
        Returns:
            Tuple contenant:
            - Liste des nœuds dans le chemin le plus court
            - Distance totale du chemin
        """
        # Initialisation
        n = len(adj_dict)
        distances = {node: float('inf') for node in adj_dict}
        distances[start] = 0
        previous = {node: None for node in adj_dict}
        
        # Relaxation des arêtes
        for _ in range(n - 1):
            for node in adj_dict:
                for neighbor, weight in adj_dict[node]:
                    if distances[node] + weight < distances[neighbor]:
                        distances[neighbor] = distances[node] + weight
                        previous[neighbor] = node
        
        # Vérification des cycles négatifs
        for node in adj_dict:
            for neighbor, weight in adj_dict[node]:
                if distances[node] + weight < distances[neighbor]:
                    return [], float('inf')  # Cycle négatif détecté
        
        # Reconstruire le chemin
        path = []
        if distances[end] != float('inf'):
            current = end
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()
        
        return path, distances[end] 