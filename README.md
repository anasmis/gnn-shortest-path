# Résolveur de Chemin le Plus Court avec Réseaux de Neurones Graphiques (GNN)

**Auteur :** Anas AIT ALI

## Présentation

Ce projet implémente un Réseau de Neurones Graphique (GNN) pour résoudre le problème du chemin le plus court dans des graphes. Le modèle est développé en utilisant PyTorch Geometric et peut être entraîné sur des environnements CPU ou GPU.

## Fonctionnalités

* Prédiction du chemin le plus court entre deux nœuds dans un graphe.
* Prédiction de la distance minimale entre deux nœuds.
* Accélération GPU avec CUDA (optionnelle).
* Entraînement avec apprentissage progressif (curriculum learning).
* Comparaison avec des algorithmes classiques tels que Dijkstra et Bellman-Ford.

## Architecture du Projet

Le projet est structuré comme suit :

* `src/` : Contient les scripts principaux pour l'entraînement et l'évaluation du modèle.
* `colab_train.py` : Script pour l'entraînement du modèle dans un environnement Google Colab.
* `requirements.txt` : Liste des dépendances nécessaires pour exécuter le projet.
* `DOCUMENTATION.md` : Documentation détaillée sur l'utilisation du projet.

## Installation

1. **Cloner le dépôt :**

```bash
git clone https://github.com/anasmis/gnn-shortest-path.git
cd gnn-shortest-path
```

2. **Créer un environnement virtuel (optionnel mais recommandé) :**

```bash
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
```

3. **Installer les dépendances :**

```bash
pip install -r requirements.txt
```

## Utilisation

### Entraînement du Modèle

```bash
python src/train.py --epochs 100 --batch_size 32
```

*Paramètres optionnels :*

* `--epochs` : Nombre d'époques d'entraînement (par défaut : 100).
* `--batch_size` : Taille des lots (par défaut : 32).
* `--lr` : Taux d'apprentissage (par défaut : 0.001).
* `--cuda` : Utiliser CUDA si disponible (par défaut : False).

### Évaluation du Modèle

```bash
python src/evaluate.py --model_path best_gnn_shortest_path.pt
```

### Visualisation des Résultats

```bash
python src/visualize.py --model_path best_gnn_shortest_path.pt
```

## Comparaison avec les Algorithmes Classiques

Le projet inclut une comparaison entre le GNN et les algorithmes classiques de recherche de chemin le plus court :

* **Dijkstra** : Pour graphes pondérés sans poids négatifs.
* **Bellman-Ford** : Capable de gérer des arêtes à poids négatifs.

## Contributions

Les contributions sont les bienvenues !

1. Forkez le dépôt.
2. Créez une branche (`git checkout -b fonctionnalite`).
3. Commitez vos modifications (`git commit -am "Ajout d'une fonctionnalité"`).
4. Poussez la branche (`git push origin fonctionnalite`).
5. Ouvrez une Pull Request.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

---

Pour toute question, suggestion ou retour, vous pouvez me contacter directement via GitHub.
