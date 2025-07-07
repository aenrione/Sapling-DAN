from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def precision_at_k(ranked_items, true_items, k):
    return np.sum([item in true_items for item in ranked_items[:k]]) / k


def recall_at_k(ranked_items, true_items, k):
    if len(true_items) == 0:
        return 0.0
    return np.sum([item in true_items for item in ranked_items[:k]]) / len(true_items)


def ndcg_at_k(ranked_items, true_items, k):
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in true_items:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum([1 / np.log2(i + 2)
                    for i in range(min(len(true_items), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0


def diversity_at_k(ranked_items, similarity_matrix, k):
    """
    Calcula la diversidad como 1 - promedio de similaridad entre los ítems recomendados.
    similarity_matrix: matriz de similaridad entre ítems (shape [n_items, n_items])
    """
    items = ranked_items[:k]
    if len(items) < 2:
        return 0.0
    sim_sum = 0.0
    count = 0
    for i, _ in enumerate(items):
        for j in range(i + 1, len(items)):
            sim = similarity_matrix[items[i], items[j]]
            sim_sum += sim
            count += 1
    return 1 - (sim_sum / count if count > 0 else 0)


def novelty_at_k(ranked_items, item_popularity, k):
    """
    Calcula la novedad como el promedio del log inverso de la popularidad de los ítems.
    item_popularity: array donde item_popularity[i] es la cantidad de usuarios que interactuaron con el ítem i.
    """
    novelty = 0.0
    for item in ranked_items[:k]:
        pop = item_popularity[item]
        novelty += -np.log2((pop + 1) / np.sum(item_popularity))
    return novelty / k


def compute_item_popularity(train_matrix: np.ndarray):
    """
    Retorna un vector con la cantidad de usuarios que han interactuado con cada ítem.
    """
    return np.array(train_matrix.sum(axis=0)).flatten()


def compute_item_similarity(train_matrix: np.ndarray):
    """
    Calcula la matriz de similaridad del coseno entre ítems (shape: [num_items, num_items]).
    """
    item_matrix = train_matrix.T
    return cosine_similarity(item_matrix)


def evaluate(pred_matrix, train_matrix, test_matrix, k=10):
    similarity_matrix = compute_item_similarity(train_matrix)
    item_popularity = compute_item_popularity(train_matrix)

    num_users = pred_matrix.shape[0]
    precisions, recalls, ndcgs = [], [], []
    diversities, novelties = [], []

    for user in range(num_users):
        train_items = set(np.where(train_matrix[user] == 1)[0])
        test_items = set(np.where(test_matrix[user] == 1)[0])
        if not test_items:
            continue

        scores = pred_matrix[user].copy()
        scores[list(train_items)] = -np.inf
        ranked_items = np.argsort(-scores)

        precisions.append(precision_at_k(ranked_items, test_items, k))
        recalls.append(recall_at_k(ranked_items, test_items, k))
        ndcgs.append(ndcg_at_k(ranked_items, test_items, k))
        diversities.append(diversity_at_k(ranked_items, similarity_matrix, k))
        novelties.append(novelty_at_k(ranked_items, item_popularity, k))

    results = {
        f"precision@{k}": np.mean(precisions),
        f"recall@{k}": np.mean(recalls),
        f"ndcg@{k}": np.mean(ndcgs),
        f"diversity@{k}": np.mean(diversities),
        f"novelty@{k}": np.mean(novelties)
    }

    return results
