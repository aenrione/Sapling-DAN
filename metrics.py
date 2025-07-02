
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
    ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_items), k))])
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

def evaluate(pred_matrix, train_matrix, test_matrix, k=10):
    num_users = pred_matrix.shape[0]
    precisions, recalls, ndcgs = [], [], []

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

    return {
        "precision@{}".format(k): np.mean(precisions),
        "recall@{}".format(k): np.mean(recalls),
        "ndcg@{}".format(k): np.mean(ndcgs)
    }
