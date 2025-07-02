import warnings
import argparse
import numpy as np
from loader import Loader
from pipeline import Pipeline
from combine import Combine
from regularization import Regularization
from dan import LAE_DAN, EASE_DAN, RLAE_DAN
from sapling_similarity import SaplingSimilarity
from metrics import evaluate
warnings.filterwarnings("ignore")

def train_test_split(matrix, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    num_users, num_items = matrix.shape
    train = np.zeros_like(matrix)
    test = np.zeros_like(matrix)

    for user in range(num_users):
        items = np.where(matrix[user] == 1)[0]
        if len(items) < 2:
            train[user, items] = 1
            continue
        test_size = max(1, int(len(items) * test_ratio))
        test_items = np.random.choice(items, size=test_size, replace=False)
        train_items = list(set(items) - set(test_items))
        train[user, train_items] = 1
        test[user, test_items] = 1

    return train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pipeline", "combine", "regularization", "dan", "sapling"], default="pipeline", help="Model to evaluate")
    parser.add_argument("--normalization", choices=["LAE", "EASE", "RLAE"], default="RLAE", help="Normalization method (for DAN-based models)")
    parser.add_argument("--k", type=int, nargs="+", default=[10], help="List of K values for metrics")
    args = parser.parse_args()

    loader = Loader(number_of_samples=5000)
    loader.create_user_positive_reviews()
    full_matrix = loader.create_user_item_matrix()
    train, test = train_test_split(full_matrix)

    config = {
        'reg_p': 20,
        'alpha': 0.2,
        'beta': 0.3,
        'drop_p': 0.5,
        'xi': 0.3,
        'eta': 10
    }
    gamma = 0.5
    alpha = 0.5

    if args.model == "pipeline":
        model = Pipeline(train, config, gamma)
        predictions = model.run(args.normalization)
    elif args.model == "combine":
        model = Combine(train, config, gamma, alpha)
        predictions = model.run(args.normalization)
    elif args.model == "regularization":
        model = Regularization(train, config, gamma)
        predictions = model.get_weighted_matrix(args.normalization)
    elif args.model == "dan":
        if args.normalization == "LAE":
            model = LAE_DAN(train, config)
        elif args.normalization == "EASE":
            model = EASE_DAN(train, config)
        elif args.normalization == "RLAE":
            model = RLAE_DAN(train, config)
        W = model.get_weighted_matrix()
        predictions = train @ W
    elif args.model == "sapling":
        sapling = SaplingSimilarity(train, gamma)
        users_similarity = sapling.sapling(projection=0)
        items_similarity = sapling.sapling(projection=1)
        predictions = sapling.recommendation_matrix(users_similarity, items_similarity)
    else:
        raise ValueError("Invalid model choice. Choose from 'pipeline', 'combine', 'regularization', 'dan', or 'sapling'.")

    for k_val in args.k:
        results = evaluate(predictions, train, test, k=k_val)
        print(f"Metrics @ {k_val} for model {args.model}: ", results)

# CLI Example:
# py main.py --model pipeline --normalization RLAE --k 10 20
# py main.py --model combine --normalization EASE --k 10 20
# py main.py --model regularization --normalization LAE --k 10 20
# py main.py --model dan --normalization RLAE --k 10 20
# py main.py --model sapling --k 10 20
# py main.py --model dan --normalization EASE --k 10 20
