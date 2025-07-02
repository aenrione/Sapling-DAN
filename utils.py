import numpy as np
import warnings
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

def run_experiments(loader, config, gamma, alpha, k_values):
    loader.create_user_positive_reviews()
    full_matrix = loader.create_user_item_matrix()
    train, test = train_test_split(full_matrix)

    # Lista de experimentos (modelo, normalización)
    experiments = [
        ("pipeline", "RLAE"),
        ("combine", "EASE"),
        ("regularization", "LAE"),
        ("dan", "RLAE"),
        ("sapling", None),
        ("dan", "EASE"),
    ]

    for model_name, normalization in experiments:
        if model_name == "pipeline":
            model = Pipeline(train, config, gamma)
            predictions = model.run(normalization)
        elif model_name == "combine":
            model = Combine(train, config, gamma, alpha)
            predictions = model.run(normalization)
        elif model_name == "regularization":
            model = Regularization(train, config, gamma)
            predictions = model.get_weighted_matrix(normalization)
        elif model_name == "dan":
            if normalization == "LAE":
                model = LAE_DAN(train, config)
            elif normalization == "EASE":
                model = EASE_DAN(train, config)
            elif normalization == "RLAE":
                model = RLAE_DAN(train, config)
            W = model.get_weighted_matrix()
            predictions = train @ W
        elif model_name == "sapling":
            sapling = SaplingSimilarity(train, gamma)
            users_similarity = sapling.sapling(projection=0)
            items_similarity = sapling.sapling(projection=1)
            predictions = sapling.recommendation_matrix(users_similarity, items_similarity)
        else:
            raise ValueError("Modelo no válido.")

        for k in k_values:
            results = evaluate(predictions, train, test, k=k)
            print(f"Metrics @ {k} for model {model_name} (normalization={normalization}): {results}")