import numpy as np
import warnings
from pipeline import Pipeline
from combine import Combine
from regularization import Regularization
from dan import LAE_DAN, EASE_DAN, RLAE_DAN
from sapling_similarity import SaplingSimilarity
from metrics import evaluate
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt

def plot_grouped_bar(df, metric, k_val):
    # Filter for the desired k
    df_k = df[df["k"] == k_val]
    models = df_k["model"].unique()
    datasets = df_k["dataset"].unique()
    n_models = len(models)
    n_datasets = len(datasets)
    
    # Bar width and positions
    bar_width = 0.8 / n_datasets
    x = np.arange(n_models)
    
    plt.figure(figsize=(12, 6))
    
    for i, dataset in enumerate(datasets):
        values = []
        for model in models:
            val = df_k[(df_k["model"] == model) & (df_k["dataset"] == dataset)][metric]
            values.append(val.values[0] if not val.empty else 0)
        # Offset each dataset's bars
        plt.bar(x + i * bar_width, values, width=bar_width, label=dataset)
        # Add value labels
        for xi, v in zip(x + i * bar_width, values):
            plt.text(xi, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
    
    plt.xticks(x + bar_width * (n_datasets-1) / 2, models)
    plt.xlabel("Model")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.upper()}@{k_val} for each model and dataset")
    plt.legend(title="Dataset")
    plt.tight_layout()
    plt.show()

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
        ("pipeline", "LAE"),
        ("combine", "LAE"),
        ("regularization", "LAE"),
        ("dan", "LAE"),
        ("sapling", None),
        ("dan", "LAE"),
    ]
    results_list = []
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
            results_list.append({
                "dataset": loader.dataset_name,
                "model": model_name,
                "normalization": normalization,
                "k": k,
                "precision": results["precision@{}".format(k)],
                "recall": results["recall@{}".format(k)],
                "ndcg": results["ndcg@{}".format(k)]
            })
    return results_list

def sensitivity_analysis_combine(loader, config, gamma, k_values):
    loader.create_user_positive_reviews()
    full_matrix = loader.create_user_item_matrix()
    train, test = train_test_split(full_matrix)

    alpha_values = np.arange(0, 1.1, 0.1)

    results_list = []
    for alpha in alpha_values:
        model = Combine(train, config, gamma, alpha)
        predictions = model.run("LAE")

        for k in k_values:
            results = evaluate(predictions, train, test, k=k)
            print(f"Metrics @ {k} for model combine (alpha={alpha}): {results}")
            results_list.append({
                "dataset": loader.dataset_name,
                "model": "combine",
                "normalization": "LAE",
                "k": k,
                "alpha": alpha,
                "precision": results["precision@{}".format(k)],
                "recall": results["recall@{}".format(k)],
                "ndcg": results["ndcg@{}".format(k)]
            })

    return results_list