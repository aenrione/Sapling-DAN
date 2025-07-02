from collections import defaultdict
import numpy as np
from datasets import load_from_disk

class Loader:
    def __init__(self, dataset_name="amazon_reviews_books", subset_name="full", number_of_samples=100, reviews_column="rating", asin_column="asin", user_id_column="user_id"):
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.number_of_samples = number_of_samples
        self.reviews = load_from_disk(self.dataset_name)[self.subset_name]
        self.user_positive_reviews = defaultdict(list)
        self.user_set = set()
        self.item_set = set()
        self.user_list_path = f"{self.dataset_name}_{self.subset_name}_user_list.txt"
        self.item_list_path = f"{self.dataset_name}_{self.subset_name}_item_list.txt"
        self.user_positive_reviews_path = f"{self.dataset_name}_{self.subset_name}_user_positive_reviews.txt" 
        self.reviews_column = reviews_column
        self.asin_column = asin_column
        self.user_id_column = user_id_column

    def create_user_positive_reviews(self):
        for index, review in enumerate(self.reviews):
            if float(review[self.reviews_column]) >= 3:
                user_id = review[self.user_id_column]
                asin = review[self.asin_column]
                self.user_positive_reviews[user_id].append(asin)
                self.user_set.add(user_id)
                self.item_set.add(asin)

            if index == self.number_of_samples - 1:
                break

        user2id = {uid: i for i, uid in enumerate(sorted(self.user_set))}
        item2id = {asin: i for i, asin in enumerate(sorted(self.item_set))}

        with open(self.user_list_path, "w", encoding="utf-8") as f:
            f.write("original_id remap_id\n")
            for uid, rid in user2id.items():
                f.write(f"{uid} {rid}\n")

        with open(self.item_list_path, "w", encoding="utf-8") as f:
            f.write("original_id remap_id\n")
            for asin, rid in item2id.items():
                f.write(f"{asin} {rid}\n")

        with open(self.user_positive_reviews_path, "w", encoding="utf-8") as f:
            for user_id, asin_list in self.user_positive_reviews.items():
                remap_user = user2id[user_id]
                remap_items = [str(item2id[asin]) for asin in asin_list if asin in item2id]
                if remap_items:
                    f.write(f"{remap_user} " + " ".join(remap_items) + "\n")

    def create_user_item_matrix(self):
        user_item_map = {}
        max_user_id = 0
        max_item_id = 0

        with open(self.user_positive_reviews_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                user_id = int(parts[0])
                item_ids = list(map(int, parts[1:]))
                user_item_map[user_id] = item_ids
                max_user_id = max(max_user_id, user_id)
                max_item_id = max(max_item_id, max(item_ids, default=0))

        matrix = np.zeros((max_user_id + 1, max_item_id + 1), dtype=np.int8)

        for user_id, item_ids in user_item_map.items():
            for item_id in item_ids:
                matrix[user_id, item_id] = 1

        return matrix


if __name__ == "__main__":
    loader = Loader()
    loader.create_user_positive_reviews()
    user_item_matrix = loader.create_user_item_matrix()

    print("User-Item Matrix:")
    print(user_item_matrix)
