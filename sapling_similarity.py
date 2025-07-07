import numpy as np

class SaplingSimilarity:
    """Sapling Similarity class for computing similarity matrices based on user-item interactions."""

    def __init__(self, user_item_matrix: np.ndarray, gamma: float = 0.5):
        self.user_item_matrix = user_item_matrix.astype(np.float64)
        self.gamma = gamma

    def sapling(self, projection: int = 0) -> np.ndarray:
        """Similarity algorithm for projection-based similarity matrix.

        Args:
            self.user_item_matrix (np.ndarray): User-item interaction matrix where each data point is either 0 or 1.
            The matrix should be of shape (n_users, n_items)
            projection (int, optional): Projection type. 0 for user-based, 1 for item-based. Defaults to 0.

        Returns:
            np.ndarray: Similarity matrix of shape (n_users, n_users) for user-based projection or
                        (n_items, n_items) for item-based projection.
        """

        if projection == 0:
            number_of_users = self.user_item_matrix.shape[1]
            users_interactions: np.ndarray = np.sum(self.user_item_matrix, axis=1)
            co_ocurrences_matrix: np.ndarray = np.dot(self.user_item_matrix, self.user_item_matrix.T)
            B = np.nan_to_num((1 - (co_ocurrences_matrix * (1 - co_ocurrences_matrix / users_interactions) + (users_interactions - co_ocurrences_matrix.T).T * (1 - (users_interactions - co_ocurrences_matrix.T).T / (
                number_of_users - users_interactions))).T / (users_interactions * (1 - users_interactions / number_of_users))).T * np.sign(((co_ocurrences_matrix * number_of_users / users_interactions).T / users_interactions).T - 1))
        else:
            number_of_items = self.user_item_matrix.shape[0]
            items_interactions: np.ndarray = np.sum(self.user_item_matrix, axis=0)
            co_ocurrences_matrix: np.ndarray = np.dot(self.user_item_matrix.T, self.user_item_matrix)
            B = np.nan_to_num((1 - (co_ocurrences_matrix * (1 - co_ocurrences_matrix / items_interactions) + (items_interactions - co_ocurrences_matrix.T).T * (1 - (items_interactions - co_ocurrences_matrix.T).T / (
                number_of_items - items_interactions))).T / (items_interactions * (1 - items_interactions / number_of_items))).T * np.sign(((co_ocurrences_matrix * number_of_items / items_interactions).T / items_interactions).T - 1))
        return B

    def recommendation_matrix(self, users_similarity: np.ndarray, items_similarity: np.ndarray) -> np.ndarray:
        """Compute the final recommendation matrix based on user and item similarities.

        Args:
            users_similarity (np.ndarray): User similarity matrix.
            items_similarity (np.ndarray): Item similarity matrix.

        Returns:
            np.ndarray: Final recommendation matrix.
        """
        users_recommendations = np.nan_to_num(np.dot(users_similarity, self.user_item_matrix).T / np.sum(abs(users_similarity), axis=1)).T
        items_recommendations = np.nan_to_num(np.dot(self.user_item_matrix, items_similarity) / np.sum(abs(items_similarity), axis=0))
        
        return (1 - self.gamma) * users_recommendations + self.gamma * items_recommendations

if __name__ == "__main__":
    matrix = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0],
                       [1, 0, 0, 1, 0],
                       [0, 1, 0, 1, 1]])
    user = np.array([1, 0, 1, 0, 0])

    gamma = 0.5

    sapling_similarity = SaplingSimilarity(matrix, gamma)
    users_similarity = sapling_similarity.sapling(projection=0)
    items_similarity = sapling_similarity.sapling(projection=1)

    recommendation_matrix = sapling_similarity.recommendation_matrix(users_similarity, items_similarity)
    print("Recommendation Matrix:\n", recommendation_matrix)
    print("User Similarity Matrix:\n", users_similarity)
    print("Item Similarity Matrix:\n", items_similarity)
