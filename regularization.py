import numpy as np
from dan import LAE_DAN, EASE_DAN, RLAE_DAN
from sapling_similarity import SaplingSimilarity

class Regularization:
    """
    Regularization class for computing weighted matrices using different normalization methods.
    """

    def __init__(self, user_item_matrix: np.ndarray, dan_config: dict, gamma: float = 0.5):
        self.user_item_matrix = user_item_matrix
        self.config = dan_config
        self.gamma = gamma

    def get_weighted_matrix(self, normalization: str = "LAE") -> np.ndarray:
        """Get the weighted matrix based on the specified normalization method."""
        sapling = SaplingSimilarity(self.user_item_matrix, self.gamma)
        sapling_matrix = sapling.sapling(projection=1)

        if normalization == "LAE":
            dan_model = LAE_DAN(self.user_item_matrix, self.config, sapling_matrix)
        elif normalization == "EASE":
            dan_model = EASE_DAN(self.user_item_matrix, self.config, sapling_matrix)
        elif normalization == "RLAE":
            dan_model = RLAE_DAN(self.user_item_matrix, self.config, sapling_matrix)
        else:
            raise ValueError("Normalization method not recognized. Use 'LAE', 'EASE', or 'RLAE'.")

        dan_matrix = dan_model.get_weighted_matrix()
        recommendation_matrix = self.user_item_matrix @ dan_matrix

        return recommendation_matrix


if __name__ == "__main__":
    matrix = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0],
                       [1, 0, 0, 1, 0],
                       [0, 1, 0, 1, 1]])

    config = {
        'reg_p': 20,
        'alpha': 0.2,
        'beta': 0.3,
        'drop_p': 0.5,
        'xi': 0.3
    }
    gamma = 0.5

    regularization = Regularization(matrix, config, gamma)
    recommendation_matrix = regularization.get_weighted_matrix(normalization="RLAE")
    print("Recommendation Matrix:\n", recommendation_matrix)
