import numpy as np
from sapling_similarity import SaplingSimilarity
from dan import LAE_DAN, EASE_DAN, RLAE_DAN

class Combine:
    """Combine class for processing user-item interaction matrices and computing similarity matrices."""
    
    def __init__(self, user_item_matrix: np.ndarray, dan_config: dict, gamma: float = 0.5, alpha: float = 0.5):
        self.user_item_matrix = user_item_matrix
        self.config = dan_config
        self.gamma = gamma
        self.alpha = alpha

    def run(self, normalization: str = "LAE") -> np.ndarray:
        """Run the pipeline to compute the weighted matrix and similarity matrix."""
        if normalization == "LAE":
            dan_model = LAE_DAN(self.user_item_matrix, self.config)
        elif normalization == "EASE":
            dan_model = EASE_DAN(self.user_item_matrix, self.config)
        elif normalization == "RLAE":
            dan_model = RLAE_DAN(self.user_item_matrix, self.config)
        else:
            raise ValueError("Normalization method not recognized. Use 'LAE', 'EASE', or 'RLAE'.")

        sapling_similarity = SaplingSimilarity(self.user_item_matrix, self.gamma)

        weighted_matrix = dan_model.get_weighted_matrix()
        recommendation_matrix_dan = self.user_item_matrix @ weighted_matrix

        users_similarity = sapling_similarity.sapling(projection=0)
        items_similarity = sapling_similarity.sapling(projection=1)
        
        recommendation_matrix_sapling = sapling_similarity.recommendation_matrix(users_similarity, items_similarity)
        
        return (1 - self.alpha) * recommendation_matrix_dan + self.alpha * recommendation_matrix_sapling


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
    alpha = 0.5
    
    combine = Combine(matrix, config, gamma, alpha)
    recommendation_matrix = combine.run(normalization="RLAE")
    print("Recommendation Matrix:\n", recommendation_matrix)
