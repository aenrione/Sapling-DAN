import numpy as np


class LAE_DAN:
    def __init__(self, matrix, config: dict, sapling_matrix=None):
        self.matrix: np.ndarray = matrix
        self.numbers_of_items: int = matrix.shape[1]
        self.numbers_of_users: int = matrix.shape[0]
        self.reg_p = config['reg_p']
        self.alpha = 1 - config['alpha']
        self.beta = config['beta']
        self.drop_p = config['drop_p']
        self.gamma = 1
        self.sapling_matrix = sapling_matrix
        self.eta = config.get('eta', 0.0)

    def get_weighted_matrix(self):
        """
        Returns the weighted matrix based on the given parameters.
        """

        item_counts = np.sum(self.matrix, axis=0)
        user_counts = np.sum(self.matrix, axis=1)

        normalized_matrix: np.ndarray = (self.matrix * np.power(user_counts[:, np.newaxis], -self.beta)).T

        co_ocurrences_matrix = normalized_matrix.dot(self.matrix)

        if self.sapling_matrix is not None:
            identity = np.eye(self.numbers_of_items)
            penalty = identity - self.sapling_matrix
            co_ocurrences_matrix += self.eta * penalty


        lambda_parameter = self.reg_p + self.drop_p / (1 - self.drop_p) * item_counts
        co_ocurrences_matrix[np.diag_indices(self.numbers_of_items)] += lambda_parameter.reshape(-1)

        P = np.linalg.inv(co_ocurrences_matrix)

        B_DLAE = np.eye(self.numbers_of_items) - P * lambda_parameter

        item_power_term = np.power(item_counts, -(1 - self.alpha))

        self.W = B_DLAE * (1 / item_power_term).reshape(-1, 1) * item_power_term

        self.W[np.diag_indices(self.numbers_of_items)] = 0

        return self.W


class EASE_DAN:
    def __init__(self, matrix, config: dict, sapling_matrix=None):
        self.matrix: np.ndarray = matrix
        self.numbers_of_items: int = matrix.shape[1]
        self.numbers_of_users: int = matrix.shape[0]
        self.reg_p = config['reg_p']
        self.alpha = 1 - config['alpha']
        self.beta = config['beta']
        self.drop_p = config['drop_p']
        self.xi = config['xi']
        self.sapling_matrix = sapling_matrix
        self.eta = config.get('eta', 0.0)

    def get_weighted_matrix(self):
        """
        Returns the weighted matrix based on the given parameters.
        """

        item_counts = np.sum(self.matrix, axis=0)
        user_counts = np.sum(self.matrix, axis=1)

        normalized_matrix: np.ndarray = (self.matrix * np.power(user_counts[:, np.newaxis], -self.beta)).T

        co_ocurrences_matrix = normalized_matrix.dot(self.matrix)

        if self.sapling_matrix is not None:
            identity = np.eye(self.numbers_of_items)
            penalty = identity - self.sapling_matrix
            co_ocurrences_matrix += self.eta * penalty

        lambda_parameter = self.reg_p + self.drop_p / (1 - self.drop_p) * item_counts

        co_ocurrences_matrix[np.diag_indices(self.numbers_of_items)] += lambda_parameter.reshape(-1)

        P = np.linalg.inv(co_ocurrences_matrix)
        B_DLAE = np.eye(self.numbers_of_items) - P / np.diag(P)
        item_power_term = np.power(item_counts, -(1 - self.alpha))

        self.W = B_DLAE * (1 / item_power_term).reshape(-1, 1) * item_power_term
        self.W[np.diag_indices(self.numbers_of_items)] = 0

        return self.W


class RLAE_DAN:
    def __init__(self, matrix, config: dict, sapling_matrix=None):
        self.matrix: np.ndarray = matrix
        self.numbers_of_items: int = matrix.shape[1]
        self.numbers_of_users: int = matrix.shape[0]
        self.reg_p = config['reg_p']
        self.alpha = 1 - config['alpha']
        self.beta = config['beta']
        self.drop_p = config['drop_p']
        self.xi = config['xi']
        self.sapling_matrix = sapling_matrix
        self.eta = config.get('eta', 0.0)

    def get_weighted_matrix(self):
        """
        Returns the weighted matrix based on the given parameters.
        """

        item_counts = np.sum(self.matrix, axis=0)
        user_counts = np.sum(self.matrix, axis=1)

        normalized_matrix: np.ndarray = (self.matrix * np.power(user_counts[:, np.newaxis], -self.beta)).T

        co_ocurrences_matrix = normalized_matrix.dot(self.matrix)

        if self.sapling_matrix is not None:
            identity = np.eye(self.numbers_of_items)
            penalty = identity - self.sapling_matrix
            co_ocurrences_matrix += self.eta * penalty

        lambda_parameter = self.reg_p + self.drop_p / (1 - self.drop_p) * item_counts

        co_ocurrences_matrix[np.diag_indices(self.numbers_of_items)] += lambda_parameter.reshape(-1)

        P = np.linalg.inv(co_ocurrences_matrix)
        diag_P = np.diag(P)

        condition = (1 - lambda_parameter * diag_P) > self.xi
        lagrangian = ((1 - self.xi) / diag_P * lambda_parameter) * condition.astype(float)
        B_DLAE = np.eye(self.numbers_of_items) - P * (lambda_parameter.reshape(-1) + lagrangian)
        item_power_term = np.power(item_counts, -(1 - self.alpha))

        self.W = B_DLAE * (1 / item_power_term).reshape(-1, 1) * item_power_term
        self.W[np.diag_indices(self.numbers_of_items)] = 0

        return self.W


if __name__ == "__main__":
    matrix = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0],
                       [1, 0, 0, 1, 0],
                       [1, 0, 0, 1, 0],
                       [1, 0, 0, 1, 0],
                       [0, 1, 0, 1, 1]])

    config = {
        'reg_p': 20,
        'alpha': 0.2,
        'beta': 0.3,
        'drop_p': 0.5,
        'xi': 0.3
    }

    dan = LAE_DAN(matrix, config)
    weighted_matrix = dan.get_weighted_matrix()

    # Alternatively, you can use EASE_DAN
    dan_ease = EASE_DAN(matrix, config)
    weighted_matrix_ease = dan_ease.get_weighted_matrix()

    # Or RLAE_DAN
    dan_rlae = RLAE_DAN(matrix, config)
    weighted_matrix_rlae = dan_rlae.get_weighted_matrix()

    print("Weighted Matrix:\n", weighted_matrix)
    print("Weighted Matrix EASE:\n", weighted_matrix_ease)
    print("Weighted Matrix RLAE:\n", weighted_matrix_rlae)
