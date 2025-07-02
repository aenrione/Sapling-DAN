import numpy as np
from typing import List, Dict, Tuple, Union
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

class RecommendationMetrics:
    """
    A comprehensive class for calculating recommendation system metrics.
    Supports Recall@K, Precision@K, NDCG@K, and other common metrics.
    """
    
    def __init__(self, k_values: List[int] | None = None):
        """
        Initialize the metrics calculator.
        
        Args:
            k_values: List of K values to calculate metrics for. Default: [1, 5, 10, 20]
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]
        self.k_values = k_values
    
    def load_data_from_txt(self, train_file: str, test_file: str, valid_file: str | None = None) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """
        Load data from txt files in the format used in the dan directory.
        
        Args:
            train_file: Path to training data file
            test_file: Path to test data file  
            valid_file: Path to validation data file (optional)
            
        Returns:
            Tuple of (train_data, test_data, valid_data) where each is a list of lists
        """
        def load_file(file_path: str) -> List[List[int]]:
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    # Skip empty lines
                    if line.strip():
                        # Split by space and convert to integers
                        items = [int(x) for x in line.strip().split()]
                        data.append(items)
            return data
        
        train_data = load_file(train_file)
        test_data = load_file(test_file)
        valid_data = load_file(valid_file) if valid_file else None
        
        return train_data, test_data, valid_data
    
    def create_interaction_matrix(self, data: List[List[int]], num_users: int, num_items: int) -> np.ndarray:
        """
        Create a user-item interaction matrix from list format.
        
        Args:
            data: List of lists where each inner list contains item IDs for a user
            num_users: Total number of users
            num_items: Total number of items
            
        Returns:
            Binary interaction matrix of shape (num_users, num_items)
        """
        matrix = np.zeros((num_users, num_items), dtype=int)
        for user_id, items in enumerate(data):
            for item_id in items:
                if item_id < num_items:  # Ensure item_id is within bounds
                    matrix[user_id, item_id] = 1
        return matrix
    
    def calculate_metrics(self, 
                         train_data: List[List[int]], 
                         test_data: List[List[int]], 
                         recommendation_matrix: np.ndarray,
                         valid_data: List[List[int]] | None = None) -> Dict[str, Dict[int, float]]:
        """
        Calculate all metrics for the recommendation system.
        
        Args:
            train_data: Training data in list format
            test_data: Test data in list format
            recommendation_matrix: Recommendation scores matrix from pipeline
            valid_data: Validation data in list format (optional)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        num_users, num_items = recommendation_matrix.shape
        
        # Create interaction matrices
        train_matrix = self.create_interaction_matrix(train_data, num_users, num_items)
        test_matrix = self.create_interaction_matrix(test_data, num_users, num_items)
        
        # Calculate metrics
        metrics = {}
        
        # Calculate Recall@K, Precision@K, NDCG@K
        for k in self.k_values:
            recall_k = self.calculate_recall_at_k(train_matrix, test_matrix, recommendation_matrix, k)
            precision_k = self.calculate_precision_at_k(train_matrix, test_matrix, recommendation_matrix, k)
            ndcg_k = self.calculate_ndcg_at_k(train_matrix, test_matrix, recommendation_matrix, k)
            
            metrics[f'recall@{k}'] = recall_k
            metrics[f'precision@{k}'] = precision_k
            metrics[f'ndcg@{k}'] = ndcg_k
        
        # Calculate additional metrics
        metrics['mrr'] = self.calculate_mrr(train_matrix, test_matrix, recommendation_matrix)
        metrics['hit_rate'] = self.calculate_hit_rate(train_matrix, test_matrix, recommendation_matrix, k=10)
        
        return metrics
    
    def calculate_recall_at_k(self, 
                            train_matrix: np.ndarray, 
                            test_matrix: np.ndarray, 
                            recommendation_matrix: np.ndarray, 
                            k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            train_matrix: Training interaction matrix
            test_matrix: Test interaction matrix
            recommendation_matrix: Recommendation scores matrix
            k: Number of top recommendations to consider
            
        Returns:
            Recall@K score
        """
        recalls = []
        
        for user_id in range(len(train_matrix)):
            # Get user's test items
            test_items = np.where(test_matrix[user_id] == 1)[0]
            if len(test_items) == 0:
                continue
            
            # Get user's training items to exclude from recommendations
            train_items = np.where(train_matrix[user_id] == 1)[0]
            
            # Get recommendation scores for this user
            user_scores = recommendation_matrix[user_id].copy()
            
            # Set training items to -inf so they won't be recommended
            user_scores[train_items] = -np.inf
            
            # Get top-k recommended items
            top_k_items = np.argsort(user_scores)[::-1][:k]
            
            # Calculate recall
            hits = len(set(top_k_items) & set(test_items))
            recall = hits / len(test_items) if len(test_items) > 0 else 0
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def calculate_precision_at_k(self, 
                               train_matrix: np.ndarray, 
                               test_matrix: np.ndarray, 
                               recommendation_matrix: np.ndarray, 
                               k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            train_matrix: Training interaction matrix
            test_matrix: Test interaction matrix
            recommendation_matrix: Recommendation scores matrix
            k: Number of top recommendations to consider
            
        Returns:
            Precision@K score
        """
        precisions = []
        
        for user_id in range(len(train_matrix)):
            # Get user's test items
            test_items = np.where(test_matrix[user_id] == 1)[0]
            if len(test_items) == 0:
                continue
            
            # Get user's training items to exclude from recommendations
            train_items = np.where(train_matrix[user_id] == 1)[0]
            
            # Get recommendation scores for this user
            user_scores = recommendation_matrix[user_id].copy()
            
            # Set training items to -inf so they won't be recommended
            user_scores[train_items] = -np.inf
            
            # Get top-k recommended items
            top_k_items = np.argsort(user_scores)[::-1][:k]
            
            # Calculate precision
            hits = len(set(top_k_items) & set(test_items))
            precision = hits / k if k > 0 else 0
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def calculate_ndcg_at_k(self, 
                          train_matrix: np.ndarray, 
                          test_matrix: np.ndarray, 
                          recommendation_matrix: np.ndarray, 
                          k: int) -> float:
        """
        Calculate NDCG@K.
        
        Args:
            train_matrix: Training interaction matrix
            test_matrix: Test interaction matrix
            recommendation_matrix: Recommendation scores matrix
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@K score
        """
        ndcgs = []
        
        for user_id in range(len(train_matrix)):
            # Get user's test items
            test_items = np.where(test_matrix[user_id] == 1)[0]
            if len(test_items) == 0:
                continue
            
            # Get user's training items to exclude from recommendations
            train_items = np.where(train_matrix[user_id] == 1)[0]
            
            # Get recommendation scores for this user
            user_scores = recommendation_matrix[user_id].copy()
            
            # Set training items to -inf so they won't be recommended
            user_scores[train_items] = -np.inf
            
            # Get top-k recommended items
            top_k_items = np.argsort(user_scores)[::-1][:k]
            
            # Create relevance vector (1 for relevant items, 0 for irrelevant)
            relevance = np.zeros(k)
            for i, item in enumerate(top_k_items):
                if item in test_items:
                    relevance[i] = 1
            
            # Calculate DCG
            dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))
            
            # Calculate IDCG (ideal DCG)
            ideal_relevance = np.ones(min(k, len(test_items)))
            idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def calculate_mrr(self, 
                     train_matrix: np.ndarray, 
                     test_matrix: np.ndarray, 
                     recommendation_matrix: np.ndarray) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            train_matrix: Training interaction matrix
            test_matrix: Test interaction matrix
            recommendation_matrix: Recommendation scores matrix
            
        Returns:
            MRR score
        """
        mrr_scores = []
        
        for user_id in range(len(train_matrix)):
            # Get user's test items
            test_items = np.where(test_matrix[user_id] == 1)[0]
            if len(test_items) == 0:
                continue
            
            # Get user's training items to exclude from recommendations
            train_items = np.where(train_matrix[user_id] == 1)[0]
            
            # Get recommendation scores for this user
            user_scores = recommendation_matrix[user_id].copy()
            
            # Set training items to -inf so they won't be recommended
            user_scores[train_items] = -np.inf
            
            # Get ranked items
            ranked_items = np.argsort(user_scores)[::-1]
            
            # Find the rank of the first relevant item
            for rank, item in enumerate(ranked_items, 1):
                if item in test_items:
                    mrr_scores.append(1.0 / rank)
                    break
        
        return np.mean(mrr_scores) if mrr_scores else 0.0
    
    def calculate_hit_rate(self, 
                          train_matrix: np.ndarray, 
                          test_matrix: np.ndarray, 
                          recommendation_matrix: np.ndarray, 
                          k: int = 10) -> float:
        """
        Calculate Hit Rate@K (whether at least one relevant item is in top-K).
        
        Args:
            train_matrix: Training interaction matrix
            test_matrix: Test interaction matrix
            recommendation_matrix: Recommendation scores matrix
            k: Number of top recommendations to consider
            
        Returns:
            Hit Rate@K score
        """
        hits = 0
        total_users = 0
        
        for user_id in range(len(train_matrix)):
            # Get user's test items
            test_items = np.where(test_matrix[user_id] == 1)[0]
            if len(test_items) == 0:
                continue
            
            total_users += 1
            
            # Get user's training items to exclude from recommendations
            train_items = np.where(train_matrix[user_id] == 1)[0]
            
            # Get recommendation scores for this user
            user_scores = recommendation_matrix[user_id].copy()
            
            # Set training items to -inf so they won't be recommended
            user_scores[train_items] = -np.inf
            
            # Get top-k recommended items
            top_k_items = np.argsort(user_scores)[::-1][:k]
            
            # Check if any relevant item is in top-k
            if len(set(top_k_items) & set(test_items)) > 0:
                hits += 1
        
        return hits / total_users if total_users > 0 else 0.0
    
    def print_metrics(self, metrics: Dict[str, Dict[int, float]]):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary containing calculated metrics
        """
        print("=" * 50)
        print("RECOMMENDATION SYSTEM METRICS")
        print("=" * 50)
        
        # Print K-based metrics
        for k in self.k_values:
            print(f"\nMetrics @ K={k}:")
            print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")
            print(f"  Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            print(f"  NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")
        
        # Print other metrics
        print(f"\nOther Metrics:")
        print(f"  MRR: {metrics['mrr']:.4f}")
        print(f"  Hit Rate@10: {metrics['hit_rate']:.4f}")
        print("=" * 50)



