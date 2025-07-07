# Amazon Book Recommendation System - Combine + EASE Model

## Overview

This repository implements a sophisticated book recommendation system using the **Combine + EASE** model on the **Amazon Reviews Book** dataset. The system leverages advanced collaborative filtering techniques to provide personalized book recommendations with rich metadata integration.

## Project Structure

### Core Components

- **`recommend.py`** - Main recommendation engine with user profiling and practical examples
- **`Informe.ipynb`** - Comprehensive analysis and model comparisons
- **`utils.py`** - Experiment management and evaluation utilities
- **`loader.py`** - Data loading and preprocessing
- **`combine.py`** - Combine model implementation with EASE normalization

### Data Management

- **`asin_index.pkl`** - Cached metadata index for fast book lookups
- **`metadata/`** - Amazon book metadata dataset
- **`amazon_reviews_books/`** - User reviews dataset

## Features

### 🎯 Personalized Recommendations
- Generate top-k book recommendations for any user
- Efficient metadata integration using cached indexing
- Rich book information including titles, descriptions, and ratings

### 📊 User Profiling
- Display user's reading history and preferences
- Show recent reviews with both book titles and review content
- Interactive user profile creation

### ⚡ Performance Optimized
- Parallel processing for large-scale metadata handling
- Cached ASIN indexing for sub-second lookups
- Efficient batch processing of recommendations

### 🔬 Research & Analysis
- Comprehensive model comparison framework
- Sensitivity analysis for hyperparameter tuning
- Performance evaluation across multiple metrics

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 2. Run Recommendation Example

```bash
# Generate personalized recommendations
uv run python recommend.py
```

This will:
- Load the Amazon Books dataset
- Train the Combine + EASE model
- Create a user profile with reading history
- Generate top-5 personalized recommendations
- Save results to `recommendations_output.txt`

### 3. Explore Analysis

```bash
# Open Jupyter notebook for detailed analysis
jupyter notebook Informe.ipynb
```

## Model Architecture

### Combine + EASE Model
The system uses a sophisticated combination of:
- **Combine Model**: Integrates multiple similarity approaches
- **EASE Normalization**: Enhanced performance through advanced normalization
- **Collaborative Filtering**: User-item interaction modeling

### Configuration
```python
config = {
    'reg_p': 20,      # Regularization parameter
    'alpha': 0.2,     # Alpha weight
    'beta': 0.3,      # Beta weight
    'drop_p': 0.5,    # Dropout probability
    'xi': 0.3,        # Xi parameter
    'eta': 10         # Eta parameter
}
gamma = 0.5          # Sapling parameter
alpha = 0.5          # Combine weight
```

## Dataset

### Amazon Reviews Book Dataset
- **4.4M+ books** with rich metadata
- **User reviews** with ratings and text
- **Book information** including titles, descriptions, categories

### Data Processing
- Efficient loading with Hugging Face datasets
- Automatic ASIN indexing for fast lookups
- Batch processing for large-scale operations

## Usage Examples

### Generate Recommendations
```python
from recommend import main

# Run complete recommendation pipeline
main()
```

### Experiment Management
```python
from utils import run_experiments, sensitivity_analysis
from loader import Loader

# Run experiments with different configurations
loader = Loader(number_of_samples=5000)
results = run_experiments(loader, config, gamma, alpha, k_values=[5, 10, 20])

# Perform sensitivity analysis
sensitivity_analysis(loader)
```

### Custom User Recommendations
```python
# Modify user_id in recommend.py to get recommendations for different users
user_id = 42  # Change this to any user ID
```

## Performance

### Speed Optimizations
- **First run**: ~6-8 minutes (includes index creation)
- **Subsequent runs**: ~10-15 seconds
- **Metadata lookup**: <2 seconds for any book

### Memory Efficiency
- Streaming data processing
- Cached indexing system
- Minimal memory footprint

## Output Format

### User Profile
```
User 0 profile:
  - God Gave Us You - 1578563232
```

### User Reviews
```
User 0 recent reviews:
  - God Gave Us You | "Great gift and value!!" (Rating: 5.0/5, ASIN: 1578563232)
  - God Gave Us You | "A FAVORITE book" (Rating: 5.0/5, ASIN: 1578563232)
```

### Recommendations
```
Top 5 Recommendations:
--------------------------------------------------------------------------------
 1. The Red Sea Rules: 10 God-Given Strategies for Difficult Times
    ASIN: 0529104407
    Score: 0.0830
    Description: About the Author
```

## Research & Development

### Current Focus
- Model performance optimization
- Hyperparameter tuning
- Scalability improvements

### Future Enhancements
- Multi-modal recommendation (text + metadata)
- Real-time recommendation updates
- A/B testing framework
- API development for production deployment

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{amazon_book_recommendations,
  title={Amazon Book Recommendation System - Combine + EASE Model},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/amazon-book-recommendations}
}
```



