import numpy as np
import warnings
import time
import pandas as pd
import pickle
import os
from multiprocessing import Pool, cpu_count
from loader import Loader
from combine import Combine
from utils import train_test_split
from datasets import load_dataset, load_from_disk
warnings.filterwarnings("ignore")

def create_asin_index_chunk(chunk_data):
    """Create ASIN index for a chunk of metadata"""
    chunk_index = {}
    for i, item in enumerate(chunk_data):
        if not isinstance(item, dict):
            continue
        asin = item.get('parent_asin')
        if asin:
            chunk_index[asin] = i
    return chunk_index

def create_asin_index_sequential(metadata_data, debug=True):
    """Create ASIN index using sequential processing (more reliable)"""
    if debug:
        print("Creating ASIN index using sequential processing...")
        start_time = time.time()
    
    asin_index = {}
    total_items = len(metadata_data)
    
    for i, item in enumerate(metadata_data):
        if not isinstance(item, dict):
            continue
        asin = item.get('parent_asin')
        if asin:
            asin_index[asin] = i
        
        # Progress update every 100k items
        if debug and i % 100000 == 0 and i > 0:
            progress = (i / total_items) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (total_items - i) if i > 0 else 0
            print(f"Index progress: {progress:.1f}% ({i}/{total_items}) - ETA: {eta:.1f}s")
    
    if debug:
        elapsed = time.time() - start_time
        print(f"Created index with {len(asin_index)} unique ASINs in {elapsed:.2f}s")
    
    return asin_index

def load_or_create_asin_index():
    """Load existing ASIN index or create a new one"""
    index_file = "asin_index.pkl"
    
    if os.path.exists(index_file):
        print(f"Loading existing ASIN index...")
        with open(index_file, 'rb') as f:
            return pickle.load(f)
    
    print("Creating new ASIN index...")
    metadata = load_from_disk("metadata")
    metadata_data = metadata["full"]
    
    asin_index = {}
    total_items = len(metadata_data)
    
    for i, item in enumerate(metadata_data):
        if not isinstance(item, dict):
            continue
        asin = item.get('parent_asin')
        if asin:
            asin_index[asin] = i
        
        if i % 1000000 == 0 and i > 0:
            print(f"Index progress: {i/1000000:.1f}M/{total_items/1000000:.1f}M")
    
    print(f"Created index with {len(asin_index)} ASINs")
    
    with open(index_file, 'wb') as f:
        pickle.dump(asin_index, f)
    print("Index saved!")
    
    return asin_index

def load_book_details_from_metadata(loader, debug=True):
    """Optimized loading of book details using Hugging Face datasets for efficient processing."""
    if debug:
        print("Loading book details from metadata dataset...")
        start_time = time.time()

    book_details = {}

    try:
        # Load metadata from disk
        metadata = load_from_disk("metadata")
        metadata_data = metadata["full"]

        if debug:
            print(f"Metadata dataset loaded: {len(metadata_data)} items")
            print("Processing metadata in batches...")

        # Determine the correct ASIN field by checking the first few items
        sample_item = metadata_data[0]
        asin_field = None
        for field in ['parent_asin', 'asin', 'ASIN', 'id']:
            if field in sample_item:
                asin_field = field
                break

        if asin_field is None:
            raise ValueError("No ASIN field found in metadata.")

        if debug:
            print(f"Using ASIN field: {asin_field}")

        # Process data in batches using dataset methods
        batch_size = 10000
        total_items = len(metadata_data)
        processed_count = 0

        for i in range(0, total_items, batch_size):
            batch_end = min(i + batch_size, total_items)
            
            if debug and i % 50000 == 0:
                progress = (i / total_items) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1)) * (total_items - i) if i > 0 else 0
                print(f"Progress: {progress:.1f}% ({i}/{total_items}) - ETA: {eta:.1f}s")

            # Process batch
            for j in range(i, batch_end):
                item = metadata_data[j]
                
                # Get ASIN
                asin = item.get(asin_field)
                if not asin:
                    continue
                
                asin = str(asin)
                
                # Extract title and description
                title = item.get('title', 'Unknown Title')
                description = item.get('description', 'No description available')
                
                # Handle cases where fields might be lists
                if isinstance(title, list):
                    title = title[0] if title else 'Unknown Title'
                if isinstance(description, list):
                    description = description[0] if description else 'No description available'
                
                # Ensure strings
                title = str(title) if title else 'Unknown Title'
                description = str(description) if description else 'No description available'
                
                # Truncate description for efficiency
                if len(description) > 200:
                    description = description[:200] + '...'
                
                book_details[asin] = {
                    'title': title,
                    'description': description
                }
                processed_count += 1

    except Exception as e:
        print(f"Warning: Could not load book details from metadata: {e}")
        if debug:
            print("Falling back to reviews dataset...")
        return load_book_details_from_reviews_fallback(loader, debug)

    if debug:
        elapsed = time.time() - start_time
        print(f"Loaded details for {len(book_details)} books from metadata in {elapsed:.2f}s")
        print(f"Processed {processed_count} items total")

    return book_details

def load_book_details_from_reviews_fallback(loader, debug=True):
    """Fallback function to load book details from reviews dataset"""
    if debug:
        print("Loading book details from reviews dataset (fallback)...")
        start_time = time.time()
    
    book_details = {}
    
    try:
        reviews = load_from_disk(loader.dataset_name)
        reviews_data = reviews[loader.subset_name]
        
        if debug:
            print(f"Reviews dataset loaded: {len(reviews_data)} items")
        
        # Process only the first number_of_samples
        for i, review in enumerate(reviews_data):
            if i >= loader.number_of_samples:
                break
                
            # Handle the review structure properly
            if isinstance(review, dict):
                asin = review.get(loader.asin_column, '')
                title = review.get('title', 'Unknown Title')
                description = review.get('description', 'No description available')
            else:
                continue
            
            if not asin:
                continue
                
            asin = str(asin)
            
            # Handle cases where fields might be lists
            if isinstance(title, list):
                title = title[0] if title else 'Unknown Title'
            if isinstance(description, list):
                description = description[0] if description else 'No description available'
            
            # Ensure strings
            title = str(title) if title else 'Unknown Title'
            description = str(description) if description else 'No description available'
            
            # Only add if not already present (avoid duplicates)
            if asin not in book_details:
                book_details[asin] = {
                    'title': title,
                    'description': description[:200] + '...' if len(description) > 200 else description
                }
                
    except Exception as e:
        print(f"Warning: Could not load book details from reviews: {e}")
    
    if debug:
        elapsed = time.time() - start_time
        print(f"Loaded details for {len(book_details)} books from reviews in {elapsed:.2f}s")
    
    return book_details

def load_user_reviews(loader, user_id, id2item, num_reviews=3, debug=True):
    """Load sample reviews for a user efficiently"""
    if debug:
        print(f"Loading reviews for user {user_id}...")
        start_time = time.time()
    
    user_reviews = []
    
    try:
        reviews = load_from_disk(loader.dataset_name)
        reviews_data = reviews[loader.subset_name]
        
        # Get user's items
        user_items = []
        with open(loader.user_positive_reviews_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) == user_id:
                    user_items = [int(item_id) for item_id in parts[1:]]
                    break
        
        # Find reviews for these items
        item_asins = [id2item.get(item_id, '') for item_id in user_items]
        asin_set = set(item_asins)
        
        # Process reviews directly
        for review in reviews_data:
            if len(user_reviews) >= num_reviews:
                break
                
            # Handle the review structure properly
            if isinstance(review, dict):
                asin = review.get(loader.asin_column, '')
                rating = review.get(loader.reviews_column, 0)
                title = review.get('title', 'Unknown Title')
            else:
                continue
            
            if asin in asin_set:
                if isinstance(title, list):
                    title = title[0] if title else 'Unknown Title'
                
                user_reviews.append({
                    'asin': str(asin),
                    'title': str(title),
                    'rating': rating
                })
                    
    except Exception as e:
        print(f"Warning: Could not load user reviews: {e}")
    
    if debug:
        elapsed = time.time() - start_time
        print(f"Loaded {len(user_reviews)} reviews for user {user_id} in {elapsed:.2f}s")
    
    return user_reviews

def load_item_mappings(loader, debug=True):
    """Load item ID mappings to get original ASINs"""
    if debug:
        print("Loading item mappings...")
        start_time = time.time()
    
    item2id = {}
    id2item = {}
    
    with open(loader.item_list_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                asin, item_id = parts[0], int(parts[1])
                item2id[asin] = item_id
                id2item[item_id] = asin
    
    if debug:
        elapsed = time.time() - start_time
        print(f"Loaded {len(item2id)} item mappings in {elapsed:.2f}s")
    
    return item2id, id2item

def create_user_profile(user_id, user_item_matrix, id2item, book_details, user_reviews, num_items=5, debug=True):
    """Create a user profile with some liked items and reviews"""
    if debug:
        print(f"Creating profile for user {user_id}...")
        start_time = time.time()
    
    # Get items that this user has interacted with
    user_items = np.where(user_item_matrix[user_id] == 1)[0]
    
    if len(user_items) == 0:
        if debug:
            print(f"User {user_id} has no interactions. Creating a random profile...")
        # Create a random profile for demonstration
        all_items = list(id2item.keys())
        user_items = np.random.choice(all_items, size=min(num_items, len(all_items)), replace=False)
    
    if debug:
        print(f"User {user_id} profile:")
        for item_id in user_items[:num_items]:
            asin = id2item.get(item_id, f"Item_{item_id}")
            book_info = book_details.get(asin, {})
            title = book_info.get('title', f"Book {asin}")
            print(f"  - {title} (ASIN: {asin})")
    
    # Show user reviews
    if user_reviews:
        if debug:
            print(f"\nUser {user_id} recent reviews:")
            for review in user_reviews:
                print(f"  - {review['title']} (Rating: {review['rating']}/5, ASIN: {review['asin']})")
    
    if debug:
        elapsed = time.time() - start_time
        print(f"Profile created in {elapsed:.2f}s")
    
    return user_items

def get_recommendations(user_id, user_item_matrix, model, id2item, k=5, debug=True):
    """Get top-k recommendations for a user"""
    if debug:
        print(f"Generating recommendations for user {user_id}...")
        start_time = time.time()
    
    # Get user's prediction scores
    user_scores = model[user_id]
    
    # Get items the user hasn't interacted with
    user_interacted = np.where(user_item_matrix[user_id] == 1)[0]
    candidate_items = np.setdiff1d(np.arange(len(user_scores)), user_interacted)
    
    # Get scores for candidate items
    candidate_scores = user_scores[candidate_items]
    
    # Sort by score (descending) and get top-k
    top_indices = np.argsort(candidate_scores)[::-1][:k]
    top_items = candidate_items[top_indices]
    top_scores = candidate_scores[top_indices]
    
    if debug:
        elapsed = time.time() - start_time
        print(f"Generated {k} recommendations in {elapsed:.2f}s")
    
    return top_items, top_scores

def format_recommendation_output(recommended_items, scores, id2item, book_details, k_recommendations):
    """Format recommendations for output"""
    output_lines = []
    
    output_lines.append(f"Top {k_recommendations} Recommendations:")
    output_lines.append("-" * 80)
    
    for i, (item_id, score) in enumerate(zip(recommended_items, scores), 1):
        asin = id2item.get(item_id, f"Item_{item_id}")
        book_info = book_details.get(asin, {})
        title = book_info.get('title', f"Book {asin}")
        description = book_info.get('description', 'No description available')
        
        line = f"{i:2d}. {title}"
        line += f"\n    ASIN: {asin}"
        line += f"\n    Score: {score:.4f}"
        line += f"\n    Description: {description}"
        line += "\n"
        
        output_lines.append(line)
    
    return output_lines

def load_book_details_for_recommendations(recommended_asins, debug=True):
    """Load book details only for the recommended ASINs from metadata dataset with cached indexing."""
    if debug:
        print(f"Loading book details for {len(recommended_asins)} recommended books from metadata...")
        start_time = time.time()

    book_details = {}

    try:
        # Load or create ASIN index
        asin_index = load_or_create_asin_index()
        
        # Load metadata from disk
        metadata = load_from_disk("metadata")
        metadata_data = metadata["full"]

        if debug:
            print("Looking up recommended books in index...")

        # Search through the index for the specific ASINs
        found_count = 0
        for asin in recommended_asins:
            if asin in asin_index:
                item = metadata_data[asin_index[asin]]
                
                # Extract title and description
                title = item.get('title', 'Unknown Title')
                description = item.get('description', 'No description available')
                
                # Handle cases where fields might be lists
                if isinstance(title, list):
                    title = title[0] if title else 'Unknown Title'
                if isinstance(description, list):
                    description = description[0] if description else 'No description available'
                
                # Ensure strings
                title = str(title) if title else 'Unknown Title'
                description = str(description) if description else 'No description available'
                
                # Truncate description for efficiency
                if len(description) > 200:
                    description = description[:200] + '...'
                
                book_details[asin] = {
                    'title': title,
                    'description': description
                }
                found_count += 1
                
                if debug:
                    print(f"Found book: {title} (ASIN: {asin})")
                
                # Stop if we found all recommended books
                if found_count >= len(recommended_asins):
                    break

    except Exception as e:
        print(f"Warning: Could not load book details from metadata: {e}")

    if debug:
        elapsed = time.time() - start_time
        print(f"Found details for {len(book_details)} recommended books in {elapsed:.2f}s")

    return book_details

def get_book_details_for_asins(asins):
    """Get book details for specific ASINs from metadata"""
    asin_index = load_or_create_asin_index()
    metadata = load_from_disk("metadata")
    metadata_data = metadata["full"]
    
    book_details = {}
    for asin in asins:
        if asin in asin_index:
            item = metadata_data[asin_index[asin]]
            
            title = item.get('title', 'Unknown Title')
            description = item.get('description', 'No description available')
            
            if isinstance(title, list):
                title = title[0] if title else 'Unknown Title'
            if isinstance(description, list):
                description = description[0] if description else 'No description available'
            
            title = str(title) if title else 'Unknown Title'
            description = str(description) if description else 'No description available'
            
            if len(description) > 200:
                description = description[:200] + '...'
            
            book_details[asin] = {
                'title': title,
                'description': description
            }
    
    return book_details

def main():
    # Configuration
    config = {
        'reg_p': 20, 'alpha': 0.2, 'beta': 0.3, 'drop_p': 0.5, 'xi': 0.3, 'eta': 10
    }
    gamma = 0.5
    alpha = 0.5
    k_recommendations = 5
    user_id = 0
    
    print("=== Amazon Book Recommendation System ===")
    print("Using Combine + EASE Model")
    print()
    
    # Load data and train model
    print("Loading dataset and training model...")
    start_time = time.time()
    
    loader = Loader(number_of_samples=5000)
    loader.create_user_positive_reviews()
    full_matrix = loader.create_user_item_matrix()
    
    # Load item mappings
    item2id, id2item = {}, {}
    with open(loader.item_list_path, "r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                asin, item_id = parts[0], int(parts[1])
                item2id[asin] = item_id
                id2item[item_id] = asin
    
    # Train model
    combine_model = Combine(full_matrix, config, gamma, alpha)
    predictions = combine_model.run("EASE")
    
    print(f"Model trained in {time.time() - start_time:.1f}s")
    print()
    
    # Get recommendations
    print(f"Generating recommendations for User {user_id}...")
    recommended_items, scores = get_recommendations(user_id, full_matrix, predictions, id2item, k_recommendations)
    recommended_asins = [id2item.get(item_id, f"Item_{item_id}") for item_id in recommended_items]
    
    # Get book details
    book_details = get_book_details_for_asins(recommended_asins)
    
    # Get user profile and reviews
    user_items = np.where(full_matrix[user_id] == 1)[0]
    if len(user_items) == 0:
        all_items = list(id2item.keys())
        user_items = np.random.choice(all_items, size=min(5, len(all_items)), replace=False)
    
    user_asins = [id2item.get(item_id, f"Item_{item_id}") for item_id in user_items[:5]]
    user_book_details = get_book_details_for_asins(user_asins)
    
    user_reviews = load_user_reviews(loader, user_id, id2item)
    review_asins = [review['asin'] for review in user_reviews]
    review_book_details = get_book_details_for_asins(review_asins)
    
    # Generate output
    output_lines = []
    output_lines.append("=== Amazon Book Recommendation System ===")
    output_lines.append("Using Combine + EASE Model")
    output_lines.append("")
    output_lines.append(f"Dataset: {loader.dataset_name} - {full_matrix.shape[0]} users, {full_matrix.shape[1]} items")
    output_lines.append(f"User: {user_id}")
    output_lines.append("")
    
    # User profile
    output_lines.append("User Profile:")
    for item_id in user_items[:5]:
        asin = id2item.get(item_id, f"Item_{item_id}")
        book_info = user_book_details.get(asin, {})
        title = book_info.get('title', f"Book {asin}")
        output_lines.append(f"  - {title} ({asin})")
    output_lines.append("")
    
    # User reviews
    if user_reviews:
        output_lines.append("Recent Reviews:")
        for review in user_reviews:
            asin = review['asin']
            book_info = review_book_details.get(asin, {})
            book_title = book_info.get('title', f"Book {asin}")
            review_title = review.get('title', 'No Review Title')
            output_lines.append(f"  - {book_title} | \"{review_title}\" (Rating: {review['rating']}/5)")
        output_lines.append("")
    
    # Recommendations
    output_lines.append("Top 5 Recommendations:")
    output_lines.append("-" * 50)
    for i, (item_id, score) in enumerate(zip(recommended_items, scores), 1):
        asin = id2item.get(item_id, f"Item_{item_id}")
        book_info = book_details.get(asin, {})
        title = book_info.get('title', f"Book {asin}")
        description = book_info.get('description', 'No description available')
        
        output_lines.append(f"{i}. {title}")
        output_lines.append(f"   Score: {score:.4f}")
        output_lines.append(f"   Description: {description}")
        output_lines.append("")
    
    # Model info
    output_lines.append("Model: Combine + EASE")
    output_lines.append(f"Configuration: {config}")
    output_lines.append(f"Total time: {time.time() - start_time:.1f}s")
    
    # Print and save in one operation
    output_text = "\n".join(output_lines)
    print(output_text)
    
    with open("recommendations_output.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    
    print(f"\nResults saved to: recommendations_output.txt")

if __name__ == "__main__":
    main()
