from flask import Flask, request, jsonify, render_template
from mpnet_model import AllMpnetBaseV2
from typing import List, Tuple, Dict
import torch
import re
import time
from functools import lru_cache
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the MPNet model
model = AllMpnetBaseV2(device="cuda" if torch.cuda.is_available() else "cpu")
stats = {"single_pair_computations": 0, "batch_computations": 0, "total_time": 0.0}

def preprocess_text(text: str) -> str:
    """Preprocess text by normalizing whitespace and removing special characters."""
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    text = re.sub(r'[^\w\s.,!?]', '', text)   # Remove special characters except punctuation
    return text

@lru_cache(maxsize=1000)
def cached_similarity(sentence1: str, sentence2: str) -> float:
    """Compute cached similarity for a pair of sentences."""
    preprocessed = [preprocess_text(s) for s in (sentence1, sentence2)]
    start_time = time.time()
    similarity = model.perform_cosine_similarity_between_2_sentences(preprocessed)
    stats["single_pair_computations"] += 1
    stats["total_time"] += time.time() - start_time
    return similarity

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/compute_similarity', methods=['POST'])
def compute_similarity():
    """Compute similarity for a single pair of sentences."""
    try:
        data = request.get_json()
        sentence1 = data.get('sentence1', '')
        sentence2 = data.get('sentence2', '')
        use_cache = data.get('use_cache', False)

        if not sentence1 or not sentence2:
            return jsonify({'error': 'Both sentences are required'}), 400

        preprocessed = [preprocess_text(sentence1), preprocess_text(sentence2)]
        
        start_time = time.time()
        if use_cache:
            similarity = cached_similarity(preprocessed[0], preprocessed[1])
        else:
            similarity = model.perform_cosine_similarity_between_2_sentences(preprocessed)
            stats["single_pair_computations"] += 1
            stats["total_time"] += time.time() - start_time

        return jsonify({'similarity': similarity})
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/compute_batch_similarity', methods=['POST'])
def compute_batch_similarity():
    """Compute similarity for a batch of sentence pairs."""
    try:
        data = request.get_json()
        pairs = data.get('pairs', [])
        use_cache = data.get('use_cache', False)

        if not pairs or len(pairs) < 1:
            return jsonify({'error': 'At least one pair of sentences is required'}), 400

        results = []
        start_time = time.time()
        for pair in pairs:
            sentence1, sentence2 = pair.get('sentence1', ''), pair.get('sentence2', '')
            if not sentence1 or not sentence2:
                continue
            preprocessed = [preprocess_text(sentence1), preprocess_text(sentence2)]
            if use_cache:
                similarity = cached_similarity(preprocessed[0], preprocessed[1])
            else:
                similarity = model.perform_cosine_similarity_between_2_sentences(preprocessed)
            results.append({
                'sentence1': sentence1,
                'sentence2': sentence2,
                'similarity': similarity
            })
        stats["batch_computations"] += 1
        stats["total_time"] += time.time() - start_time

        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error computing batch similarity: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the LRU cache."""
    try:
        cached_similarity.cache_clear()
        return jsonify({'message': 'Cache cleared'})
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Return computation statistics."""
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)