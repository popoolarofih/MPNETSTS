from flask import Flask, request, jsonify, render_template
from mpnet_model import AllMpnetBaseV2
from image_model import ImageEmbedder
from video_model import VideoEmbedder
import torch
import re
import time
from functools import lru_cache
import logging
import requests
from io import BytesIO
from PIL import Image
import os
import tempfile
import networkx as nx
import torch.nn.functional as F

app = Flask(__name__) 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
text_model = AllMpnetBaseV2(device=device)
image_model = ImageEmbedder(device=device)
video_model = VideoEmbedder(device=device)
stats = {
    "text_single_pair_computations": 0, 
    "text_batch_computations": 0, 
    "image_single_pair_computations": 0, 
    "image_batch_computations": 0, 
    "video_single_pair_computations": 0, 
    "video_batch_computations": 0, 
    "image_dedup_computations": 0, 
    "video_dedup_computations": 0, 
    "total_time": 0.0
}

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,!?]', '', text)
    return text

@lru_cache(maxsize=1000)
def cached_text_similarity(sentence1: str, sentence2: str) -> float:
    preprocessed = [preprocess_text(s) for s in (sentence1, sentence2)]
    start_time = time.time()
    similarity = text_model.perform_cosine_similarity_between_2_sentences(preprocessed)
    stats["text_single_pair_computations"] += 1
    stats["total_time"] += time.time() - start_time
    return similarity

@lru_cache(maxsize=500)
def cached_image_embedding(url: str) -> torch.Tensor:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.content
        image = Image.open(BytesIO(content))
        return image_model.encode_image(image)
    except Exception as e:
        raise ValueError(f"Error loading image from {url}: {str(e)}")

@lru_cache(maxsize=200)
def cached_video_embedding(url: str) -> torch.Tensor:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(content)
            tmp.flush()
            emb = video_model.encode_video(tmp.name)
        os.unlink(tmp.name)
        return emb
    except Exception as e:
        raise ValueError(f"Error loading video from {url}: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute_similarity', methods=['POST'])
def compute_similarity():
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
            similarity = cached_text_similarity(preprocessed[0], preprocessed[1])
        else:
            similarity = text_model.perform_cosine_similarity_between_2_sentences(preprocessed)
            stats["text_single_pair_computations"] += 1
            stats["total_time"] += time.time() - start_time
        return jsonify({'similarity': similarity})
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/compute_batch_similarity', methods=['POST'])
def compute_batch_similarity():
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
                similarity = cached_text_similarity(preprocessed[0], preprocessed[1])
            else:
                similarity = text_model.perform_cosine_similarity_between_2_sentences(preprocessed)
            results.append({
                'sentence1': sentence1,
                'sentence2': sentence2,
                'similarity': similarity
            })
        stats["text_batch_computations"] += 1
        stats["total_time"] += time.time() - start_time
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error computing batch similarity: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/compute_image_similarity', methods=['POST'])
def compute_image_similarity():
    try:
        data = request.get_json()
        url1 = data.get('url1', '')
        url2 = data.get('url2', '')
        use_cache = data.get('use_cache', False)
        if not url1 or not url2:
            return jsonify({'error': 'Both image URLs are required'}), 400
        start_time = time.time()
        if use_cache:
            emb1 = cached_image_embedding(url1)
            emb2 = cached_image_embedding(url2)
        else:
            resp1 = requests.get(url1, timeout=10)
            resp1.raise_for_status()
            img1 = Image.open(BytesIO(resp1.content))
            emb1 = image_model.encode_image(img1)
            resp2 = requests.get(url2, timeout=10)
            resp2.raise_for_status()
            img2 = Image.open(BytesIO(resp2.content))
            emb2 = image_model.encode_image(img2)
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        stats["image_single_pair_computations"] += 1
        stats["total_time"] += time.time() - start_time
        return jsonify({'similarity': similarity})
    except Exception as e:
        logger.error(f"Error computing image similarity: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/compute_batch_image_similarity', methods=['POST'])
def compute_batch_image_similarity():
    try:
        data = request.get_json()
        pairs = data.get('pairs', [])
        use_cache = data.get('use_cache', False)
        if not pairs:
            return jsonify({'error': 'At least one pair required'}), 400
        results = []
        start_time = time.time()
        for pair in pairs:
            url1 = pair.get('url1', '')
            url2 = pair.get('url2', '')
            if not url1 or not url2:
                continue
            if use_cache:
                emb1 = cached_image_embedding(url1)
                emb2 = cached_image_embedding(url2)
            else:
                resp1 = requests.get(url1, timeout=10)
                resp1.raise_for_status()
                img1 = Image.open(BytesIO(resp1.content))
                emb1 = image_model.encode_image(img1)
                resp2 = requests.get(url2, timeout=10)
                resp2.raise_for_status()
                img2 = Image.open(BytesIO(resp2.content))
                emb2 = image_model.encode_image(img2)
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            results.append({
                'url1': url1,
                'url2': url2,
                'similarity': similarity
            })
        stats["image_batch_computations"] += 1
        stats["total_time"] += time.time() - start_time
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error computing batch image similarity: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/dedup_images', methods=['POST'])
def dedup_images():
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        use_cache = data.get('use_cache', False)
        threshold = data.get('threshold', 0.9)
        if len(urls) < 2:
            return jsonify({'error': 'At least two image URLs required'}), 400
        start_time = time.time()
        embeddings = []
        for url in urls:
            if use_cache:
                emb = cached_image_embedding(url)
            else:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content))
                emb = image_model.encode_image(img)
            embeddings.append(emb)
        embeddings = torch.stack(embeddings)
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        G = nx.Graph()
        n = len(urls)
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] > threshold:
                    G.add_edge(i, j)
        components = list(nx.connected_components(G))
        kept = [urls[min(comp)] for comp in components]
        removed = len(urls) - len(kept)
        groups = [[urls[i] for i in comp] for comp in components]
        stats["image_dedup_computations"] += 1
        stats["total_time"] += time.time() - start_time
        return jsonify({'unique': kept, 'removed': removed, 'groups': groups})
    except Exception as e:
        logger.error(f"Error in image deduplication: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/compute_video_similarity', methods=['POST'])
def compute_video_similarity():
    try:
        data = request.get_json()
        url1 = data.get('url1', '')
        url2 = data.get('url2', '')
        use_cache = data.get('use_cache', False)
        if not url1 or not url2:
            return jsonify({'error': 'Both video URLs are required'}), 400
        start_time = time.time()
        if use_cache:
            emb1 = cached_video_embedding(url1)
            emb2 = cached_video_embedding(url2)
        else:
            resp1 = requests.get(url1, timeout=30)
            resp1.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp1:
                tmp1.write(resp1.content)
                tmp1.flush()
                emb1 = video_model.encode_video(tmp1.name)
            os.unlink(tmp1.name)
            resp2 = requests.get(url2, timeout=30)
            resp2.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp2:
                tmp2.write(resp2.content)
                tmp2.flush()
                emb2 = video_model.encode_video(tmp2.name)
            os.unlink(tmp2.name)
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        stats["video_single_pair_computations"] += 1
        stats["total_time"] += time.time() - start_time
        return jsonify({'similarity': similarity})
    except Exception as e:
        logger.error(f"Error computing video similarity: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/compute_batch_video_similarity', methods=['POST'])
def compute_batch_video_similarity():
    try:
        data = request.get_json()
        pairs = data.get('pairs', [])
        use_cache = data.get('use_cache', False)
        if not pairs:
            return jsonify({'error': 'At least one pair required'}), 400
        results = []
        start_time = time.time()
        for pair in pairs:
            url1 = pair.get('url1', '')
            url2 = pair.get('url2', '')
            if not url1 or not url2:
                continue
            if use_cache:
                emb1 = cached_video_embedding(url1)
                emb2 = cached_video_embedding(url2)
            else:
                resp1 = requests.get(url1, timeout=30)
                resp1.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp1:
                    tmp1.write(resp1.content)
                    tmp1.flush()
                    emb1 = video_model.encode_video(tmp1.name)
                os.unlink(tmp1.name)
                resp2 = requests.get(url2, timeout=30)
                resp2.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp2:
                    tmp2.write(resp2.content)
                    tmp2.flush()
                    emb2 = video_model.encode_video(tmp2.name)
                os.unlink(tmp2.name)
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            results.append({
                'url1': url1,
                'url2': url2,
                'similarity': similarity
            })
        stats["video_batch_computations"] += 1
        stats["total_time"] += time.time() - start_time
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Error computing batch video similarity: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/dedup_videos', methods=['POST'])
def dedup_videos():
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        use_cache = data.get('use_cache', False)
        threshold = data.get('threshold', 0.9)
        if len(urls) < 2:
            return jsonify({'error': 'At least two video URLs required'}), 400
        start_time = time.time()
        embeddings = []
        for url in urls:
            if use_cache:
                emb = cached_video_embedding(url)
            else:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(resp.content)
                    tmp.flush()
                    emb = video_model.encode_video(tmp.name)
                os.unlink(tmp.name)
            embeddings.append(emb)
        embeddings = torch.stack(embeddings)
        similarities = F.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        G = nx.Graph()
        n = len(urls)
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] > threshold:
                    G.add_edge(i, j)
        components = list(nx.connected_components(G))
        kept = [urls[min(comp)] for comp in components]
        removed = len(urls) - len(kept)
        groups = [[urls[i] for i in comp] for comp in components]
        stats["video_dedup_computations"] += 1
        stats["total_time"] += time.time() - start_time
        return jsonify({'unique': kept, 'removed': removed, 'groups': groups})
    except Exception as e:
        logger.error(f"Error in video deduplication: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    try:
        cached_text_similarity.cache_clear()
        cached_image_embedding.cache_clear()
        cached_video_embedding.cache_clear()
        return jsonify({'message': 'Cache cleared'})
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    return jsonify(stats)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)