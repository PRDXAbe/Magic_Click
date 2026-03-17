"""
Face Embedding Database Configuration
"""
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
DB_DIR = os.path.join(DATA_DIR, 'db')

# ML App directories
CAPTURED_VIDEOS_DIR = os.path.join(BASE_DIR, 'captured_videos')
CAPTURED_SHOTS_DIR = os.path.join(BASE_DIR, 'captured_shots')

# Create directories if they don't exist
for dir_path in [DATA_DIR, IMAGES_DIR, DB_DIR, CAPTURED_VIDEOS_DIR, CAPTURED_SHOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Database paths
SQLITE_PATH = os.path.join(DB_DIR, 'metadata.db')
FAISS_INDEX_PATH = os.path.join(DB_DIR, 'faiss.index')
CAMERAS_CONFIG_PATH = os.path.join(DATA_DIR, 'cameras.json')

# Database paths
SQLITE_PATH = os.path.join(DB_DIR, 'metadata.db')
FAISS_INDEX_PATH = os.path.join(DB_DIR, 'faiss.index')

# Face matching settings
SIMILARITY_THRESHOLD = 0.6  # Minimum similarity to consider a match
EMBEDDING_DIM = 512  # InsightFace embedding dimension

# Server settings
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True
