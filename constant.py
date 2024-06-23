from chromadb.utils import embedding_functions
import torch

COLLECTION_NAME = "collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSISTENT_STORAGE = "_vector_db"

device = "cpu"

EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL, device=device)

