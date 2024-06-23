from chromadb.utils import embedding_functions


COLLECTION_NAME = "collection"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
PERSISTENT_STORAGE = "_vector_db"


EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
