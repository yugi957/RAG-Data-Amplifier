import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import random

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "fast_docs"
BATCH_SIZE = 64

# Initialize Chroma client and collection
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection = client.get_collection(
    name=COLLECTION_NAME,
)

# Function to get a random sample of n results
def get_random_sample(collection, n_results):
    existing_count = collection.count()
    all_ids = collection.get(include=[])['ids']
    # print(all_ids)
    
    if n_results > existing_count:
        n_results = existing_count  # Limit to the number of existing entries

    random_ids = random.sample(all_ids, n_results)
    random_documents = collection.get(ids=random_ids, include=['embeddings', 'documents'])
    # random_documents = collection.get(ids=random_ids)

    return random_documents

query_results = collection.query(
    query_texts=["Find me some delicious food!"],
    n_results=1,
)

# print(query_results.keys())
# print(dict_keys(['ids', 'distances', 'metadatas', 'embeddings', 'documents']))

# print(query_results["documents"])

existing_count = collection.count()
# print(existing_count)

# Retrieve a random sample of n results
n_results = 5  # Adjust the number of results as needed
random_sample = get_random_sample(collection, n_results)

print(f"Random sample of {n_results} documents:")
print(random_sample["embeddings"])
# for doc in random_sample["documents"]:
    # print(doc)

existing_count = collection.count()
# print(f"Total documents in collection: {existing_count}")

