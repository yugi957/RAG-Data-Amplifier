import random
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.utils import embedding_functions
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "vis_docs"
NUM_CLUSTERS = 5  # Adjust the number of clusters as needed
TSNE_PERPLEXITY = 30
TSNE_N_COMPONENTS = 2

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)


def generate_embeddings(df: pd.DataFrame):
    texts = df['text'].tolist()
    embeddings = embedding_func(texts)
    embeddings = np.array(embeddings)
    return embeddings

def cluster_and_visualize(embeddings):
    embeddings_array = np.array(embeddings)
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42)
    kmeans.fit(embeddings_array)
    labels = kmeans.labels_
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=TSNE_N_COMPONENTS, perplexity=TSNE_PERPLEXITY, random_state=42)
    tsne_results = tsne.fit_transform(embeddings_array)
    # Create a DataFrame for visualization
    df = pd.DataFrame(tsne_results, columns=['tsne_1', 'tsne_2'])
    df['cluster'] = labels
    # Plot t-SNE results
    plt.figure(figsize=(10, 7))
    for cluster in range(NUM_CLUSTERS):
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['tsne_1'], cluster_data['tsne_2'], label=f'Cluster {cluster}', s=10)

    plt.title('t-SNE visualization of clustered embeddings')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend()
    plt.show()