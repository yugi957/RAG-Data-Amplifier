from pprint import pprint
from constant import EMBEDDING_FUNCTION, COLLECTION_NAME, PERSISTENT_STORAGE
import chromadb
import pandas as pd
import os
import random
import json
from sklearn.manifold import TSNE
import numpy as np
from chromadb.config import Settings
import matplotlib.pyplot as plt

settings = Settings(
    allow_reset=True,
)


def store_dataframe(df: pd.DataFrame):
    chroma_client = chromadb.PersistentClient(
        path=PERSISTENT_STORAGE, settings=settings)
    chroma_client.reset()
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)

    texts = df['text'].tolist()
    ids = [str(i) for i in range(0, len(texts))]
    df.drop(columns=['text'], inplace=True)
    metadatas = df.to_dict(orient='records')

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )

    metadata_column_types = dict(df.dtypes)
    metadata_type_and_classes = {}
    for column_name, column_type in metadata_column_types.items():
        if column_type == 'object':
            uniques_values = df[column_name].unique()
            # if len(uniques_values) < 30:
            metadata_type_and_classes[column_name] = (
                str(column_type), list(uniques_values))
        else:
            metadata_type_and_classes[column_name] = (str(column_type), None)

    with open(f"{PERSISTENT_STORAGE}/metadata_type_and_classes.json", "w") as f:
        json.dump(metadata_type_and_classes, f)

    return metadata_type_and_classes


def get_metadata_type_and_classes():
    with open(f"{PERSISTENT_STORAGE}/metadata_type_and_classes.json", "r") as f:
        return json.load(f)


# def query(text : str):
#     chroma_client = chromadb.PersistentClient(path=PERSISTENT_STORAGE, settings=settings)
#     collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)

#     query_result = collection.query(
#         query_texts=[text],
#         n_results=5
#     )
#     return query_result


def query_all(filter):
    chroma_client = chromadb.PersistentClient(
        path=PERSISTENT_STORAGE, settings=settings)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)

    query_result = collection.get(
        where=filter,
        include=["documents", "metadatas", "embeddings"]
    )

    return query_result


def query_random_sample(filter, n_results=5):
    chroma_client = chromadb.PersistentClient(
        path=PERSISTENT_STORAGE, settings=settings)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)

    all = query_all(filter)

    documents = all["documents"]
    ids = all["ids"]
    metadatas = all["metadatas"]

    random_sample = random.sample(
        list(zip(documents, ids, metadatas)), min(n_results, len(documents)))

    return {
        "documents": [doc for doc, _, _ in random_sample],
        "ids": [id for _, id, _ in random_sample],
        "metadatas": [metadata for _, _, metadata in random_sample]
    }


def query_semantic(text: str, filter, n_results=5):
    chroma_client = chromadb.PersistentClient(
        path=PERSISTENT_STORAGE, settings=settings)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)

    query_result = collection.query(
        query_texts=[text],
        where=filter,
        n_results=n_results
    )
    return query_result


def create_filter(metadata):
    if metadata is None:
        return {}
    if len(metadata.items()) == 1 and list(metadata.values())[0][0] == "object":
        key, (input_type, val1, val2) = metadata.items()

        if len(val1) == 1:
            return {key: {"$eq": val1[0]}}
        else:
            or_class_filter = {"$or": []}
            for val in val1:
                or_class_filter["$or"].append({key: {"$eq": val}})

            return or_class_filter

    if len(metadata.items()) == 1 and list(metadata.values())[0][0] != "object":
        key, (input_type, val1, val2) = list(metadata.items())[0]
        return {
            "$and": [
                        {key: {"$gte": val1}},
                        {key: {"$lte": val2}}
                    ]
        }

    filter = {"$and": []}
    for key, (input_type, val1, val2) in metadata.items():
        if input_type == "object":
            if len(val1) == 1:
                filter["$and"].append({key: {"$eq": val1[0]}})
            else:
                or_class_filter = {"$or": []}
                for val in val1:
                    or_class_filter["$or"].append({key: {"$eq": val}})
                filter["$and"].append(or_class_filter)
        else:
            filter["$and"].append(
                {
                    "$and": [
                        {key: {"$gte": val1}},
                        {key: {"$lte": val2}}
                    ]
                }
            )
    return filter


def middle_embedding_vectordb(filter):
    all = query_all(filter)
    embeddings = all["embeddings"]
    embeddings = np.array(embeddings)
    middle = embeddings.mean(axis=0)
    return middle


def distances_from_middle_db(middle, filter):
    chroma_client = chromadb.PersistentClient(
        path=PERSISTENT_STORAGE, settings=settings)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)

    all = query_all(filter)
    embeddings = all["embeddings"]
    embeddings = np.array(embeddings)
    distances = np.linalg.norm(embeddings - middle, axis=1)
    return distances


def distances_from_middle(middle, embeddings):
    distances = np.linalg.norm(embeddings - middle, axis=1)
    return distances


def histogram(db_distances, generated_distances):
    plt.hist(db_distances, bins=100, alpha=0.5, label='original')
    plt.hist(generated_distances, bins=100, alpha=0.5, label='generated')
    plt.legend(loc='upper right')
    plt.show()


def generate_embeddings(df: pd.DataFrame):
    texts = df['text'].tolist()
    embeddings = EMBEDDING_FUNCTION(texts)
    embeddings = np.array(embeddings)
    return embeddings


# def average_embedding(embeddings):
#     return sum(embeddings) / len(embeddings)

# def tsne_graph(dfs : list[pd.DataFrame]):
#     embeddings = []
#     for df in dfs:
#         embeddings += generate_embeddings(df)

#     tsne = TSNE(EMBEDDING_FUNCTION.models[0].embedding_size)
#     tsne_results = tsne.fit_transform(embeddings)

#     df_subset['tsne-2d-one'] = tsne_results[:,0]
#     df_subset['tsne-2d-two'] = tsne_results[:,1]

#     plt.figure(figsize=(16,10))
#     sns.scatterplot(
#         x="tsne-2d-one", y="tsne-2d-two",
#         hue="y",
#         palette=sns.color_palette("hls", 10),
#         data=df_subset,
#         legend="full",
#         alpha=0.3
#     )


# df = pd.read_csv('./dataset/concat-formatted-reddit-dataset.csv')
# pprint(store_dataframe(df))

# pprint(query("I love anime"))

# pprint(get_metadata_type_and_classes())

# subreddits = df['subreddit'].unique()

# filter = create_filter({
#     "subreddit": ("object", ["anime", "harrypotter"], None),
#     "ups": ("int64", 0, 100),
#     "authorisgold" : ("float64", 1, 1)
# })

# pprint(filter)

# pprint(query_all(filter))

# pprint(query_random_sample(filter, 5))

# pprint(query_semantic("Generate a ton of weeb data", filter, 5))


# {
#     "$or"[
#         "subreddit": {
#             "$eq" : "anime"
#         },
#         "meta": {
#             "$eq" : "anime"
#         }
#     ]
# }