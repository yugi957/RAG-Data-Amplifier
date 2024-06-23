from constant import EMBEDDING_FUNCTION, COLLECTION_NAME, PERSISTENT_STORAGE
import chromadb
import pandas as pd
import os
import json

from chromadb.config import Settings

settings = Settings(
    allow_reset=True,
)

def store_dataframe(df : pd.DataFrame):
    chroma_client = chromadb.PersistentClient(path=PERSISTENT_STORAGE,settings=settings)
    chroma_client.reset()
    collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    
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
            metadata_type_and_classes[column_name] = (str(column_type), list(uniques_values))
        else:
            metadata_type_and_classes[column_name] = (str(column_type), None)

    with open(f"{PERSISTENT_STORAGE}/metadata_type_and_classes.json", "w") as f:
        json.dump(metadata_type_and_classes, f)
        
    return metadata_type_and_classes

def get_metadata_type_and_classes():
    with open(f"{PERSISTENT_STORAGE}/metadata_type_and_classes.json", "r") as f:
        return json.load(f)
    

def query(text : str):
    chroma_client = chromadb.PersistentClient(path=PERSISTENT_STORAGE, settings=settings)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    
    query_result = collection.query(
        query_texts=[text],
        n_results=5
    )
    return query_result


def query_random_sample(filters):
    chroma_client = chromadb.PersistentClient(path=PERSISTENT_STORAGE, settings=settings)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    
    query_result = collection.get(
        where=filters,
    )

def create_filter(metadata):
    filter = {"$or": []}
    for key, (input_type, val1, val2) in metadata.items():
        if input_type == "object":
            filter["$or"].append({key: {"$eq": val1}})
        else:
            filter["$or"].append(
                {
                    "$and" : [
                        {key: {"$gte": val1}},
                        {key: {"$lte": val2}}
                    ]
                }
            )
    return filter

from pprint import pprint

# df = pd.read_csv('./dataset/concat-formatted-reddit-dataset.csv')
# pprint(store_dataframe(df))

# pprint(query("I love anime"))

# pprint(get_metadata_type_and_classes())



pprint(create_filter({
    "subreddit": ("object", "anime", None),
    "ups": ("int64", 0, 100),
}))

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