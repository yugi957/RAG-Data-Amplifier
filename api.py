from constant import EMBEDDING_FUNCTION, COLLECTION_NAME, PERSISTENT_STORAGE
import chromadb
import pandas as pd


def store_dataframe(df : pd.DataFrame):
    chroma_client = chromadb.PersistentClient(storage=PERSISTENT_STORAGE)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)

    for index, row in df.iterrows():
        print(row.to_dict())




