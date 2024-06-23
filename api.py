from constant import EMBEDDING_FUNCTION, COLLECTION_NAME, PERSISTENT_STORAGE
import chromadb
import pandas as pd
import os

def store_dataframe(df : pd.DataFrame):
    chroma_client = chromadb.PersistentClient(path=PERSISTENT_STORAGE)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    
    texts = df['text'].tolist()
    ids = [str(i) for i in range(0, len(texts))]
    df.drop(columns=['text'], inplace=True)
    metadatas = df.to_dict(orient='records')
    
    # collection.add(
    #     documents=texts,
    #     ids=ids,
    #     metadatas=metadatas
    # )

    metadata_column_types = dict(df.dtypes)
    metadata_type_and_classes = {}
    for column_name, column_type in metadata_column_types.items():
        if column_type == 'object':
            print(str(column_type))
            uniques_values = df[column_name].unique()
            if len(uniques_values) < 30:
                metadata_type_and_classes[column_name] = (str(column_type), list(uniques_values))
        else:
            metadata_type_and_classes[column_name] = (str(column_type), None)

    return metadata_type_and_classes

def query(text : str):
    chroma_client = chromadb.PersistentClient(path=PERSISTENT_STORAGE)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    
    query_result = collection.query(
        query_texts=[text],
        n_results=5
    )
    return query_result





# from pprint import pprint
# pprint(query("I love anime"))

df = pd.read_csv('./dataset/formatted-reddit-dataset/entertainment_anime.csv')

store_dataframe(df)