from constant import EMBEDDING_FUNCTION, COLLECTION_NAME, PERSISTENT_STORAGE
import chromadb
import pandas as pd
import os

def store_dataframe(df : pd.DataFrame):
    chroma_client = chromadb.PersistentClient(path=PERSISTENT_STORAGE)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    
    if not os.path.exists(f"{PERSISTENT_STORAGE}/base_id.txt"):
        with open(f"{PERSISTENT_STORAGE}/base_id.txt", 'w') as f:
            f.write("0")

    with open(f"{PERSISTENT_STORAGE}/base_id.txt", 'r') as f:
        BASE_ID = int(f.read())

    texts = df['text'].tolist()
    ids = [str(i) for i in range(BASE_ID, BASE_ID + len(texts))]
    BASE_ID += len(texts)
    metadatas = df.drop(columns=['text']).to_dict(orient='records') 
    
    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )

    with open(f"{PERSISTENT_STORAGE}/base_id.txt", 'w') as f:
        f.write(str(BASE_ID))

    # for index, row in df.iterrows():
    #     print(index)
    #     row_dict = row.to_dict()
    #     text = row_dict['text']
    #     cur_id = str(index)
        
    #     del row_dict['text']
    #     metadata = row_dict

    #     # print(text)
    #     # print(metadata)

    #     collection.add(
    #         documents=[text],
    #         ids=[cur_id],
    #         metadatas=[metadata]
    #     )


def query(text : str):
    chroma_client = chromadb.PersistentClient(path=PERSISTENT_STORAGE)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)
    
    query_result = collection.query(
        query_texts=[text],
        n_results=1
    )
    return query_result



from pprint import pprint
pprint(query("I love anime"))

# df = pd.read_csv('./dataset/formatted-reddit-dataset/entertainment_anime.csv')

# store_dataframe(df)