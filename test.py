from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import warnings
import openai
import pandas as pd
import os
from ast import literal_eval

# Chroma's client library for Python
import chromadb

# I've set this to our new embeddings model, this can be changed to the embedding model of your choice
EMBEDDING_MODEL = "text-embedding-3-small"

chroma_client = chromadb.PersistentClient(path="./chroma_db")


os.environ["OPENAI_API_KEY"] = 'sk-proj-fgPiNeOokaEPUywp38kVT3BlbkFJpFRG12AZKA2SY6O679et'

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)

reddit_collection = chroma_client.get_or_create_collection(
    name='reddit')

# Load the dataset
csv_directory = './reddit-dataset-master'
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
new_headers = [
    'text', 'id', 'subreddit', 'meta', 'time',
    'author', 'ups', 'downs', 'authorlinkkarma',
    'authorcommentkarma', 'authorisgold'
]

reddit_comment_df = pd.DataFrame()
for csv_file in csv_files:
    file_path = os.path.join(csv_directory, csv_file)
    # Read only the first 100 rows
    data = pd.read_csv(file_path, nrows=100)
    data = data.drop(data.columns[0], axis=1)
    print(data.columns)
    print(file_path)
    data.columns = new_headers
    # Create a string representation of each row
    data['document'] = data.apply(lambda row: ', '.join(
        [f"{col}: {row[col]}" for col in data.columns if col != 'document']), axis=1)

    reddit_comment_df = pd.concat([reddit_comment_df, data])


# Function to read CSV files and add data to ChromaDB
def load_data():
    print("start")
    reddit_collection.add(
        ids=[str(i) for i in range(len(reddit_comment_df))],
        documents=reddit_comment_df['document'].tolist(),
    )
    print("done")

def query_collection(collection, query, max_results, dataframe):
    results = collection.query(
        query_texts=query, n_results=max_results, include=['distances'])
    df = pd.DataFrame({
        'id': results['ids'][0],
        'score': results['distances'][0],
        'text': dataframe.iloc[results['ids'][0]]['text'],
    })

    return df


query_result = query_collection(
    collection=reddit_collection,
    query="modern art in Europe",
    max_results=10,
    dataframe=reddit_comment_df
)
print(query_result)