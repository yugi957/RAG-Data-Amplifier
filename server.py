from flask import Flask, request, jsonify
from pyngrok import ngrok
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.utils import embedding_functions
from test import get_random_sample

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "fast_docs"

# Initialize Chroma client and collection
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection = client.get_collection(
    name=COLLECTION_NAME,
)

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, this is your Flask app!"

load_dotenv()

ngrok_auth_token = os.getenv('NGROK_AUTH_TOKEN')

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY')
)

@app.route('/few-shot-prompt', methods=['POST'])
def few_shot_prompt():
    data = request.get_json()

    # Retrieve examples and the input prompt from the request
    # Construct the few-shot prompt
    random_shots = get_random_sample(collection, 5)
    few_shot_prompt = "Use these documents::::"
    for shot in random_shots['documents']:
        few_shot_prompt += f"{shot}\n\n"
    few_shot_prompt += f":::to generate 10 different texts that are similar to those documents considering context, writing style, and topic\n"

    # Make a request to the OpenAI API using the new interface
    try:
        stream = client.chat.completions.create(
            model="gpt-4o",  # or another suitable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": few_shot_prompt}
            ],
            max_tokens=1500
        )
        # print(stream)
        response = stream.choices[0].message.content
        return jsonify({"response": response})

    except openai.APIStatusError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    ngrok.set_auth_token(ngrok_auth_token)

    # Set up a tunnel to the Flask server port
    public_url = ngrok.connect(5000).public_url
    print("Public URL:", public_url)

    # Run the Flask server
    app.run(port=5000)