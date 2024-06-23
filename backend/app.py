from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import time
from threading import Lock
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import torch
import api
from api import generate_data


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

FILE_DIRECTORY = './download'
OUTPUT_FILE_NAME = 'output.csv'

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

progress_data = {"progress": 0, "tags": [], "augment_progress": 0}
progress_lock = Lock()


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        with progress_lock:
            progress_data["progress"] = 0
            progress_data["tags"] = []
        socketio.start_background_task(target=process_file, file_path=filepath)
        return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200


def process_file(file_path):
    settings = Settings(
        allow_reset=True,
    )

    COLLECTION_NAME = "collection"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    PERSISTENT_STORAGE = "vector_db"

    device = "cpu"

    EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL, device=device)

    with progress_lock:
        progress_data["progress"] = 15
        socketio.emit('progress', {'progress': 15})

    df = pd.read_csv(file_path)
    chroma_client = chromadb.PersistentClient(
        path=PERSISTENT_STORAGE, settings=settings)
    chroma_client.reset()
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=EMBEDDING_FUNCTION)

    with progress_lock:
        progress_data["progress"] = 30
        socketio.emit('progress', {'progress': 30})

    texts = df['text'].tolist()
    # list(map(str, range(len(texts))
    ids = [str(i) for i in range(0, len(texts))]
    df.drop(columns=['text'], inplace=True)
    metadatas = df.to_dict(orient='records')

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )

    with progress_lock:
        progress_data["progress"] = 85
        socketio.emit('progress', {'progress': 85})

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

    with progress_lock:
        progress_data["tags"] = metadata_type_and_classes
    socketio.emit('progress', {'progress': 100})
    socketio.emit('completed', {'tags': progress_data["tags"]})


@app.route('/tags', methods=['GET'])
def get_tags():
    with progress_lock:
        return jsonify({"tags": progress_data["tags"]})


@app.route('/augment', methods=['POST'])
def augment_data():
    data = request.get_json()
    modifier = data.get('modifier', '')

    socketio.start_background_task(
        target=process_augmentation, tags=data['items'], modifier=modifier)
    return jsonify({"message": "Augmentation started"}), 200


def process_augmentation(tags, modifier):
    filter = api.create_filter(tags)

    retrieval = ''
    if modifier:
        retrieval = api.query_semantic(modifier, filter)
        print(f"Retrieval: {retrieval}", flush=True)
    n_per_access = 60
    total = 600
    df = pd.DataFrame(columns=["text"])
    for i in range(int(total / n_per_access)):
        with progress_lock:
            progress_data["augment_progress"] = 100 * i * n_per_access / total
            socketio.emit('augment_progress', {
                          'progress': 100 * i * n_per_access / total})

        if not modifier:
            retrieval = api.query_random_sample(filter)
            print(f"Retrieval: {retrieval} from no modifier", flush=True)
        print(f'{i} of {total/n_per_access}', flush=True)

        generation = generate_data(retrieval)
        print(generation, flush=True)
        for text in generation:
            df.loc[len(df.index)] = text
    # print(retrieval, flush=True)
    # print('\n')
    # print(generation, flush=True)
    print(df, flush=True)

    # Simulate data augmentation and generating sample texts
    sample_texts = ["Sample text with tags " +
                    ", ".join(tags) + f" and modifier {modifier}"]
    with progress_lock:
        progress_data["augment_sample_texts"] = sample_texts
    socketio.emit('augment_progress', {'progress': 100})

    df.to_csv(os.path.join(FILE_DIRECTORY, OUTPUT_FILE_NAME), index=False)

    histogram_b64 = api.generate_histogram(filter)
    print(histogram_b64)
    socketio.emit('augment_completed', {
                  'sample_texts': sample_texts, 'charts': [histogram_b64]})


@app.route('/download', methods=['GET'])
def download_file():
    try:
        # Full path to the file
        file_path = os.path.join(FILE_DIRECTORY, OUTPUT_FILE_NAME)

        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=OUTPUT_FILE_NAME)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
