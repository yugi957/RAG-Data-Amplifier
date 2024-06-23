from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import time
from threading import Lock

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

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
    for progress in range(0, 101, 40):
        socketio.sleep(1)
        with progress_lock:
            progress_data["progress"] = progress
            socketio.emit('progress', {'progress': progress})
    # Simulate file processing and generating tags
    tags = ["Tag1", "Tag2", "Tag3"]
    with progress_lock:
        progress_data["tags"] = tags
    socketio.emit('progress', {'progress': 100})
    socketio.emit('completed', {'tags': tags})

@app.route('/tags', methods=['GET'])
def get_tags():
    with progress_lock:
        return jsonify({"tags": progress_data["tags"]})

@app.route('/augment', methods=['POST'])
def augment_data():
    data = request.get_json()
    tags = data.get('tags', [])
    modifier = data.get('modifier', '')

    socketio.start_background_task(target=process_augmentation, tags=tags, modifier=modifier)
    return jsonify({"message": "Augmentation started"}), 200

def process_augmentation(tags, modifier):
    for progress in range(0, 101, 40):
        socketio.sleep(1)
        with progress_lock:
            progress_data["augment_progress"] = progress
            socketio.emit('augment_progress', {'progress': progress})
    
    # Simulate data augmentation and generating sample texts
    sample_texts = ["Sample text with tags " + ", ".join(tags) + f" and modifier {modifier}"]
    with progress_lock:
        progress_data["augment_sample_texts"] = sample_texts
    socketio.emit('augment_progress', {'progress': 100})
    socketio.emit('augment_completed', {'sample_texts': sample_texts})

if __name__ == '__main__':
    socketio.run(app, debug=True)
