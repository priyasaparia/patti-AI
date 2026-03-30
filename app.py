import os
os.environ["KERAS_BACKEND"] = "jax"

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import timedelta
import uuid, base64, datetime, json
import numpy as np
import threading
import requests
import time
import keras

from PIL import Image

app = Flask(__name__, static_folder='static')
app.secret_key = 'pattiAI_super_secret_key_2024_xyz'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_NAME'] = 'patti_session'
app.config['SESSION_COOKIE_PATH'] = '/'
app.config['SESSION_COOKIE_DOMAIN'] = None
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

CORS(app, 
     supports_credentials=True, 
     origins=["http://localhost:5000", "http://127.0.0.1:5000"],
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['patti_ai']
    users_col = db['users']
    searches_col = db['searches']
    print("✅ MongoDB connected")
except Exception as e:
    print(f"⚠️  MongoDB error: {e}")
    client = None
    db = None
    users_col = None
    searches_col = None

# ── Upload folder ─────────────────────────────────────────────────────────────
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Model (loaded in background thread to avoid blocking startup) ─────────────
model = None
model_loading_error = None
model_loading_status = "initializing"

def load_model_background():
    global model, model_loading_error, model_loading_status
    try:
        model_loading_status = "downloading"
        print("⏳ Loading model from Hugging Face...")
        from huggingface_hub import hf_hub_download
        import keras

        print("📥 Downloading model from HuggingFace...")
        model_path = hf_hub_download(
            repo_id="priyanshisaparia/Pattiai",
            filename="plant_disease_resnet50.keras",
            token=os.environ.get('HUGGINGFACE_HUB_TOKEN'),
            local_files_only=False
        )
        print(f"✅ Model downloaded to: {model_path}")

        print("🔄 Loading model into memory...")
        model_loading_status = "loading"
        model = keras.saving.load_model(model_path)
        model_loading_status = "ready"
        print("✅ Model loaded successfully!")
    except Exception as e:
        model_loading_error = str(e)
        model_loading_status = "failed"
        print(f"⚠️ Model load failed: {e}")
        import traceback
        traceback.print_exc()
        model = None

# Start model loading in background thread
model_thread = threading.Thread(target=load_model_background)
model_thread.daemon = True
model_thread.start()

class_names = {}
CLASS_PATH = 'class_indices.json'
if os.path.exists(CLASS_PATH):
    with open(CLASS_PATH) as f:
        class_to_idx = json.load(f)
    class_names = {v: k for k, v in class_to_idx.items()}
    print(f"✅ Classes loaded → {len(class_names)} classes")
else:
    print(f"⚠️  class_indices.json not found!")


def predict_image(img_path):
    if model is None:
        return {"disease": "Model not loaded", "confidence": 0, "healthy": False}
    try:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, 0)
        preds = model(arr)
        preds = np.array(preds)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = class_names.get(idx, f"Class_{idx}")
        healthy = 'healthy' in label.lower()
        return {
            "disease": label,
            "confidence": round(conf * 100, 2),
            "healthy": healthy
        }
    except Exception as e:
        return {"disease": str(e), "confidence": 0, "healthy": False}


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route('/api/signup', methods=['POST'])
def signup():
    if users_col is None:
        return jsonify({'error': 'Database not connected'}), 500
    d = request.get_json()
    if not d or not d.get('email') or not d.get('password') or not d.get('name'):
        return jsonify({'error': 'All fields required'}), 400
    if users_col.find_one({'email': d['email']}):
        return jsonify({'error': 'Email already registered'}), 400
    users_col.insert_one({
        'name': d['name'],
        'email': d['email'],
        'password': generate_password_hash(d['password']),
        'created_at': datetime.datetime.utcnow()
    })
    return jsonify({'message': 'Account created!'})


@app.route('/api/login', methods=['POST'])
def login():
    if users_col is None:
        return jsonify({'error': 'Database not connected'}), 500
    d = request.get_json()
    if not d or not d.get('email') or not d.get('password'):
        return jsonify({'error': 'All fields required'}), 400
    user = users_col.find_one({'email': d['email']})
    if not user or not check_password_hash(user['password'], d['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    session.permanent = True
    session['user_id'] = str(user['_id'])
    session['user_name'] = user['name']
    session['user_email'] = user['email']
    print(f"✅ Login: {user['email']} | session: {dict(session)}")
    return jsonify({'name': user['name'], 'email': user['email']})


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out'})


@app.route('/api/me', methods=['GET'])
def me():
    print(f"🔍 /api/me called | session: {dict(session)}")
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    return jsonify({'name': session['user_name'], 'email': session['user_email']})


# ── Predict ───────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    print(f"🔍 /api/predict | session: {dict(session)}")
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if model is None:
        if model_loading_status == "loading" or model_loading_status == "downloading":
            return jsonify({'error': 'Model is still loading, please try again in a moment'}), 503
        return jsonify({'error': 'Model not loaded', 'details': model_loading_error}), 503

    try:
        img_data = None
        fname = ''
        path = ''

        if 'file' in request.files:
            f = request.files['file']
            filename = secure_filename(f.filename)
            ext = filename.rsplit('.', 1)[-1] if '.' in filename else 'jpg'
            fname = f"{uuid.uuid4()}.{ext}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            f.save(path)
            with open(path, 'rb') as fp:
                img_data = base64.b64encode(fp.read()).decode()

        elif request.is_json and 'image_b64' in request.json:
            b64 = request.json['image_b64'].split(',')[-1]
            img_bytes = base64.b64decode(b64)
            fname = f"{uuid.uuid4()}.jpg"
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            with open(path, 'wb') as fp:
                fp.write(img_bytes)
            img_data = b64
        else:
            return jsonify({'error': 'No image provided'}), 400

        result = predict_image(path)

        if searches_col is not None:
            searches_col.insert_one({
                'user_id': session['user_id'],
                'user_email': session['user_email'],
                'filename': fname,
                'image_full': img_data,
                'result': result,
                'timestamp': datetime.datetime.utcnow()
            })

        return jsonify(result)

    except Exception as e:
        print(f"❌ Predict error: {e}")
        return jsonify({'error': str(e)}), 500


# ── History ───────────────────────────────────────────────────────────────────
@app.route('/api/history', methods=['GET'])
def history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    if searches_col is None:
        return jsonify([])
    docs = list(searches_col.find(
        {'user_id': session['user_id']},
        sort=[('timestamp', -1)],
        limit=20
    ))
    for d in docs:
        d['_id'] = str(d['_id'])
        d['timestamp'] = d['timestamp'].isoformat()
    return jsonify(docs)


# ── Health check ─────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    if model is None:
        if model_loading_status == "failed":
            return jsonify({"status": "error", "model": "failed", "error": model_loading_error}), 503
        return jsonify({"status": "ok", "model": "loading", "progress": model_loading_status}), 200
    return jsonify({"status": "ok", "model": "ready"}), 200


# ── Keep Alive (Render free tier fix) ────────────────────────────────────────
def keep_alive():
    url = os.environ.get('RENDER_EXTERNAL_URL', 'http://localhost:10000') + '/health'
    while True:
        time.sleep(600)  # ping every 10 minutes
        try:
            requests.get(url)
            print("Keep-alive ping sent.")
        except Exception as e:
            print(f"Keep-alive failed: {e}")

thread = threading.Thread(target=keep_alive)
thread.daemon = True
thread.start()


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    print("🌿 Starting Patti AI server...")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
