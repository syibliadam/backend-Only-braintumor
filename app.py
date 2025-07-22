# ==== IMPORTS ====
import os
import sys
import io
import datetime
import base64
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail
from pymongo import MongoClient
from tensorflow.keras.models import load_model, Model
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
import jwt
from hashlib import sha256
import matplotlib.cm as cm
import tensorflow as tf
import requests
import gdown  # ✅ ditambahkan untuk download dari Google Drive

# 🔁 Cloudinary
import cloudinary
import cloudinary.uploader

# ==== ENV & PATH ====
load_dotenv()
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ==== INIT APP ====
app = Flask(__name__)
CORS(app)

# ==== CONFIG ====
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# MongoDB
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client['tumorvision_db']
users_collection = db['users']

# Flask-Mail
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
mail = Mail(app)

# Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('API_KEY'),
    api_secret=os.getenv('API_SECRET')
)

# ==== BLUEPRINT ====
from auth.route import auth_bp
app.register_blueprint(auth_bp, url_prefix='/auth')

# ==== LOAD MODEL DARI GOOGLE DRIVE JIKA BELUM ADA ====
MODEL_PATH = 'stacked_fold_1.h5'
FILE_ID = '1rW-1a0YNfn9yE2pXP5xl0yrnuIJCgkAX'  # ✅ file ID dari Google Drive

def download_model_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"[INFO] Model tidak ditemukan. Mengunduh dari Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url, output_path, quiet=False)
            print("✅ Model berhasil diunduh.")
        except Exception as e:
            print("❌ Gagal mengunduh model:", e)
            sys.exit(1)
    else:
        print("✅ Model sudah tersedia secara lokal.")

download_model_from_drive(FILE_ID, MODEL_PATH)

try:
    model = load_model(MODEL_PATH)
    print("✅ Model klasifikasi loaded successfully.")
except Exception as e:
    print("❌ Error loading klasifikasi model:", e)
    sys.exit(1)

# ==== UTILS ====
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_superimposed_image(image_array, heatmap):
    img = cv2.resize(image_array, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    superimposed_img = cv2.addWeighted(img, 0.6, jet_heatmap, 0.4, 0)
    return superimposed_img

def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, np.array(img)
    except UnidentifiedImageError:
        raise ValueError("File bukan gambar valid.")

def handle_prediction(image_bytes, filename, token=None):
    try:
        image, original_image = preprocess_image(image_bytes)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    except ValueError as e:
        return {'error': str(e)}, 400

    start_time = datetime.datetime.now()
    prediction = model.predict(image)
    duration_ms = (datetime.datetime.now() - start_time).total_seconds() * 1000

    predicted_class_idx = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    class_labels = {0: 'Meningioma', 1: 'Glioma', 2: 'Pituitary'}
    label = class_labels.get(predicted_class_idx, 'Unknown') if confidence >= 0.85 else 'Tidak Diketahui'

    now = datetime.datetime.utcnow().replace(microsecond=0)
    result_string = f"{label}-{confidence:.5f}-{now.isoformat()}"
    result_hash = sha256(result_string.encode()).hexdigest()

    # Grad-CAM
    gradcam_base64 = None
    try:
        heatmap = make_gradcam_heatmap(image, model, last_conv_layer_name='top_conv')
        overlay_img = generate_superimposed_image(original_image, heatmap)
        _, buffer = cv2.imencode(".jpg", overlay_img)
        gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print("❌ Gagal generate Grad-CAM:", e)

    # Simpan ke history
    if token:
        try:
            decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            username = decoded.get('username')
            exists = users_collection.find_one({
                'username': username,
                'history.hash': result_hash
            })
            if not exists:
                users_collection.update_one(
                    {'username': username},
                    {'$push': {
                        'history': {
                            'timestamp': now,
                            'result': label,
                            'confidence': f"{confidence * 100:.2f}%",
                            'filename': filename,
                            'hash': result_hash,
                            'image_url': None
                        }}
                    }
                )
        except Exception as e:
            print("❌ Gagal menyimpan history:", e)

    return {
        'class_index': predicted_class_idx,
        'class_name': label,
        'confidence': f"{confidence * 100:.2f}%",
        'probabilities': prediction[0].tolist(),
        'gradcam': gradcam_base64,
        'inference_time_ms': f"{duration_ms:.2f} ms"
    }, 200

# ==== ROUTES ====
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file diunggah'}), 400

    uploaded_file = request.files['file']
    filename = uploaded_file.filename
    image_bytes = uploaded_file.read()

    # Upload ke Cloudinary
    try:
        upload_result = cloudinary.uploader.upload(io.BytesIO(image_bytes), resource_type="image")
        image_url = upload_result.get('secure_url')
    except Exception as e:
        print("❌ Gagal upload ke Cloudinary:", e)
        image_url = None

    token = request.headers.get('Authorization')
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]

    result, status = handle_prediction(image_bytes, filename, token)
    result['image_url'] = image_url
    return jsonify(result), status

@app.route('/predict-from-url', methods=['POST'])
def predict_from_url():
    data = request.get_json()
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({'error': 'URL gambar tidak ditemukan'}), 400

    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Gagal mengunduh gambar dari URL'}), 400
        image_bytes = response.content
    except Exception as e:
        return jsonify({'error': f'Gagal mengakses URL: {e}'}), 400

    token = request.headers.get('Authorization')
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]

    result, status = handle_prediction(image_bytes, image_url.split('/')[-1], token)
    result['image_url'] = image_url
    return jsonify(result), status

# ==== RUN ====
if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
