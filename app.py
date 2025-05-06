# from flask import Flask, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Ensure GPU is used (if available)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("✅ GPU is enabled")
#     except RuntimeError as e:
#         print(f"⚠️ GPU Error: {e}")

# # Load trained model
# MODEL_PATH = r"C:\Users\nandi\Desktop\dbp\dog_breed_classifier_finetuned.h5"
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

# model = tf.keras.models.load_model(MODEL_PATH)
# print("✅ Model loaded successfully")

# # Load class labels
# LABELS_FILE = r"C:\Users\nandi\Desktop\dbp\labels.txt"

# if not os.path.exists(LABELS_FILE):
#     print("⚠️ labels.txt not found! Generating a new one with 120 placeholder labels...")
#     labels = [f"Breed_{i}" for i in range(120)]  # Replace with actual breed names
#     with open(LABELS_FILE, "w") as f:
#         f.write("\n".join(labels))
# else:
#     with open(LABELS_FILE, "r") as f:
#         labels = [line.strip() for line in f.readlines()]

# if len(labels) != 120:
#     raise ValueError(f"❌ Expected 120 labels, but found {len(labels)}. Fix labels.txt!")

# print(f"✅ Loaded {len(labels)} labels")

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/')
# def home():
#     return jsonify({"message": "🐶 Dog Breed Classifier API is running!"})

# @app.route('/test', methods=['GET'])
# def test():
#     return jsonify({"message": "✅ Test endpoint working!"})

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': '❌ No file uploaded'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': '❌ Empty filename'}), 400

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(UPLOAD_FOLDER, filename)
#     file.save(file_path)

#     try:
#         print(f"🔍 Processing image: {file_path}")

#         # Preprocess image (Ensure it matches the model's expected input size)
#         img = image.load_img(file_path, target_size=(299, 299))  # Fixed input size
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0  # Normalize

#         # Make prediction
#         predictions = model.predict(img_array)
#         print(f"📊 Raw predictions: {predictions}")  # Debugging line

#         if len(predictions) == 0:
#             return jsonify({'error': '❌ Empty prediction output'}), 500

#         predicted_index = np.argmax(predictions)

#         if predicted_index >= len(labels):
#             return jsonify({
#                 'error': '❌ Predicted index is out of range',
#                 'predicted_index': int(predicted_index),
#                 'labels_count': len(labels)
#             }), 500

#         predicted_class = labels[predicted_index]
#         confidence = float(np.max(predictions))

#         os.remove(file_path)  # Cleanup

#         return jsonify({'breed': predicted_class, 'confidence': confidence}), 200

#     except Exception as e:
#         return jsonify({'error': f'❌ {str(e)}'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Ensure GPU is used (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU is enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU Error: {e}")

# Load trained model
MODEL_PATH = r"C:\Users\nandi\Desktop\dbp\dog_breed_classifier_finetuned.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# Load class labels
LABELS_FILE = r"C:\Users\nandi\Desktop\dbp\labels.txt"

if not os.path.exists(LABELS_FILE):
    print("⚠️ labels.txt not found! Generating a new one with 120 placeholder labels...")
    labels = [f"Breed_{i}" for i in range(120)]  # Replace with actual breed names
    with open(LABELS_FILE, "w") as f:
        f.write("\n".join(labels))
else:
    with open(LABELS_FILE, "r") as f:
        labels = [line.strip() for line in f.readlines()]

if len(labels) != 120:
    raise ValueError(f"❌ Expected 120 labels, but found {len(labels)}. Fix labels.txt!")

print(f"✅ Loaded {len(labels)} labels")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return jsonify({"message": "🐶 Dog Breed Classifier API is running!"})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "✅ Test endpoint working!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '❌ No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '❌ Empty filename'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        print(f"🔍 Processing image: {file_path}")

        # Preprocess image
        img = image.load_img(file_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(img_array)
        print(f"📊 Raw predictions: {predictions}")

        if len(predictions) == 0:
            return jsonify({'error': '❌ Empty prediction output'}), 500

        predicted_index = np.argmax(predictions)
        if predicted_index >= len(labels):
            return jsonify({'error': '❌ Predicted index out of range',
                            'predicted_index': int(predicted_index),
                            'labels_count': len(labels)}), 500

        predicted_class = labels[predicted_index]
        confidence = float(np.max(predictions))

        os.remove(file_path)  # Cleanup

        return jsonify({'breed': predicted_class, 'confidence': confidence}), 200

    except Exception as e:
        return jsonify({'error': f'❌ {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
