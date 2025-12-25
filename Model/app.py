import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load your trained TensorFlow model
MODEL_PATH = "TrainModel"  # Update this path to point to your model file
model = load_model(MODEL_PATH)

# Helper function to preprocess image (you can customize this)
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image / 255.0  # Normalize the image

@app.route('/')
def home():
    return "Welcome to the TensorFlow model prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read the image file
        img = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        processed_image = preprocess_image(img, target_size=(224, 224))  # Adjust based on your model input

        # Make the prediction using the model
        predictions = model.predict(processed_image)

        # Extract prediction details
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        return jsonify({
            "predicted_class": int(predicted_class),
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
 