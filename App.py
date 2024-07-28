import sys
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import numpy as np
import io
from flask import Flask, request, jsonify


app = Flask(__name__)

# Load the model
try:
    model = tf.saved_model.load('saved_model')

except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Class labels
class_labels = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-fruit']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Convert file to BytesIO object
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # For SavedModel format, we need to use the signatures
        infer = model.signatures["serving_default"]
        preds = infer(tf.constant(img_array))
        output_key = list(preds.keys())[0]  # Get the first (and likely only) output key
        predicted_class = tf.argmax(preds[output_key], axis=1)

        return jsonify({'class': class_labels[predicted_class.numpy()[0]]})

    return jsonify({'error': 'File could not be processed'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')