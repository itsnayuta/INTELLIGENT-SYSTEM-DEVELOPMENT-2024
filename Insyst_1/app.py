from flask import Flask, request, render_template,jsonify, send_from_directory
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model
# Load pre-trained models
cnn_model = load_model('cnn_mnist_model.h5', compile=False)
rnn_model = load_model('rnn_mnist_model.h5', compile=False)

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocess the image before passing to model
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 as required by MNIST
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=(0, -1))  # Reshape to (1, 28, 28, 1)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file is uploaded
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        
        if file.filename == '':
            return 'No selected file'
        
        # Open the image file
        if file:
            image = Image.open(file.stream)  # Open image
            preprocessed_image = preprocess_image(image)

            # Save the uploaded image to display it later
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            image.save(image_path)

            # Predict with both CNN and RNN models
            cnn_prediction = np.argmax(cnn_model.predict(preprocessed_image), axis=-1)[0]
            rnn_prediction = np.argmax(rnn_model.predict(preprocessed_image), axis=-1)[0]

            # Generate the URL for the uploaded image
            image_url = f'/{image_path}'

            # Render the result template with predictions and image URL
            return render_template('result.html', cnn_prediction=cnn_prediction, rnn_prediction=rnn_prediction, image_url=image_url)
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
