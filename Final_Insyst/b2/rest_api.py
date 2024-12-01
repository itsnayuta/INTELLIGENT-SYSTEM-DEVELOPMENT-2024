from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_file
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---the filename of the saved model---
filename = 'cuoicungdiemthi.h5'
filename_cnn = 'cuoicungdiemthi_cnn.h5'
filename_rnn = 'cuoicungdiemthi_rnn.h5'
filename_lstm = 'cuoicungdiemthi_lstm.h5'

# ---load the saved model---
try:
    loaded_model = load_model(filename)
    loaded_model_cnn = load_model(filename_cnn)
    loaded_model_rnn = load_model(filename_rnn)
    loaded_model_lstm = load_model(filename_lstm)
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def home():
    return send_file('predict.html')

@app.route('/diabetes/v1/predict', methods=['POST'])
def predict():
    # ---get the features to predict---
    features = request.json
    # ---create the features list for prediction---
    features_list = [
        float(features["1"]),
        float(features["2"]),
        float(features["3"])
    ]
    
    try:
        # ---get the prediction from the models---
        prediction = loaded_model.predict(np.array([features_list]))
        prediction_cnn = loaded_model_cnn.predict(np.array([features_list]))
        prediction_rnn = loaded_model_rnn.predict(np.array([features_list]))
        prediction_lstm = loaded_model_lstm.predict(np.array([features_list]))

        # ---handle predictions based on model outputs---
        prediction = float(prediction[0][0])  # Assuming the output is 2D (1, 1)
        prediction_cnn = float(prediction_cnn[0][0])  # Assuming CNN output is also 2D (1, 1)
        
        # Handle RNN and LSTM predictions if they return 3D arrays
        if len(prediction_rnn.shape) > 2:
            prediction_rnn = float(prediction_rnn[0][2][0])  # Adjust according to your model's output
        else:
            prediction_rnn = float(prediction_rnn[0][0])  # Fallback to 2D handling
        
        if len(prediction_lstm.shape) > 2:
            prediction_lstm = float(prediction_lstm[0][2][0])  # Adjust according to your model's output
        else:
            prediction_lstm = float(prediction_lstm[0][0])  # Fallback to 2D handling
        
        # Prepare the response
        response = {
            'prediction': prediction,
            'prediction_cnn': prediction_cnn,
            'prediction_rnn': prediction_rnn,
            'prediction_lstm': prediction_lstm
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
