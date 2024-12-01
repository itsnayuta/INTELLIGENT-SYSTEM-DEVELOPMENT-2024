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
loaded_model = load_model('cuoicungdiemthi.h5')
loaded_model_cnn = load_model('cuoicungdiemthi_cnn.h5')
loaded_model_rnn = load_model('cuoicungdiemthi_rnn.h5')
loaded_model_lstm = load_model('cuoicungdiemthi_lstm.h5')


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

    # ---get the prediction class---
    prediction = loaded_model.predict(np.array([features_list]))
    prediction_cnn = loaded_model_cnn.predict(np.array([features_list]))
    prediction_rnn = loaded_model_rnn.predict(np.array([features_list]))
    prediction_lstm = loaded_model_lstm.predict(np.array([features_list]))
    prediction = float(prediction[0][0])
    prediction_cnn = float(prediction_cnn[0][0])
    prediction_rnn = float(prediction_rnn[0][2][0])
    prediction_lstm = float(prediction_lstm[0][2][0])
    response = {
        'prediction': prediction,
        'prediction_cnn': prediction_cnn,
        'prediction_rnn': prediction_rnn,
        'prediction_lstm': prediction_lstm
        # 'confidence': str(round(np.amax(confidence[0]) * 100, 2))
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
