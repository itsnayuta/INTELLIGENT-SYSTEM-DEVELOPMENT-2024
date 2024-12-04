from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load pre-trained models
model_cnn = tf.keras.models.load_model("trained_models/cnn_model.h5")
model_rnn = tf.keras.models.load_model("trained_models/lstm_model.h5")
model_lstm = tf.keras.models.load_model("trained_models/lstm_model.h5")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        model_type = data.get("model", "cnn")  # Default to CNN if no model is selected
        lat = float(data.get("lat", 0))
        bedroom = int(data.get("bedroom", 0))
        areaM2 = float(data.get("areaM2", 0))

        # Kiểm tra giá trị hợp lệ
        if not (lat and bedroom and areaM2):
            return jsonify({"error": "Invalid input data"})

        # Combine inputs into a single numpy array
        input_data = np.array([[lat, bedroom, areaM2]])

        # Chọn mô hình dựa trên lựa chọn
        if model_type == "cnn":
            model = model_cnn
        elif model_type == "rnn":
            model = model_rnn
        elif model_type == "lstm":
            model = model_lstm
        else:
            return jsonify({"error": "Invalid model selected"})

        # Dự đoán giá nhà
        predicted_price = model.predict(input_data)

        # Trả kết quả
        return jsonify({"price": round(float(predicted_price[0][0]), 2)})

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
