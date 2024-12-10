from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load("trained_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read data from request body
        input_data = request.get_json()
        
        data = pd.DataFrame(input_data)

        # Make predictions
        print(data)
        predictions = model.predict(data)
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400



app.run(host="0.0.0.0", port=5000)

