from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

base_path = os.getenv("MODEL_BASE_PATH", "./mle-intv-main/Application/")  # Default folder: ./mle-intv-main/Application/

@app.route("/score/<model_name>", methods=["POST"])
def predict(model_name):
    try:
        # Construct the full path of the model file
        model_path = os.path.join(base_path, f"{model_name}.joblib")

        # Load the saved model  
        model = joblib.load(model_path)
        
        # Check if the file exists
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model '{model_name}' not found"}), 404
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

