from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from train_model import load_data, preprocess_data, create_pipeline_data_processing, train_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    try:
        # Load the JSON file sent from Postman
        config = request.get_json()

        # Read configuration from JSON
        file_path = "mle-intv-main/main_Train/data/" + config["data_name"]
        model_save_path = "mle-intv-main/Application/model_versions/" + config["model_save_name"] + ".joblib"
        numeric_features = config["numeric_features"]
        categorical_features = config["categorical_features"]
        RANDOM_STATE = config["random_state"]           

        # Loading and preparing data
        dataFrame = load_data(file_path)
        df_X, df_Y = preprocess_data(dataFrame)

        # Create the preprocessing pipeline and model
        classifier = create_pipeline_data_processing(numeric_features, categorical_features)

        # Split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            df_X, df_Y, random_state=RANDOM_STATE
        )

        # Train the model
        clf = train_model(classifier, X_train, y_train)

        # Evaluate the model
        classification, confusion, auc = evaluate_model(clf, X_test, y_test)

        # Save the trained model
        save_model(clf, model_save_path)

        respounse = {
                    "a-message": "Modelo entrenado exitosamente",
                    "model saved in ": model_save_path,
                    "classification report": classification,
                    "matrix confussion: ": confusion,
                    "auc score": auc
                    }

        return jsonify(respounse)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
