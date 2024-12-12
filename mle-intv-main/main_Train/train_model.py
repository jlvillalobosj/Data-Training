# Standard Library Imports
import math
import json
import logging

# General Imports
import argparse
import joblib
import pandas as pd

# SKLearn Imports

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, RocCurveDisplay



def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df_X = df.drop("y", axis=1)
    df_label = df["y"]

    return df_X, df_label

def create_pipeline_data_processing(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    
    categorical_transformer = OneHotEncoder(handle_unknown="infrequent_if_exist")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=10000))]
    )

    return clf

def train_model(clf, X_train, y_train):
    return clf.fit(X_train, y_train)

def evaluate_model(clf, X_test, y_test):
    print("model score: %.3f" % clf.score(X_test, y_test))
    
    tprobs = clf.predict_proba(X_test)[:, 1]   # matriz de dimensiones (n_samples, n_classes)
    print(classification_report(y_test, clf.predict(X_test)))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, clf.predict(X_test)))
    print(f'AUC: {roc_auc_score(y_test, tprobs)}')
    RocCurveDisplay.from_estimator(estimator=clf,X= X_test, y=y_test)


def save_model(clf, file_path_model):
    joblib.dump(clf, file_path_model)
    print(f"Model saved to {file_path_model}")


def load_model(file_path):
    return joblib.load(file_path)



def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    file_path = config["data_path"]
    model_save_path = config["model_save_path"]
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    RANDOM_STATE = config["random_state"]
    
    # Load Data
    dataFrame = load_data(file_path)

    # Separate dependent and independent variables
    df_X, df_Y = preprocess_data(dataFrame)
    
    # Create pipeline for process data
    classifier = create_pipeline_data_processing(numeric_features, categorical_features)



    # Splitting data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        df_X, df_Y, random_state=RANDOM_STATE
    )

    # Train model
    clf = train_model(classifier, X_train, y_train)

    # Evaluate model
    evaluate_model(clf, X_test, y_test)

    # Save model
    save_model(clf, model_save_path)


if __name__ == "__main__":
    main()