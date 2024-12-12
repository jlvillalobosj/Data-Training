import joblib
import pandas as pd
import boto3
import os

def get_file_from_s3(bucket_name,file_name):
    # Set up the S3 client
    s3 = boto3.client('s3')
    # Bucket and file details
    bucket_name = bucket_name
    object_key = file_name
    local_file_path = f"/tmp/{object_key}"  # Temporary download

    # Download the file from S3
    s3.download_file(bucket_name, object_key, local_file_path)
    return local_file_path


def lambda_handler(event, context):
    # Load the saved model
    file_model = get_file_from_s3(os.getenv("BUCKET_NAME"),os.getenv("FILE_NAME"))
    model = joblib.load(file_model)
    try:
        data = pd.DataFrame(event)

        # Make predictions
        predictions = model.predict(data)
        return {
            "statusCode": 200,
            "body": f"predictions: {predictions.tolist()}"
        }
    except Exception as e:
        return {
            "statusCode": 200,
            "body": f"error: {str(e)}"
        }