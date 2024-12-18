# Using a base image with Python
FROM public.ecr.aws/lambda/python:3.9

# Create a directory for the app
WORKDIR /app

# Copiar los archivos necesarios
COPY mle-intv-main/Application/AWS/model_car_AWS.py ${LAMBDA_TASK_ROOT}

# Copy the necessary file
RUN pip install pandas joblib boto3 --target "${LAMBDA_TASK_ROOT}"
RUN pip install -U scikit-learn --target "${LAMBDA_TASK_ROOT}"

# Command to run the application
CMD ["model_car_AWS.lambda_handler"]