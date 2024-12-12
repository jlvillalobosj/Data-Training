# Using a base image with Python
FROM python:3.9-slim

# Create a directory for the app
WORKDIR /app
ARG filemodel

# Copiar los archivos necesarios
COPY mle-intv-main/controller/model_cars.py /app/
COPY ${filemodel} /app/

# Copy the necessary files
RUN pip install numpy scipy pandas joblib flask
RUN pip install scikit-learn==1.5.1

# Expose port for API
EXPOSE 5000

# Command to run the application
CMD ["python", "model_cars.py"]