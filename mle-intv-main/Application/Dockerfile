# Using a base image with Python
FROM python:3.9-slim

# Create a directory for the app
WORKDIR /app

# Copiar los archivos necesarios
COPY app.py trained_model.joblib requirements.txt /app/

# Copy the necessary files
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for API
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]