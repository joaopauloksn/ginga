# Use a base image with Python and the necessary dependencies
FROM python:3.9-slim

# Install kubectl
RUN apt-get update && \
    apt-get install -y curl && \
    curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x ./kubectl && \
    mv ./kubectl /usr/local/bin/kubectl

# Set environment variables for Kubernetes deployment
ENV DEPLOYMENT_NAME=iotoccupancy-sample-mosquitto
ENV NAMESPACE=iotoccupancy

# Copy the Python script into the container
COPY load_test_scaler.py /app/load_test_scaler.py

# Set the working directory
WORKDIR /app

# Run the Python script
CMD ["python", "load_test_scaler.py"]
