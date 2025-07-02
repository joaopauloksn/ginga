# Use an official Python runtime as a base image
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required Python packages
RUN pip install paho-mqtt prometheus_client

# Run mqtt_client.py when the container launches
CMD ["python", "loadtest.py"]