import os
import logging
import pandas as pd
import numpy as np
from pymongo import MongoClient
from tensorflow.keras.models import load_model
from kubernetes import client, config

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB Connection
def connect_to_mongo():
    mongo_user = os.getenv('MONGO_USERNAME')
    mongo_password = os.getenv('MONGO_PASSWORD')

    if not mongo_user or not mongo_password:
        logger.error("MongoDB credentials not set in environment variables.")
        return None

    mongo_uri = (
        f"mongodb://{mongo_user}:{mongo_password}@"
        "mas-mongo-ce-0.mas-mongo-ce-svc.mongoce.svc.cluster.local:27017,"
        "mas-mongo-ce-1.mas-mongo-ce-svc.mongoce.svc.cluster.local:27017,"
        "mas-mongo-ce-2.mas-mongo-ce-svc.mongoce.svc.cluster.local:27017"
    )

    try:
        client = MongoClient(
            mongo_uri,
            authMechanism="SCRAM-SHA-256",
            tlsInsecure=True,
            authSource="admin",
            connectTimeoutMS=60000,
            socketTimeoutMS=60000,
            maxPoolSize=100,
            retryWrites=True,
            tls=True,
        )

        mongodb = client["mongo_scaling"]
        collection = mongodb['microservices']
        logger.info("Connected to MongoDB successfully.")
        return collection
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return None

def get_latest_data():
    collection = connect_to_mongo()
    if collection is None:
        logger.error("Failed to connect to MongoDB or collection not found.")
        return None

    data = pd.DataFrame(list(collection.find().sort("timestamp", -1).limit(1)))
    if data.empty:
        logger.error("No data found for prediction.")
        return None

    logger.info(f"Data used for prediction:\n{data}")
    return data

def prepare_data_for_prediction(data):
    # Ensure we only use the relevant columns
    if 'm_delivered' not in data.columns or 'm_undelivered' not in data.columns:
        logger.error("Required columns 'm_delivered' and 'm_undelivered' are missing.")
        return None

    # Calculate total messages
    data['total_messages'] = data['m_delivered'] + data['m_undelivered']

    # Log the values for debugging
    logger.debug(f"Delivered messages: {data['m_delivered'].iloc[0]}")
    logger.debug(f"Undelivered messages: {data['m_undelivered'].iloc[0]}")
    logger.debug(f"Total messages: {data['total_messages'].iloc[0]}")

    # Check for NaN in total_messages
    if pd.isna(data['total_messages'].iloc[0]):
        logger.warning("Total messages is NaN. Returning None to indicate a problem.")
        return None  # Signal to the calling function that the input is not valid

    # Normalize using a fixed maximum value used during training (e.g., 300)
    max_total_messages_training = 300  # Replace this with the actual max used during training if different
    data['total_messages'] = (data['total_messages'] / max_total_messages_training) * 300

    # Safely convert to int after ensuring no NaN
    data['total_messages'] = data['total_messages'].fillna(0).astype(int)

    logger.debug(f"Normalized total messages for prediction: {data['total_messages'].iloc[0]}")
    
    X_scaled = data[['total_messages']].to_numpy().astype(np.float32).reshape(1, -1)
    return X_scaled


def predict_from_model(model_path):
    data = get_latest_data()
    if data is None or data.empty:
        logger.error("No data available for prediction.")
        return 1  # Return a default of 1 replica when no data is available

    data_scaled = prepare_data_for_prediction(data)
    if data_scaled is None:
        logger.error("Invalid data for prediction (e.g., NaN detected). Returning a default of 1 replica.")
        return 1  # Return a default of 1 replica when data preparation fails due to NaN

    # Proceed with prediction if data is valid
    try:
        model = load_model(model_path, compile=False)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1  # Return a default of 1 replica if the model loading fails

    try:
        prediction = model.predict(data_scaled)
        logger.debug(f"Raw model prediction: {prediction}")
        predicted_replicas = int(np.ceil(prediction[0][0]))
        logger.info(f"Predicted number of replicas: {predicted_replicas}")
        return predicted_replicas
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return 1  # Return a default of 1 replica if an error occurs during prediction

def get_current_replicas(deployment_name, namespace):
    try:
        # Load the Kubernetes configuration
        config.load_incluster_config()  # Use load_kube_config() if running outside the cluster

        # Create an API instance
        api_instance = client.AppsV1Api()
        deployment = api_instance.read_namespaced_deployment(name=deployment_name, namespace=namespace)
        current_replicas = deployment.spec.replicas

        logger.debug(f"Current number of replicas for '{deployment_name}' in namespace '{namespace}': {current_replicas}")
        return current_replicas
    except client.exceptions.ApiException as e:
        logger.error(f"Failed to get current number of replicas: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while getting current number of replicas: {e}")
        return None

def scale_deployment(deployment_name, namespace, replicas):
    current_replicas = get_current_replicas(deployment_name, namespace)
    if current_replicas is None:
        logger.error("Could not retrieve current replicas. Scaling operation aborted.")
        return

    # Only scale if the desired number of replicas is different from the current number
    if current_replicas == replicas:
        logger.info(f"No scaling needed. The current number of replicas ({current_replicas}) matches the desired number.")
        return

    try:
        # Load the Kubernetes configuration
        config.load_incluster_config()  # Use load_kube_config() if running outside the cluster

        # Create an API instance
        api_instance = client.AppsV1Api()

        # Prepare the scale patch
        scale_body = {
            "spec": {
                "replicas": replicas
            }
        }

        # Patch the deployment with the new replica count
        response = api_instance.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=scale_body
        )

        logger.info(f"Deployment '{deployment_name}' in namespace '{namespace}' scaled to {replicas} replicas.")
        return response
    except client.exceptions.ApiException as e:
        logger.error(f"Failed to scale deployment: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during scaling: {e}")



def predict_and_scale(model_path):
    predicted_replicas = predict_from_model(model_path)
    if predicted_replicas is not None:
        deployment_name = os.getenv('DEPLOYMENT_NAME', 'default-deployment')
        namespace = os.getenv('NAMESPACE', 'default-namespace')

        # Ensure predicted replicas are within reasonable bounds (e.g., minimum of 1)
        predicted_replicas = max(1, predicted_replicas)

        scale_deployment(deployment_name, namespace, predicted_replicas)

if __name__ == "__main__":
    model_path = os.getenv('MODEL_PATH', '/mnt/model/best_replica_predictor_model.h5')
    predict_and_scale(model_path)
