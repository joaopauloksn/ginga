import os
import logging
import time
from datetime import datetime
from model_deployment.predict import predict_and_scale

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv('MODEL_PATH', '/mnt/model/best_replica_predictor_model.h5')

# Global variable to store the last used model path
last_used_model = None

def check_for_new_model():
    """
    Checks if there is a new model available and returns its path if found.
    """
    global last_used_model

    # Check if the model file exists
    if os.path.exists(MODEL_PATH):
        # If it's a new model, update last_used_model
        if last_used_model != MODEL_PATH:
            last_used_model = MODEL_PATH
            logger.info(f"New or updated model found: {MODEL_PATH}. Using new model.")
        else:
            logger.info(f"No new model found. Continuing with the current model: {MODEL_PATH}")
    else:
        logger.error(f"Model file not found: {MODEL_PATH}")
        return None

    return last_used_model

def run_script():
    """
    Runs the prediction script at regular intervals, using the latest available model.
    """
    logger.info(f"Running Model Deployment component script at {datetime.now()}")

    while True:
        logger.info(f"Checking for new model at {datetime.now()}...")
        model_path = check_for_new_model()

        if model_path:
            logger.info("Running prediction with the current model...")
            predict_and_scale(model_path)
        else:
            logger.error("No model available for prediction.")

        # Wait for 5 seconds before the next check
        time.sleep(5)

if __name__ == "__main__":
    run_script()
