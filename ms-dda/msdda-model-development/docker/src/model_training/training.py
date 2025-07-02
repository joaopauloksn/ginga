import os
import gc
import logging
import pandas as pd
import numpy as np
from pymongo import MongoClient
import tensorflow as tf
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# TensorFlow threading and mixed precision settings
tf.config.threading.set_intra_op_parallelism_threads(2)
mixed_precision.set_global_policy('mixed_float16')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BEST_MSE_FILE = os.getenv('BEST_MSE_FILE', '/mnt/model/best_mse.txt')

def get_data_from_mongo():
    """Connects to MongoDB and retrieves records."""
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
        client.server_info()
        logger.info("Connected to MongoDB successfully.")
        data = pd.DataFrame(list(collection.find().sort("timestamp", -1)))

        if len(data) < 50:
            logger.warning(f"Insufficient data: {len(data)} records found.")
            return None

        data = data.drop(['_id', 'timestamp', 'metadata'], axis=1, errors='ignore')
        logger.info(f"Retrieved {len(data)} latest records from the database.")
        return data

    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None

def prepare_data(data):
    """Prepares the data for training by computing the total number of replicas and filtering by label."""
    logger.info("Preparing data for training...")

    # Filter the data to include only rows with label 2
    data = data[data['label'] == 2].copy()  # Ensure you're working on a copy
    logger.info(f"Number of records after filtering by label 2: {len(data)}")

    # Check if the filtered data has enough records
    if len(data) < 50:
        logger.warning(f"Training could not start due to insufficient data (less than 50 records) at {pd.Timestamp.now()}.")
        return None, None, None, None

    if 'm_delivered' not in data.columns or 'm_undelivered' not in data.columns:
        logger.error("Required columns 'm_delivered' and 'm_undelivered' are missing.")
        return None, None, None, None

    # Calculate total number of replicas
    data['total_msg'] = data['m_delivered'] + data['m_undelivered']

    # Use 'total_msg' as the only feature and 'replicas' as the target
    X = data[['total_msg']]
    y = data['replicas']

    # Check for NaN values in y and handle them
    if y.isnull().any():
        logger.warning(f"Found {y.isnull().sum()} NaN values in the target variable 'replicas'. Dropping these records.")
        non_nan_indices = ~y.isnull()
        X = X[non_nan_indices]
        y = y[non_nan_indices]

    # Handle NaN values in X by replacing with 0
    X = X.fillna(0)

    # Normalize the 'total_msg' feature to be in the range [0, 300]
    X['total_msg'] = (X['total_msg'] / X['total_msg'].max()) * 300
    X['total_msg'] = X['total_msg'].astype(int)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Data preparation complete.")
    return X_train, X_test, y_train, y_test

def load_best_mse():
    """Loads the best MSE from the file."""
    try:
        with open(BEST_MSE_FILE, 'r') as file:
            best_mse = float(file.read().strip())
            logger.info(f"Loaded best MSE: {best_mse}")
            return best_mse
    except FileNotFoundError:
        logger.info("No previous best MSE found.")
        return np.inf
    except Exception as e:
        logger.error(f"Error reading best MSE file: {e}")
        return np.inf

def save_best_mse(mse):
    """Saves the best MSE to the file."""
    try:
        with open(BEST_MSE_FILE, 'w') as file:
            file.write(str(mse))
            logger.info(f"Saved new best MSE: {mse}")
    except Exception as e:
        logger.error(f"Error saving best MSE: {e}")

def build_model(input_dim):
    """Builds an enhanced neural network model using the total replicas feature."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))  # Increased number of neurons
    model.add(Dropout(0.3))  # Increased Dropout for better regularization
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate for more stable training
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    return model

def run_training():
    """
    Runs the training process, using 'total_msg' as the only feature.
    The model is saved only if it achieves a better MSE than the current best MSE.
    """
    logger.info("Starting the training process...")

    # Reset memory
    gc.collect()
    logger.info("Memory cleared with garbage collection.")

    # Fetch data from MongoDB
    data = get_data_from_mongo()
    if data is None:
        logger.warning("No data available or insufficient records for training.")
        return None, np.inf

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data)
    if X_train is None:
        logger.error("Data preparation failed.")
        return None, np.inf

    # Load the current best MSE
    best_mse = load_best_mse()

    # Build and train the model
    model = build_model(input_dim=X_train.shape[1])

    # Add early stopping and learning rate scheduler
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,  # Increased epochs for longer training
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler],  # Added learning rate scheduler
        verbose=1
    )

    # Evaluate the final model
    y_pred = model.predict(X_test).flatten()

    # Check for NaN in predictions
    if np.isnan(y_pred).any():
        logger.error("Model prediction contains NaN values. Aborting evaluation.")
        return None, np.inf

    final_mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Final Model MSE: {final_mse:.2f}")

    # Save the model if it achieves a better MSE than the current best MSE
    if final_mse < best_mse:
        model_path = os.getenv('MODEL_PATH', '/mnt/model/best_replica_predictor_model.h5')
        model.save(model_path)
        logger.info(f"New best MSE achieved: {final_mse}. Model saved to {model_path}.")
        save_best_mse(final_mse)
    else:
        logger.info(f"Model did not improve. Best MSE remains: {best_mse}.")

    # Clear memory after training
    K.clear_session()
    gc.collect()
    logger.info("Final memory cleared after training.")

    return final_mse, best_mse

if __name__ == "__main__":
    run_training()
