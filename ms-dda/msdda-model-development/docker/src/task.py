import os
import numpy as np
import time
from datetime import datetime
from model_training.training import run_training

BEST_MSE_FILE = os.getenv('BEST_MSE_FILE', '/mnt/model/best_mse.txt')

def reset_best_mse():
    """
    Resets the best MSE file to a large number (infinity).
    """
    try:
        with open(BEST_MSE_FILE, 'w') as file:
            file.write(str(np.inf))
            print(f"Best MSE reset to infinity at {datetime.now()}.")
    except Exception as e:
        print(f"Error resetting best MSE: {e}")

def run_script():
    """
    Repeatedly calls the training function with a blocking loop.
    Waits for the current training to complete before starting a new one.
    """
    # Reset the best MSE at the start of the script
    reset_best_mse()

    while True:
        print(f"Running training at {datetime.now()}...")
        
        # Call the training function
        final_mse, best_mse = run_training()

        # Handle the training results
        if final_mse is None:
            print(f"Training could not start due to insufficient data (less than 50 records) at {datetime.now()}.")
        elif final_mse < best_mse:
            print(f"New best MSE achieved: {final_mse:.2f} (Previous best MSE: {best_mse:.2f}). Model updated.")
        else:
            print(f"Training completed. Final MSE: {final_mse:.2f}, Best MSE: {best_mse:.2f}. No improvement.")

        # Wait for 5 seconds before the next iteration
        time.sleep(5)

if __name__ == "__main__":
    print(f"Running Model Development component script at {datetime.now()}")
    run_script()
