import os
import time
import subprocess
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to scale the deployment
def scale_deployment(replicas):
    deployment_name = os.getenv("DEPLOYMENT_NAME", "iotoccupancy-sample-mosquitto")
    namespace = os.getenv("NAMESPACE", "iotoccupancy")
    
    # Log the replica scaling attempt
    logging.info(f"Attempting to scale the deployment '{deployment_name}' to {replicas} replicas in the namespace '{namespace}'")

    command = ["kubectl", "scale", "--replicas", str(replicas), "deployment", deployment_name, "-n", namespace]
    
    try:
        # Run the kubectl command to scale the deployment
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # Log the result of the scaling action
        logging.info(f"Successfully scaled to {replicas} replicas. Output: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        # Log the error if the scaling fails
        logging.error(f"Error scaling deployment: {e.stderr.strip()}")

def main():
    # Record the start time of the load test
    start_time = datetime.now()
    logging.info("Load test started")

    # Load test phases defined in terms of duration (in minutes) and number of replicas
    phases = [
        {"time": 5, "replicas": [1, 2, 4, 6]},        # Ramp-up phase
        {"time": 5, "replicas": [10]},                # First Spike
        {"time": 5, "replicas": [6, 4, 2]},           # First Cool-down
        {"time": 10, "replicas": [10, 12]},           # Second Spike
        {"time": 5, "replicas": [8, 6, 4, 2]}         # Final Cool-down
    ]
    
    # Loop through each phase of the load test
    for phase_index, phase in enumerate(phases):
        phase_duration = phase["time"]  # Duration of the current phase in minutes
        phase_replicas = phase["replicas"]  # Replica scaling targets for this phase
        interval_duration = phase_duration / len(phase_replicas)  # Duration between replica changes in minutes
        
        logging.info(f"Starting phase {phase_index + 1}: Scaling over {phase_duration} minutes with replicas {phase_replicas}")

        # Loop through each replica configuration in the current phase
        for step_index, replicas in enumerate(phase_replicas):
            elapsed_time = (datetime.now() - start_time).total_seconds() / 60  # Elapsed time in minutes
            logging.info(f"Phase {phase_index + 1}, Step {step_index + 1}: Elapsed time {elapsed_time:.2f} minutes")
            
            # Scale the deployment to the desired number of replicas
            scale_deployment(replicas)
            
            # Log the sleep period between scaling actions
            logging.info(f"Sleeping for {interval_duration:.2f} minutes before the next scaling step")
            
            # Sleep for the calculated interval duration before the next scaling change
            time.sleep(interval_duration * 60)  # Convert interval duration to seconds

    # Load test is completed
    logging.info("Load test completed successfully")


if __name__ == "__main__":
    # Run the main function to start the load test
    main()
