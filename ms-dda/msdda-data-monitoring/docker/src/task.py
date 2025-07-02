import threading
from datetime import datetime
import data_monitoring.monitor as monitor
import os

t_quantile = os.environ["T_QUANTILE"]
t_message_delivered_perc = os.environ["T_MESSAGE_DELIVERED_PERCENTAGE"]
NAMESPACE = os.environ["NAMESPACE"]
DEPLOYMENT = os.environ["DEPLOYMENT"]


def run_script():
    # Re-run the script every 10 seconds
    threading.Timer(10.0, run_script).start()
    print("############### Gathering microservice information ###############")

    # Instantiate the Microservice object with correct targets and metrics
    userLogin = monitor.Microservice(
        name="mosquitto",
        job="mqtt-load-tester-metrics",
        target_percentage_delivered=t_message_delivered_perc,  # Set target percentage delivered
        target_quantile_90=t_quantile,  # Quantile target for message latency
        namespace=NAMESPACE,
        deployment=DEPLOYMENT
    )

    # Get the metrics and time series data
    time_series = userLogin.get_metrics()

    # Create the time series entry based on the metrics
    entry = userLogin.get_time_series_entry(time_series)

    # Persist the data in MongoDB
    monitor.persist_data(entry)
    print(f"entry: {entry}")

# Start the script
print(f"Running Data Monitoring component script at {datetime.now()}")
run_script()
