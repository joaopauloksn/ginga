import requests
import datetime
import os
from pymongo import MongoClient

# Disable SSL warnings for requests
requests.packages.urllib3.disable_warnings()

# Environment variables for MongoDB and Prometheus (Thanos)
mongo_user = os.environ["MONGO_USER"]
mongo_password = os.environ["MONGO_PASSWORD"]
thanos_base_url = os.environ["THANOS_URL"]
thanos_token = os.environ["THANOS_TOKEN"]

# MongoDB connection setup
mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@mas-mongo-ce-0.mas-mongo-ce-svc.mongoce.svc.cluster.local:27017,mas-mongo-ce-1.mas-mongo-ce-svc.mongoce.svc.cluster.local:27017,mas-mongo-ce-2.mas-mongo-ce-svc.mongoce.svc.cluster.local:27017"
client = MongoClient(
    mongo_uri,
    authMechanism="SCRAM-SHA-256",
    tlsinsecure=True,
    authSource="admin",
    maxPoolSize=100,
    retryWrites=True,
    tls=True,
)
mongodb = client["mongo_scaling"]

# Authorization headers for Thanos Prometheus connection
headersAuth = {
    "Authorization": "Bearer " + str(thanos_token),
}


# Microservice class that encapsulates all logic for metrics and querying
class Microservice:
    def __init__(self, name, job, target_percentage_delivered, target_quantile_90, namespace, deployment):
        self.name = name
        self.job = job
        self.target_percentage_delivered = target_percentage_delivered
        self.target_quantile_90 = target_quantile_90
        self.namespace = namespace
        self.deployment = deployment

    def query_prometheus(self, query, query_name):
        try:
            response = requests.get(
                thanos_base_url, params={"query": query}, verify=False, headers=headersAuth
            )
            if response.status_code == 200:
                metric_result = response.json()
                if metric_result["status"] == "success":
                    if not metric_result["data"]["result"]:
                        print(f"No data for {query_name}")
                        return None
                    else:
                        return metric_result["data"]["result"]
                else:
                    print(f"Error: {metric_result['status']} for {query_name}")
                    return None
            else:
                print(f"Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while querying Prometheus: {e}")


    def get_values(self, metrics):
        if metrics is not None:
            values = metrics[0]["value"]
            try:
                value = float(values[1])
                if value == float("NaN"):
                    return None
                return value
            except ValueError:
                return None
        else:
            return None


    def get_metrics(self):
        time_series_data = {}

        # Retrieve individual metrics
        time_series_data["m_percentage__delivered"] = self.get_percentage_delivered()
        time_series_data["m_undelivered"] = self.get_undelivered_messages()
        time_series_data["m_delivered"] = self.get_delivered_messages()
        time_series_data["m_quantile_90"] = self.get_quantile_90()
        time_series_data["m_messages_perf"] = self.get_message_performance()
        time_series_data["m_cpu"] = self.get_cpu_usage()
        time_series_data["m_memory"] = self.get_memory_usage()

        # Targets
        time_series_data["t_percentage__delivered"] = self.target_percentage_delivered
        time_series_data["t_quantile_90"] = self.target_quantile_90
        time_series_data["replicas"] = self.get_number_of_pods()

        # Return the complete time series data
        return time_series_data

    def get_number_of_pods(self):
        query = 'kube_deployment_status_replicas{deployment="iotoccupancy-sample-mosquitto", namespace="iotoccupancy"}'
        query_name = "Number of Pods"

        # Call the existing query_prometheus method
        result = self.query_prometheus(query, query_name)

        if result:
            # Assuming that the first result contains the value of interest
            pod_count = result[0]['value'][1] if result[0]['value'] else None
            if pod_count is not None:
                print(f"Number of pods: {pod_count}")
                return int(pod_count)

        print("Could not retrieve the number of pods.")
        return None

    # Individual metric retrieval functions
    def get_percentage_delivered(self):
        query = f"""
        (
            sum(rate(mqtt_delivered_messages_total{{job='{self.job}'}}[1m])) by (job)
            /
            (
                sum(rate(mqtt_delivered_messages_total{{job='{self.job}'}}[1m])) by (job)
                + sum(rate(mqtt_undelivered_messages_total{{job='{self.job}'}}[1m])) by (job)
            )
        ) * 100
        """
        results = self.query_prometheus(query, "percentage_delivered")
        return self.get_values(results)

    def get_undelivered_messages(self):
        query = f"sum(rate(mqtt_undelivered_messages_total{{job='{self.job}'}}[1m])) by (job)"
        results = self.query_prometheus(query, "undelivered_messages")
        return self.get_values(results)

    def get_delivered_messages(self):
        query = f"sum(rate(mqtt_delivered_messages_total{{job='{self.job}'}}[1m])) by (job)"
        results = self.query_prometheus(query, "delivered_messages")
        return self.get_values(results)

    def get_quantile_90(self):
        query = f"histogram_quantile(0.90, sum(rate(mqtt_message_delivery_time_seconds_histogram_bucket{{job='{self.job}'}}[1m])) by (le))"
        results = self.query_prometheus(query, "quantile_90")
        return self.get_values(results)

    def get_message_performance(self):
        query = f"""
        sum(rate(mqtt_message_delivery_time_seconds_summary_sum{{job='{self.job}'}}[1m]))
        /
        sum(rate(mqtt_message_delivery_time_seconds_summary_count{{job='{self.job}'}}[1m]))
        """
        results = self.query_prometheus(query, "message_performance")
        return self.get_values(results)

    def get_cpu_usage(self):
        query = f"""
        sum(
            rate(container_cpu_usage_seconds_total{{namespace="iotoccupancy", pod=~"iotoccupancy-sample-mosquitto-.*", container!="POD"}}[1m])
        )
        """
        results = self.query_prometheus(query, "cpu_usage")
        return self.get_values(results)

    def get_memory_usage(self):
        query = f"""
        sum(
            container_memory_usage_bytes{{namespace="iotoccupancy", pod=~"iotoccupancy-sample-mosquitto-.*", container!="POD"}}
        ) / 1048576
        """
        results = self.query_prometheus(query, "memory_usage")
        return self.get_values(results)

    # Function to calculate time series entry and store it in MongoDB
    def get_time_series_entry(self, time_series_value):
        entry = time_series_value
        entry["label"] = self.get_label(time_series_value)
        entry["metadata"] = {
            "name": self.name,
            "namespace": self.namespace,
            "deployment": self.deployment,
        }
        entry["timestamp"] = datetime.datetime.utcnow()  # Add the timestamp
        return entry

    # Labeling logic
    def get_label(self, time_series):
        # If there is no messages we always considered achieved : 2
        if time_series['m_delivered'] is None and time_series['m_undelivered'] is None:
            return 2

        score = 0
        total_weight = 0

        # Define weights for the metrics
        weights = {
            "m_percentage__delivered": 3,  # Delivery success weight
            "m_quantile_90": 3  # Latency quantile weight
        }

        # Define thresholds for exceeding targets
        exceed_threshold = 1.05  # 5% above the target for label 3 (more lenient for delivery success)

        # Check m_percentage__delivered and cap it at 100%
        if time_series["m_percentage__delivered"] is not None and time_series["t_percentage__delivered"] is not None:
            delivered_percentage = min(float(time_series["m_percentage__delivered"]), 100)  # Cap at 100%
            target_delivered = float(time_series["t_percentage__delivered"])

            # Exceeds target (lenient threshold for label 3)
            if delivered_percentage >= exceed_threshold * target_delivered:
                score += weights["m_percentage__delivered"]
            # Achieves target with tolerance
            elif delivered_percentage >= 0.9 * target_delivered:  # Allow 10% deviation
                score += 0.75 * weights["m_percentage__delivered"]
            # Slightly below target, partial score
            elif delivered_percentage >= 0.75 * target_delivered:  # Allow 25% deviation
                score += 0.5 * weights["m_percentage__delivered"]

            total_weight += weights["m_percentage__delivered"]

        # Check m_quantile_90 and its target
        if time_series["m_quantile_90"] is not None and time_series["t_quantile_90"] is not None:
            quantile_90 = float(time_series["m_quantile_90"])
            target_quantile_90 = float(time_series["t_quantile_90"])

            # Exceeds target (tightened threshold for latency)
            if quantile_90 <= target_quantile_90 * 0.85:  # Exceeds by being faster
                score += weights["m_quantile_90"]
            # Achieves target with tolerance
            elif quantile_90 <= 1.1 * target_quantile_90:  # Allow 10% tolerance above target
                score += 0.75 * weights["m_quantile_90"]
            # Slightly worse than target, partial score
            elif quantile_90 <= 1.25 * target_quantile_90:  # Allow 25% tolerance above target
                score += 0.5 * weights["m_quantile_90"]

            total_weight += weights["m_quantile_90"]

        # Dynamically calculate the performance ratio based on available metrics
        performance_ratio = score / total_weight if total_weight > 0 else 0

        # Assign labels based on the performance ratio
        if performance_ratio >= 0.85:
            return 3  # Exceeds targets
        elif 0.65 <= performance_ratio < 0.85:
            return 2  # Achieves targets
        else:
            return 1  # Poor performance


# Function to persist data in MongoDB
def persist_data(entry):
    try:
        if "microservices" not in mongodb.list_collection_names():
            mongo_version = mongodb.command({"buildInfo": 1})["version"]
            if int(mongo_version[0]) >= 5:
                collection = {
                    "timeField": "timestamp",
                    "metaField": "metadata",
                    "granularity": "seconds",
                }
                mongodb.create_collection(
                    "microservices", timeseries=collection, expireAfterSeconds=21600
                )
            else:
                mongodb.create_collection("microservices")
                mongodb.microservices.ensure_index(
                    "timestamp", expireAfterSeconds=6 * 60 * 60
                )
        mongodb.microservices.insert_one(entry)
    except Exception as e:
        print(f"An error occurred while inserting data into MongoDB: {e}")
