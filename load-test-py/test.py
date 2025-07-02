import os
import ssl
import time
import random
import json
import logging
import threading
from prometheus_client import start_http_server, Counter, Summary
import paho.mqtt.client as mqtt

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load MQTT parameters from environment variables and log them
broker_address = os.getenv("MQTT_BROKER_ADDRESS", "default_broker_address")
port = int(os.getenv("MQTT_PORT", 1883))
username = os.getenv("MQTT_USERNAME", "")
password = os.getenv("MQTT_PASSWORD", "")
topic = os.getenv("MQTT_TOPIC", "default/topic")
keepalive = int(os.getenv("MQTT_KEEPALIVE", 60))  # Keep-alive interval (seconds)
max_inflight = int(os.getenv("MAX_INFLIGHT", 20))  # Max in-flight messages
ack_timeout = int(os.getenv("ACK_TIMEOUT", 60))  # Timeout for waiting for confirmations (seconds)

logger.info(f"MQTT Broker Address: {broker_address}")
logger.info(f"MQTT Port: {port}")
logger.info(f"MQTT Username: {username}")
logger.info(f"MQTT Topic: {topic}")
logger.info(f"MQTT Keepalive: {keepalive} seconds")
logger.info(f"Max Inflight Messages: {max_inflight}")
logger.info(f"Acknowledgment Timeout: {ack_timeout} seconds")

# Load load-testing parameters from environment variables and log them
frequency = float(os.getenv("MQTT_MESSAGE_FREQUENCY", 1))  # Frequency in seconds
temperature_min = float(os.getenv("TEMPERATURE_MIN", 15))  # Minimum temperature (Celsius)
temperature_max = float(os.getenv("TEMPERATURE_MAX", 30))  # Maximum temperature (Celsius)
humidity_min = float(os.getenv("HUMIDITY_MIN", 30))  # Minimum humidity (%)
humidity_max = float(os.getenv("HUMIDITY_MAX", 70))  # Maximum humidity (%)
motion_min = int(os.getenv("MOTION_MIN", 0))  # Minimum motion value
motion_max = int(os.getenv("MOTION_MAX", 1))  # Maximum motion value
co2_min = int(os.getenv("CO2_MIN", 400))  # Minimum CO2 value (ppm)
co2_max = int(os.getenv("CO2_MAX", 10000))  # Maximum CO2 value (ppm)

logger.info(f"Message Frequency: {frequency} seconds")
logger.info(f"Temperature Range: {temperature_min} - {temperature_max} °C")
logger.info(f"Humidity Range: {humidity_min} - {humidity_max} %")
logger.info(f"Motion Range: {motion_min} - {motion_max}")
logger.info(f"CO2 Range: {co2_min} - {co2_max} ppm")

# Load additional parameters and log them
num_clients = int(os.getenv("NUM_CLIENTS", 10))  # Number of clients
use_qos = int(os.getenv("USE_QOS", 0))  # Use QoS level (0, 1, or 2)
payload_multiplier = int(os.getenv("PAYLOAD_MULTIPLIER", 1))  # Factor to increase payload size
reconnect_interval = int(os.getenv("RECONNECT_INTERVAL", 10))  # Reconnect interval (seconds)

logger.info(f"Number of Clients: {num_clients}")
logger.info(f"QoS Level: {use_qos}")
logger.info(f"Payload Multiplier: {payload_multiplier}")
logger.info(f"Reconnect Interval: {reconnect_interval} seconds")

# Prometheus metrics
undelivered_messages_counter = Counter('mqtt_undelivered_messages_total', 'Total number of MQTT undelivered messages')
delivered_messages_counter = Counter('mqtt_delivered_messages_total', 'Total number of successfully delivered MQTT messages')

# Connection and message delivery time summaries
connection_time_summary = Summary('mqtt_client_connection_time_seconds', 'Time taken for MQTT clients to connect')
message_delivery_time_summary = Summary('mqtt_message_delivery_time_seconds', 'Time taken for MQTT messages to be delivered')

# Shared variables for tracking timeout
last_ack_time = time.time()

# Function to expose Prometheus metrics via HTTP
def start_prometheus_server():
    start_http_server(8000)  # Prometheus will scrape metrics on port 8000
    logger.info("Prometheus metrics server started on port 8000")

# Define callback functions
def on_connect(client, userdata, flags, rc):
    logger.debug(f"on_connect called for Client {client._client_id.decode()} with result code {rc}")
    
    # Access userdata to get connection start time
    connection_start_time = userdata.get("connection_start_time", None)
    
    if rc == 0 and connection_start_time:
        connection_duration = time.time() - connection_start_time
        connection_time_summary.observe(connection_duration)  # Record connection time
        logger.info(f"Client {client._client_id.decode()} connected in {connection_duration:.2f} seconds and observed in metrics")
    elif rc != 0:
        logger.error(f"Failed to connect to MQTT Broker with result code {rc}")

def on_publish(client, userdata, mid):
    global last_ack_time
    last_ack_time = time.time()  # Reset the acknowledgment timer
    
    # Measure message delivery time
    publish_start_time = userdata.get("publish_start_time", None)
    if publish_start_time:
        delivery_duration = time.time() - publish_start_time
        message_delivery_time_summary.observe(delivery_duration)  # Record message delivery time
        logger.info(f"Message {mid} delivered in {delivery_duration:.2f} seconds and observed in metrics")
    
    delivered_messages_counter.inc()  # Increment delivered messages counter

def on_disconnect(client, userdata, rc):
    logger.debug(f"on_disconnect called for Client {client._client_id.decode()} with result code {rc}")
    if rc != 0:
        logger.error(f"Unexpected disconnection. Attempting to reconnect... Reason Code: {rc}")
        attempt_reconnect(client)
    else:
        logger.info("Client disconnected gracefully")

# Function to attempt reconnection
def attempt_reconnect(client):
    while True:
        try:
            logger.info("Trying to reconnect...")
            client.reconnect()
            logger.info("Reconnected successfully")
            break
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {e}. Retrying in {reconnect_interval} seconds...")
            time.sleep(reconnect_interval)

# Function to generate random sensor values with increased payload size
def generate_random_sensor_data():
    base_data = {
        "temperature": round(random.uniform(temperature_min, temperature_max), 2),
        "humidity": round(random.uniform(humidity_min, humidity_max), 2),
        "motion": random.randint(motion_min, motion_max),
        "co2": random.randint(co2_min, co2_max)
    }
    # Increase payload size by multiplying the base data
    large_data = {f"sensor_{i}": base_data for i in range(payload_multiplier)}
    return json.dumps(large_data)

# Function to create and run a single MQTT client
def run_client(client_id):
    global last_ack_time
    client = mqtt.Client(client_id=str(client_id), userdata={})

    # Assign callback functions
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect

    # Set TLS/SSL context
    tls_context = ssl.create_default_context()
    tls_context.check_hostname = False
    tls_context.verify_mode = ssl.CERT_NONE
    client.tls_set_context(context=tls_context)

    # Set username and password if required
    if username:
        client.username_pw_set(username, password)

    # Set maximum number of in-flight messages to prevent client blocking
    client.max_inflight_messages_set(max_inflight)  # Set the in-flight message limit

    # Record the time before connection attempt
    client.user_data_set({"connection_start_time": time.time()})

    logger.debug(f"[Client {client_id}] Starting connection to broker at {broker_address}:{port}")
    
    # Connect to the broker with parameterized keepalive
    client.connect(broker_address, port, keepalive=keepalive)
    client.loop_start()

    try:
        while True:
            # Check if the acknowledgment timeout has been reached
            if time.time() - last_ack_time > ack_timeout:
                logger.error(f"Acknowledgment timeout reached ({ack_timeout} seconds). Restarting clients.")
                undelivered_messages_counter.inc(max_inflight)  # Assume all in-flight messages have failed
                client.disconnect()
                time.sleep(reconnect_interval)
                attempt_reconnect(client)

            # Generate and publish random sensor data
            message = generate_random_sensor_data()
            logger.info(f"[Client {client_id}] Publishing message: {message}")
            
            # Record the time before publishing
            client.user_data_set({"publish_start_time": time.time()})

            # Record the result of the publish call
            result = client.publish(topic, message, qos=use_qos)

            # Check if the message failed to be published
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                undelivered_messages_counter.inc()  # Increment undelivered messages counter
                logger.error(f"Failed to publish message {result.mid} with result code {result.rc}")
            else:
                logger.info(f"Message {result.mid} published, awaiting delivery confirmation.")

            time.sleep(frequency)

    except KeyboardInterrupt:
        logger.info(f"[Client {client_id}] Interrupted.")
    finally:
        client.loop_stop()
        client.disconnect()

# Start the Prometheus metrics server
start_prometheus_server()

# Create and start multiple client threads
client_threads = []
for i in range(num_clients):
    t = threading.Thread(target=run_client, args=(i,))
    t.start()
    client_threads.append(t)

# Wait for all threads to complete
for t in client_threads:
    t.join()
