import paho.mqtt.client as mqtt
from pymongo import MongoClient
import json
import os
from dotenv import load_dotenv

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
mqtt_uri = os.getenv("MQTT_BROKER")

# MQTT Broker details
MQTT_BROKER = mqtt_uri
MQTT_PORT = 1883
TOPIC = "iot/devices/#"  # Subscribe to all device-related topics

# MongoDB connection details
DATABASE_NAME = "IoTCo2"
COLLECTION_NAME = "measurement_data"

# MongoDB client setup
mongo_client = MongoClient(mongo_uri)
db = mongo_client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Callback function when connecting to the broker
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe(TOPIC)
    print(f"Subscribed to topic: {TOPIC}")

# Callback function when a message is received
def on_message(client, userdata, msg):
    print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")
    try:
        # Parse device id and type from the topic
        parts = msg.topic.split('/')
        if len(parts) >= 4:
            device_type = parts[2]
            device_id = parts[3]
        else:
            print("Invalid topic structure")
            return

        value = json.loads(msg.payload.decode())

        # Construct the document
        document = {
            'device_id': device_id,
            'type': device_type,
            'value': value
        }
        collection.insert_one(document)
        print(f"Stored message in MongoDB: {document}")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON message: {e}")
    except Exception as e:
        print(f"Failed to store message in MongoDB: {e}")

# Setting up MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting.
client.loop_forever()
