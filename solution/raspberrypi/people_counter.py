import subprocess
from datetime import datetime, timedelta
import os
import shutil
import time
import json
import paho.mqtt.client as mqtt
from ultralytics import YOLO
import os
from dotenv import load_dotenv

def get_raspberry_pi_serial():
    try:
        # Read the serial number from /proc/cpuinfo
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Serial'):
                    serial = line.strip().split(': ')[1]
                    return serial
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

mqtt_uri = os.getenv("MQTT_BROKER")
# MQTT Broker details
MQTT_BROKER = "mqtt_uri"
MQTT_PORT = 1883
DEVICE_ID = get_raspberry_pi_serial()
MQTT_TOPIC = f"iot/devices/camera/{DEVICE_ID}"

# Path where images will be saved
IMAGES_FOLDER = 'captured_images'

# Set up MQTT client
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# Function to take a picture using libcamera
def take_picture(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder_path}/image_{timestamp}.jpg"
    command = ['libcamera-still', '-o', filename, '--nopreview', '--timeout', '500']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Picture taken successfully and saved as {filename}!")
        return filename
    else:
        print("Failed to take picture:", result.stderr)
        return None

# Function to count people in an image
def count_people(image_path):
    model = YOLO('yolov8n.pt')  # Load the pretrained YOLOv8 model
    results = model(image_path)
    names = results[0].names    # same as model.names

    # Store number of objects detected per class label
    class_detections_values = []
    for k, v in names.items():
        class_detections_values.append(results[0].boxes.cls.tolist().count(k))
    # Create dictionary of objects detected per class
    classes_detected = dict(zip(names.values(), class_detections_values))

    return classes_detected.get('person', 0)

# Function to clean up old images
def clean_up_images(folder_path, retention_minutes=5):
    cutoff_time = datetime.now() - timedelta(minutes=retention_minutes)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_time < cutoff_time:
                os.remove(file_path)
                print(f"Removed old image: {file_path}")

while True:
    # Take a picture
    image_path = take_picture(IMAGES_FOLDER)

    if image_path:
        # Count people in the image
        people_count = count_people(image_path)
        print(f"Number of people in the image: {people_count}")

        # Send data to MQTT broker
        message = people_count
        client.publish(MQTT_TOPIC, message)
        print(f"Published message to MQTT topic {MQTT_TOPIC}: {message}")

        # Clean up old images
        clean_up_images(IMAGES_FOLDER)

    # Wait before taking another picture
    time.sleep(10)  # Adjust the sleep time as needed
