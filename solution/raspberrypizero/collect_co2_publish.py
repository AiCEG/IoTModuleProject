import mh_z19
import paho.mqtt.client as mqtt
import time
import subprocess
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

dev_id = get_raspberry_pi_serial()

mqtt_uri = os.getenv("MQTT_BROKER")
# MQTT Configuration
MQTT_BROKER = "mqtt_uri"
MQTT_PORT = 1883
MQTT_TOPIC = f"iot/devices/co2/{dev_id}"


# MQTT Client setup
client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

def read_and_publish_co2():
    while True:
        # Read CO2 concentration from MH-Z19 sensor
        co2_data = mh_z19.read()

        if co2_data:
            co2_value = co2_data.get('co2')
            if co2_value:
                # Print the CO2 value to the console
                print(f"CO2 Concentration: {co2_value} ppm")

                # Publish the CO2 value to the MQTT broker
                client.publish(MQTT_TOPIC, co2_value)

        # Wait for a specified time before the next reading
        time.sleep(10)

if __name__ == "__main__":
    try:
        mh_z19.abc_on()
        read_and_publish_co2()
    except KeyboardInterrupt:
        print("Script interrupted by user")
    finally:
       client.disconnect()
