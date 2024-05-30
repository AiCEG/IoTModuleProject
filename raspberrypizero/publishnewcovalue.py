import mh_z19
import paho.mqtt.client as mqtt
import time

# MQTT Configuration
MQTT_BROKER = "your_mqtt_broker_address"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/co2"

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
        read_and_publish_co2()
    except KeyboardInterrupt:
        print("Script interrupted by user")
    finally:
        client.disconnect()
