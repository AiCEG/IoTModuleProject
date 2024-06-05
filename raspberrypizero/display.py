import paho.mqtt.client as mqtt
from PIL import Image, ImageDraw, ImageFont
import time
from waveshare_epd import epd2in13_V2
import logging
import os
from dotenv import load_dotenv

# MQTT Configuration
mqtt_uri = os.getenv("MQTT_BROKER")
MQTT_BROKER = mqtt_uri
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/co2"

# Global variable to store CO2 value
co2_value = None

# Initialize the e-Paper display
epd = epd2in13_V2.EPD()
epd.init(epd.FULL_UPDATE)
epd.Clear(0xFF)

# MQTT Callback functions
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    global co2_value
    co2_value = msg.payload.decode()
    print(f"Received CO2 data: {co2_value}")
    update_display()

def update_display():
    global co2_value
    try:
        # Create a blank image for drawing
        Himage = Image.new('1', (epd.height, epd.width), 255)
        draw = ImageDraw.Draw(Himage)

        # Load a font
        font24 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)

        # Draw text
        draw.text((10, 10), f"CO2: {co2_value} ppm", font=font24, fill=0)

        # Display the image
        epd.display(epd.getbuffer(Himage))
        epd.sleep()
    except OSError as e:
        print(f"Error updating display: {e}")
        # Reinitialize the e-Paper display
        epd.init(epd.FULL_UPDATE)
        epd.Clear(0xFF)

# MQTT Client setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Start the MQTT client loop
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("Script interrupted by user")
    epd.init(epd.FULL_UPDATE)
    epd.Clear(0xFF)
    epd.sleep()
finally:
    client.disconnect()
