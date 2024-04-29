# IoT Module

This is our Project of the IoT Module

# Architecture Overview

- Raspberry pi
  - Camera Module
- Raspberry Zero
  - CO2 Module
- Thirdparty Server / MQTT Broker

# Software Overview

- people_counter.py runs on Raspberry Pi
- depth_estimater.py runs on Raspberry Pi

- co_measurement.py runs on Raspberry Zero

# people_counter.py

This runs on the Raspberry pi set up with the camera module.
It consists of three main features.

1. Taking pictures
2. Detecting people on the pictue and returning the people count
3. Uploading that data via MQTT

#
