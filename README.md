# IoT Module

This is our Project of the IoT Module. We are collecting co2 data and counting the people in a room with image recognition. The goal is to gather data, build a ml model to predict how many people are in a room based on the co2 levels.

# Document and Solution Overview

The Powerpoint is in the solution folder. All our final software is in the solution folder.
The rest is just what we tried, used and tested. To show the work and progress.

# Architecture Overview

- Raspberry pi
  - Camera Module
- Raspberry Pi Zero
  - CO2 Module
- Raspberry Pi
  - Discovery Collector. (MQTT Subscriber storing measurements to MongoDB)
- External Thirdparty Server / MQTT Broker (Simple docker container running on a rented server)

- Mobile Router for Connectivity

# Software Overview

- people_counter.py runs on Raspberry Pi

- collect_co2_publish.py runs on Raspberry Pi Zero

- discovery_collector.py.py runs on Raspberry

All the Scripts are registered as services to start on startup of the raspberry pi.

# people_counter.py

This runs on the Raspberry pi set up with the camera module.
It consists of three main features.

1. Taking pictures
2. Detecting people on the pictue and returning the people count
3. Uploading that data via MQTT

#
