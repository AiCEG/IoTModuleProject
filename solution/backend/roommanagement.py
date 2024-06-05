import streamlit as st
from pymongo import MongoClient
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import comodel_complete
import comodel_room
import matplotlib.pyplot as plt

load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["IoTCo2"]
rooms_collection = db["rooms"]
measurement_collection = db["measurement_data"]

# Function to load rooms data
def load_rooms():
    rooms = list(rooms_collection.find())
    return pd.DataFrame(rooms)

# Function to load measurement data
def load_measurement_data():
    measurements = list(measurement_collection.find())
    return pd.DataFrame(measurements)

def load_co_measurement_room_data(room_id):
    room = rooms_collection.find_one({"_id": room_id})
    measurements = load_measurement_data()

    #filter for device id and within date range
    measurements = measurements[measurements["device_id"].isin([room["co2_device_uuid"]])]
    measurements = measurements[(measurements["datetime"] >= room["start_measurements"]) & (measurements["datetime"] <= room["end_measurements"])]
    return pd.DataFrame(measurements)

def load_all_measurement_room_data(room_id):
    room = rooms_collection.find_one({"_id": room_id})
    measurements = load_measurement_data()

    #filter for device id and within date range
    measurements = measurements[measurements["device_id"].isin([room["co2_device_uuid"],room["camera_device_uuid"]])]
    measurements = measurements[(measurements["datetime"] >= room["start_measurements"]) & (measurements["datetime"] <= room["end_measurements"])]
    return pd.DataFrame(measurements)

# Function to add a new room
def add_room(room_data):
    rooms_collection.insert_one(room_data)

# Function to update a room
def update_room(room_id, updated_data):
    rooms_collection.update_one({"_id": room_id}, {"$set": updated_data})

# Function to delete a room
def delete_room(room_id):
    rooms_collection.delete_one({"_id": room_id})

# Function to fetch devices sending data in the last 5 minutes by type
def get_recent_devices(device_type):
    five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
    recent_measurements = measurement_collection.find({
        "datetime": {"$gte": five_minutes_ago},
        "type": device_type
    })
    recent_devices = list(set(measurement["device_id"] for measurement in recent_measurements))
    return recent_devices

def combine_date_time(date, time):
    return datetime.combine(date, time)

def room_management():
    st.subheader("Room List")
    rooms_df = load_rooms()

    if not rooms_df.empty:
        st.dataframe(rooms_df)
    else:
        st.write("No rooms available.")

    # Add new room
    st.subheader("Add New Room")
    with st.form("add_room_form"):
        room_name = st.text_input("Room Name")
        co2_device_uuid = st.selectbox("CO2 Device UUID", options=get_recent_devices("co2"))
        camera_device_uuid = st.selectbox("Camera Device UUID", options=get_recent_devices("camera"))
        m3 = st.number_input("M^3", min_value=0.0)
        m2 = st.number_input("M^2", min_value=0.0)
        start_date = st.date_input("Start Date", value=datetime.now())
        start_time = st.time_input("Start Time", value=datetime.now().time())
        end_date = st.date_input("End Date", value=datetime.now() + timedelta(days=1))
        end_time = st.time_input("End Time", value=datetime.now().time())
        submitted = st.form_submit_button("Add Room")

        if submitted:
            start_measurements = combine_date_time(start_date, start_time)
            end_measurements = combine_date_time(end_date, end_time)
            room_data = {
                "room_name": room_name,
                "co2_device_uuid": co2_device_uuid,
                "camera_device_uuid": camera_device_uuid,
                "m3": m3,
                "m2": m2,
                "start_measurements": start_measurements,
                "end_measurements": end_measurements
            }
            add_room(room_data)
            st.success("Room added successfully.")
            st.experimental_rerun()

    # Update room
    st.subheader("Update Room")
    room_ids = rooms_df["_id"].tolist() if not rooms_df.empty else []
    room_options = {str(room["_id"]): room["room_name"] for _, room in rooms_df.iterrows()}
    selected_room_id = st.selectbox("Select Room to Update", options=room_ids, format_func=lambda x: f"{x} - {room_options[str(x)]}")

    if selected_room_id:
        selected_room = rooms_collection.find_one({"_id": selected_room_id})
        with st.form("update_room_form"):
            room_name = st.text_input("Room Name", value=selected_room.get("room_name", ""))
            co2_device_uuid = st.selectbox(
                "CO2 Device UUID",
                options=get_recent_devices("co2"),
                index=get_recent_devices("co2").index(selected_room["co2_device_uuid"]) if selected_room["co2_device_uuid"] in get_recent_devices("co2") else 0
            )
            camera_device_uuid = st.selectbox(
                "Camera Device UUID",
                options=get_recent_devices("camera"),
                index=get_recent_devices("camera").index(selected_room["camera_device_uuid"]) if selected_room["camera_device_uuid"] in get_recent_devices("camera") else 0
            )
            m3 = st.number_input("M^3", value=selected_room["m3"], min_value=0.0)
            m2 = st.number_input("M^2", value=selected_room["m2"], min_value=0.0)
            start_date = st.date_input("Start Date", value=selected_room.get("start_measurements", datetime.now()).date())
            start_time = st.time_input("Start Time", value=selected_room.get("start_measurements", datetime.now()).time())
            end_date = st.date_input("End Date", value=selected_room.get("end_measurements", datetime.now() + timedelta(days=1)).date())
            end_time = st.time_input("End Time", value=selected_room.get("end_measurements", datetime.now() + timedelta(days=1)).time())
            submitted = st.form_submit_button("Update Room")

            if submitted:
                start_measurements = combine_date_time(start_date, start_time)
                end_measurements = combine_date_time(end_date, end_time)
                updated_data = {
                    "room_name": room_name,
                    "co2_device_uuid": co2_device_uuid,
                    "camera_device_uuid": camera_device_uuid,
                    "m3": m3,
                    "m2": m2,
                    "start_measurements": start_measurements,
                    "end_measurements": end_measurements
                }
                update_room(selected_room_id, updated_data)
                st.success("Room updated successfully.")
                st.experimental_rerun()

    # Remove room
    st.subheader("Remove Room")
    selected_room_id_for_deletion = st.selectbox("Select Room to Remove", options=room_ids, format_func=lambda x: f"{x} - {room_options[str(x)]}")

    if selected_room_id_for_deletion:
        if st.button("Remove Room"):
            delete_room(selected_room_id_for_deletion)
            st.success("Room removed successfully.")
            st.experimental_rerun()

def ml_models():
    # Generate ML model for room
    rooms_df = load_rooms()
    room_ids = rooms_df["_id"].tolist() if not rooms_df.empty else []
    room_options = {str(room["_id"]): room["room_name"] for _, room in rooms_df.iterrows()}

    st.subheader("Generate ML Model")

    if st.button("Train ML Model based on all rooms"):
        comodel_complete.generate_ml_model()
        st.success("ML model generation started.")

    selected_room_id_for_ml = st.selectbox("Select Room for ML Model", options=room_ids, format_func=lambda x: f"{x} - {room_options[str(x)]}")
    if st.button("Train ML Model for selected room"):
        comodel_room.generate_ml_model(selected_room_id_for_ml)
        st.success("ML model generation started.")

def measurement_data():
    st.subheader("Device Measurement Data")

    # Load rooms and let the user select a room
    rooms_df = load_rooms()
    room_ids = rooms_df["_id"].tolist() if not rooms_df.empty else []
    room_options = {str(room["_id"]): room["room_name"] for _, room in rooms_df.iterrows()}
    selected_room_id = st.selectbox("Select Room", options=room_ids, format_func=lambda x: f"{x} - {room_options[str(x)]}")

    if selected_room_id:
        # Load measurement data for the selected room
        measurements_df = load_all_measurement_room_data(selected_room_id)

        if not measurements_df.empty:
            st.dataframe(measurements_df)
        else:
            st.write("No measurement data available for the selected room.")

        # Visualization of measurement data
        st.subheader("Measurement Data Visualization")

        # Select device to visualize data
        device_uuids = measurements_df.apply(lambda x: f"{x['device_id']} - {x['type']}", axis=1).unique().tolist()
        selected_device_uuid = st.selectbox("Select Device UUID to Visualize", options=device_uuids)
        if selected_device_uuid:
            device_id = selected_device_uuid.split(" - ")[0]
            device_data = measurements_df[measurements_df["device_id"] == device_id]
            st.line_chart(device_data.set_index("datetime")["value"])

def prediction_plot():
    st.subheader("Prediction Plot")
    rooms_df = load_rooms()
    room_ids = rooms_df["_id"].tolist() if not rooms_df.empty else []
    room_options = {str(room["_id"]): room["room_name"] for _, room in rooms_df.iterrows()}
    selected_room_id = st.selectbox("Select Room for Predictions", options=room_ids, format_func=lambda x: f"{x} - {room_options[str(x)]}")

    if selected_room_id:
        # Load measurement data for the selected room
        co2_measurements = load_co_measurement_room_data(selected_room_id)
        co2_measurements['datetime'] = pd.to_datetime(co2_measurements['datetime'])
        co2_measurements.sort_values(by='datetime', inplace=True)

        # Prepare the data for prediction
        features = co2_measurements[['value']].copy()
        features.rename(columns={'value': 'co2_value'}, inplace=True)
        features['m3'] = rooms_df.loc[rooms_df['_id'] == selected_room_id, 'm3'].values[0]
        features['m2'] = rooms_df.loc[rooms_df['_id'] == selected_room_id, 'm2'].values[0]

        # Calculate CO2 changes
        features['co2_change_30s'] = features['co2_value'].diff(periods=3)
        features['co2_change_60s'] = features['co2_value'].diff(periods=6)
        features['co2_change_300s'] = features['co2_value'].diff(periods=30)
        features['co2_change_900s'] = features['co2_value'].diff(periods=90)

        features['co2_change_30s_rel'] = features['co2_change_30s'] / features['co2_value'].shift(3)
        features['co2_change_60s_rel'] = features['co2_change_60s'] / features['co2_value'].shift(6)
        features['co2_change_300s_rel'] = features['co2_change_300s'] / features['co2_value'].shift(30)
        features['co2_change_900s_rel'] = features['co2_change_900s'] / features['co2_value'].shift(90)

        features.dropna(inplace=True)

        # Predict using the room-specific model
        room_specific_predictions = comodel_room.load_and_predict(selected_room_id, features)

        # Predict using the complete model (assuming comodel_complete is a general model)
        complete_predictions = comodel_complete.load_and_predict(features)

        # Prepare data for Streamlit line chart
        prediction_df = pd.DataFrame({
            'datetime': co2_measurements['datetime'][features.index],
            'predicted_occupancy_room_specific': room_specific_predictions.flatten(),
            'predicted_occupancy_general': complete_predictions.flatten()
        }).set_index('datetime')

        # Plot the results using Streamlit line chart
        st.line_chart(prediction_df)

        # Live Prediction
        st.subheader("Live Prediction")
        latest_co2_measurement = co2_measurements.iloc[-1]
        st.write(f"Latest CO2 measurement for the room: {latest_co2_measurement['value']} at {latest_co2_measurement['datetime']}")

        # Prepare latest data for prediction
        latest_features = features.iloc[[-1]].copy()

        # Predict occupancy based on the latest CO2 measurement using room-specific model
        latest_room_specific_prediction = comodel_room.load_and_predict(selected_room_id, latest_features)
        st.write(f"Predicted number of people in the room based on the latest CO2 measurement (Room Specific): {latest_room_specific_prediction.flatten()[0]}")

        # Predict occupancy based on the latest CO2 measurement using complete model
        latest_complete_prediction = comodel_complete.load_and_predict(latest_features)
        st.write(f"Predicted number of people in the room based on the latest CO2 measurement (General): {latest_complete_prediction.flatten()[0]}")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Room Management", "Measurement Data", "ML Models", "Prediction Plot"])

if page == "Room Management":
    st.title("Room Management")
    room_management()
elif page == "Measurement Data":
    st.title("Measurement Data")
    measurement_data()
elif page == "ML Models":
    st.title("ML Models")
    ml_models()
elif page == "Prediction Plot":
    st.title("Prediction Plot")
    prediction_plot()
