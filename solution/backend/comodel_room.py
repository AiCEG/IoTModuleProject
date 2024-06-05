import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.legacy import Adam
from keras.models import load_model

def generate_ml_model(room_id):
    # MongoDB connection
    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    db = client["IoTCo2"]

    # Retrieve data from MongoDB
    co2_data = list(db.measurement_data.find({"type": "co2"}))
    camera_data = list(db.measurement_data.find({"type": "camera"}))
    room_data = list(db.rooms.find({"_id": room_id}))

    # Debug: Verify document counts
    print(f"Total CO2 documents: {len(co2_data)}")
    print(f"Total Camera documents: {len(camera_data)}")
    print(f"Total Room documents: {len(room_data)}")

    # Convert to pandas DataFrames
    co2_df = pd.DataFrame(co2_data)
    camera_df = pd.DataFrame(camera_data)
    room_df = pd.DataFrame(room_data)

    # Debug: Initial data inspection
    print("Initial CO2 DataFrame shape:", co2_df.shape)
    print("Initial Camera DataFrame shape:", camera_df.shape)
    print("Initial Room DataFrame shape:", room_df.shape)

    # Convert datetime fields to pandas datetime objects
    co2_df['datetime'] = pd.to_datetime(co2_df['datetime'])
    camera_df['datetime'] = pd.to_datetime(camera_df['datetime'])
    room_df['start_measurements'] = pd.to_datetime(room_df['start_measurements'])
    room_df['end_measurements'] = pd.to_datetime(room_df['end_measurements'])

    # Debug: Datetime conversion check
    print("CO2 datetime values:", co2_df['datetime'].head())
    print("Camera datetime values:", camera_df['datetime'].head())
    print("Room datetime ranges:", room_df[['start_measurements', 'end_measurements']].head())

    # Function to safely extract the nested 'value' key
    def extract_camera_value(val):
        if isinstance(val, dict) and 'value' in val:
            return val['value']
        return np.nan

    # Apply the function to extract values
    camera_df['value'] = camera_df['value'].apply(extract_camera_value)

    # Filter out non-numeric values in the 'value' fields
    co2_df = co2_df[pd.to_numeric(co2_df['value'], errors='coerce').notnull()]
    camera_df = camera_df[pd.to_numeric(camera_df['value'], errors='coerce').notnull()]

    # Debug: Data after filtering non-numeric values
    print("Filtered CO2 DataFrame shape:", co2_df.shape)
    print("Filtered Camera DataFrame shape:", camera_df.shape)

    # First, ensure datetime fields are sorted
    co2_df.sort_values(by='datetime', inplace=True)
    camera_df.sort_values(by='datetime', inplace=True)

    # Debug: Datetime sorting check
    print("Sorted CO2 datetime values:", co2_df['datetime'].head())
    print("Sorted Camera datetime values:", camera_df['datetime'].head())

    # Merge the DataFrames based on the nearest timestamp with a tolerance
    combined_df = pd.merge_asof(co2_df, camera_df, on='datetime', direction="nearest")
    print("Combined DataFrame shape teststest:", combined_df[:5])
    # Drop rows where merge did not find a match within the tolerance
    combined_df.dropna(subset=['value_x', 'value_y'], inplace=True)

    # Simplify the columns for clarity and ensure the correct column names
    combined_df.rename(columns={'value_x': 'co2_value', 'value_y': 'camera_value'}, inplace=True)

    # Debug: Combined DataFrame after merge
    print("Combined DataFrame shape:", combined_df.shape)
    print("Combined DataFrame columns:", combined_df.columns)
    print("Combined DataFrame sample:", combined_df.head())

    # Function to map room data based on datetime range
    def map_room_data(row, room_df):
        for _, room in room_df.iterrows():
            if room['start_measurements'] <= row['datetime'] <= room['end_measurements']:
                return pd.Series({'room_name': room['room_name'], 'm3': room['m3'], 'm2': room['m2']})
        return pd.Series({'room_name': np.nan, 'm3': np.nan, 'm2': np.nan})

    # Apply the function to the combined dataframe
    combined_df[['room_name', 'm3', 'm2']] = combined_df.apply(map_room_data, axis=1, room_df=room_df)

    # Remove rows without room_name
    combined_df.dropna(subset=['room_name'], inplace=True)

    # Debug: Final Combined DataFrame
    print("Final Combined DataFrame shape:", combined_df.shape)
    print("Final Combined DataFrame sample:", combined_df.head())

    # Check if any document count mismatch is present
    print(f"Number of rows in the final combined data:")
    print(combined_df[:5])

    # Calculate changes in CO2 over different time intervals
    combined_df.set_index('datetime', inplace=True)
    combined_df['co2_change_30s'] = combined_df['co2_value'].diff(periods=3)
    combined_df['co2_change_60s'] = combined_df['co2_value'].diff(periods=6)
    combined_df['co2_change_300s'] = combined_df['co2_value'].diff(periods=30)
    combined_df['co2_change_900s'] = combined_df['co2_value'].diff(periods=90)

    combined_df['co2_change_30s_rel'] = combined_df['co2_change_30s'] / combined_df['co2_value'].shift(3)
    combined_df['co2_change_60s_rel'] = combined_df['co2_change_60s'] / combined_df['co2_value'].shift(6)
    combined_df['co2_change_300s_rel'] = combined_df['co2_change_300s'] / combined_df['co2_value'].shift(30)
    combined_df['co2_change_900s_rel'] = combined_df['co2_change_900s'] / combined_df['co2_value'].shift(90)


    combined_df.reset_index(inplace=True)


    #remove columns from combined_df
    combined_df.drop(columns=['_id_x', '_id_y', 'device_id_x', 'device_id_y', 'type_y', 'type_x', 'datetime', 'room_name'], inplace=True)
    combined_df.dropna(subset=['co2_change_900s'], inplace=True)

    print(f"Number of rows in the final combined data with CO2 changes: ")
    print(combined_df.head())

    # model training
    # Split the data into training and testing sets
    features = combined_df[['co2_value', 'm3', 'm2',
                            'co2_change_30s', 'co2_change_60s', 'co2_change_300s', 'co2_change_900s',
                            'co2_change_30s_rel', 'co2_change_60s_rel', 'co2_change_300s_rel', 'co2_change_900s_rel']]
    target = combined_df['camera_value']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the neural network
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Define callbacks
    model_checkpoint = ModelCheckpoint('room_occupancy_model.h5', monitor='val_loss', save_best_only=True, save_format='tf')
    model.summary()
    # Train the model
    print("Training the model...")
    print(X_train_scaled.shape)
    print(y_train.shape)
    history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                        epochs=100, batch_size=32)


    # Evaluate the model
    loss = model.evaluate(X_test_scaled, y_test)
    print(f'Test Loss: {loss}')

    # Predict the number of people in the room
    y_pred = model.predict(X_test_scaled)

    # Compare predictions with actual values
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
    print(results.head(20))

    model.save(f'room_occupancy_model_final_{room_id}.h5')
    print(f"Model saved as 'room_occupancy_model{room_id}.h5'")


def load_and_predict(room_id, new_data):
    # Load the saved model
    #model_path = f'room_occupancy_model_final_{room_id}.keras'
    #model = tf.keras.models.load_model(model_path)

    new_model = load_model(f'room_occupancy_model_final_{room_id}.h5')

    # Ensure new_data has the same structure as the training data
    # Normalize the new data using the same scaler used for training data
    scaler = StandardScaler()
    new_data_scaled = scaler.fit_transform(new_data)

    # Make predictions
    predictions = new_model.predict(new_data_scaled)

    return predictions
