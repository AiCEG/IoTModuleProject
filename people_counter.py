import subprocess
from datetime import datetime
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.solutions import object_counter

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


def count_people(image_path):
    # Count people in the image
        model = YOLO('yolov8n.pt')  # Load the pretrained YOLOv8 model
        results = model(image_path)
        names = results[0].names    # same as model.names

        # store number of objects detected per class label
        class_detections_values = []
        for k, v in names.items():
            class_detections_values.append(results[0].boxes.cls.tolist().count(k))
        # create dictionary of objects detected per class
        classes_detected = dict(zip(names.values(), class_detections_values))

        return classes_detected['person']



# Path where images will be saved
images_folder = 'captured_images'

# Take a picture
image_path = take_picture(images_folder)
#image_path = "classroom.jpg"

if image_path:
    # Count people in the image
    people_count = count_people(image_path)
    print(f"Number of people in the image: {people_count}")