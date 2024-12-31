import os
import cv2
import pandas as pd
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')  # Replace with the appropriate YOLO model file

# Directory containing images
image_dir = 'D:/VisionML/PeopleCounting2/crowd'
output_csv = 'people_count.csv'

# Initialize results list
results = []

# Process each image in the directory
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust for your image types
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)

        # Perform detection
        detections = model(image)

        # Count people in the image
        count = 0
        for detection in detections:
            for box in detection.boxes:
                if box.conf > 0.5:  # Confidence threshold
                    count += 1

        # Append results
        results.append({'Image Name': filename, 'People Count': count})

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")