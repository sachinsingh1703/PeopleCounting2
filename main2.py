import os
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# Load the Faster R-CNN model pre-trained on COCO dataset
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set to evaluation mode

# Define a function to count people in an image
def count_people(image_path, confidence_threshold=0.5):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image)
    
    # Add batch dimension
    input_tensor = image_tensor.unsqueeze(0)
    
    # Perform inference
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Extract bounding boxes, labels, and scores
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']
    
    # Count people based on confidence threshold
    count = 0
    for label, score in zip(labels, scores):
        if label == 1 and score > confidence_threshold:  # Label 1 is for 'person' in COCO dataset
            count += 1
    return count

# Directory containing images
image_dir = 'D:/VisionML/PeopleCounting2/crowd'
output_csv = 'people_count.csv'

# Initialize results list
results = []

# Process each image in the directory
for filename in os.listdir(r'D:\VisionML\PeopleCounting2\crowd'):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust for your image types
        image_path = os.path.join(image_dir, filename)
        people_count = count_people(image_path, confidence_threshold=0.5)
        results.append({'Image Name': filename, 'People Count': people_count})

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")