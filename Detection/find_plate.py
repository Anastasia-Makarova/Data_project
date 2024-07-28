from ultralytics import YOLO
from roboflow import Roboflow
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch


#pretrained model weights
model_path = './Detection/runs/detect/train/weights/best.pt' 

model = YOLO(model_path)

def crop_plate(img_path):
    image = Image.open(img_path)
    results = model(img_path)
    
    for res in results:
        for box in res.boxes:
            confidence = box.conf[0].item()  # Get confidence score
            if confidence >= 0.7:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                cropped_image = image.crop((x1, y1, x2, y2))  # Crop the region
                cropped_array = np.array(cropped_image)  # Convert to numpy array
                return cropped_array
    return None  # Return None if no plates are detected
