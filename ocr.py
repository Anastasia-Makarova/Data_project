import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from ultralytics import YOLO
import numpy as np
import gdown
import os

def crop_plate(img_path):
    model_path = './Detection/runs/detect/train/weights/best.pt' 
    model = YOLO(model_path)
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
    return None  # Возврат None, если номерные знаки не найдены


def download_model_from_google_drive(file_id, destination_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination_path, quiet=False)


def ocr(img_path):
    # Загрузка модели и процессора TrOCR
    model_path = "./OCR/results/trocr-finetuned"

    try:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
    except OSError:
        ocr_model_id = '1g_9QaTWuIqa7-fdp4r5Bqx_UoGuJnqKA'
        ocr_model_path = './OCR/results/trocr-finetuned'

        download_model_from_google_drive(ocr_model_id, ocr_model_path)
        
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)


    # Вызов функции для обрезки номерного знака
    cropped_image_array = crop_plate(img_path)
    
    if cropped_image_array is None:
        return "No license plate detected"

    # Преобразование обрезанного массива numpy обратно в изображение PIL
    cropped_image = Image.fromarray(cropped_image_array)

    # Предобработка изображения для модели
    pixel_values = processor(images=cropped_image, return_tensors="pt").pixel_values

    # Генерация текста
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    # Декодирование сгенерированного текста
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

# Пример использования:
img_path = "./test_img/BK1560II.jpg"  # Замените на путь к вашему изображению
print(ocr(img_path))
