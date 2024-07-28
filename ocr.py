import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def preprocess(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    noise_filter = cv2.bilateralFilter(gray, 10, 15, 15)
    return noise_filter

def ocr(img_array):
    preprocessed_image = preprocess(img_array)

    # Convert numpy array to PIL Image
    image = Image.fromarray(preprocessed_image)

    # To RGB
    image = image.convert("RGB")

    # Initialisation
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

    # Preprocessing
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Text generation
    generated_ids = model.generate(pixel_values, max_new_tokens=50)

    # Text decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text
    
if __name__ == '__main__':

    #path = './cleared_cropped_plates'
    path = './test_img/AI6392HM.jpg'
    text = ocr(path)
    print(text)

    ''' count = 0
        correct = 0
        correct_digits = 0

        results =[]
        for file in os.listdir(path)[50:70]:

            count += 1

            file_path = os.path.join(path, file)

            true = file[:-4]
            predict = str(ocr(file_path)).replace(" ", "")[-8:]
            digits = predict[2:-2]

            if true == predict:
                correct += 1
                comparison = 'True'
            elif digits == file[2:6]:
                correct_digits += 1 
                comparison = 'Partial'
            else:
                comparison = 'False'

            results.append(f'True {true}, predicted {predict}           {comparison}')
        
        for r in results:
            print(r)

        print(f'Correct predictions {correct} out of {count} - {correct / count} rate')
        print(f'Correct digits predictions {correct + correct_digits} out of {count} - {(correct + correct_digits) / count} rate')'''
