from Detection.find_plate import crop_plate
from ocr import ocr

def img_to_text(img):
    crop = crop_plate(img)
    text = ocr(crop)
    return text
    


test_img = './test_img/KA0330HC.jpg'

res = img_to_text(test_img)
print(res)