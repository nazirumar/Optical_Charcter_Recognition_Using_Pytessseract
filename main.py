import os
import requests
from PIL import Image
import cv2
import pytesseract
import re
from tqdm import tqdm
from urllib.parse import urlparse, unquote


def get_image_name_from_url(url):
    parsed_url = urlparse(url)
    image_name = os.path.basename(parsed_url.path)
    image_name= unquote(image_name)
    if not os.path.splitext(image_name)[1]:
        image_name += '.png'
    return image_name


sample_path = r'/home/nazk33r/Documents/Opencv/Optical_Charcter_Recognition_Using_Pytessseract/sample_image/'
sample_output_path =r'/home/nazk33r/Documents/Opencv/Optical_Charcter_Recognition_Using_Pytessseract/sample_text/'
if not os.path.exists(sample_path):
    os.makedirs(sample_path)

image_url = str(input("Please enter a sample image URL:\n"))
image_name = get_image_name_from_url(image_url)
sample_image_path = os.path.join(sample_path, image_name)

if not os.path.exists(sample_image_path):
    try:
        response = requests.get(image_url, stream=True)

        total_size = int(response.headers.get('content-length', 0))
        if response.status_code == 200:
            with open(sample_image_path, 'wb') as f:
                with tqdm(
                    desc='Downloading sample image',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)
            print('Sample image downloaded and saved.')
    except requests.exceptions.RequestException as e:
        print(f'Error occurred while downloading sample image: {e}')
else:
    image = Image.open(sample_image_path)
    image = image.resize((300, 150))  # Correctly resize the image if needed
    image.save(sample_image_path)  # Save the resized image

    # Load image in grayscale
    img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply thresholding to binarize the image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply OCR to recognize text from the image
    text = pytesseract.image_to_string(img, lang='eng', config='--oem 3 --psm 6')

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text_file_name = os.path.splitext(image_name)[0] + '.txt'
    print(text_file_name)

    with open(f'{sample_output_path}{text_file_name}', 'w') as file:
        file.write(text)
    print(text)
