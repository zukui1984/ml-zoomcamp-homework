## Question nr 6 - Lambda

import numpy as np
import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image

# Model 
MODEL_PATH = "hair_classifier_empty.onnx"

# Load model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    # Download and resize
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))
    
    # Convert to array
    x = np.array(img, dtype='float32')
    
    # ImageNet preprocessing 
    x = x / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 1, 3)
    x = (x - mean) / std
    
    # Transpose and add batch dimension
    x = np.transpose(x, (2, 0, 1))
    X = np.expand_dims(x, axis=0)
    
    # Predict
    pred = session.run([output_name], {input_name: X})[0]
    return float(pred[0, 0])

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    
    return {
        'statusCode': 200,
        'body': {
            'prediction': result
        }
    }

