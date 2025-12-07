## Answer: 
Model output: 0.0916
###

import numpy as np
import onnxruntime as ort
from images import download_image, prepare_image

session = ort.InferenceSession("hair_classifier_v1.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(url)
img = prepare_image(img, target_size=(200, 200))

# Convert to array
x = np.array(img, dtype='float32')

# ImageNet preprocessing
x = x / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 1, 3)
x = (x - mean) / std

# Transpose and batch
x = np.transpose(x, (2, 0, 1))
X = np.expand_dims(x, axis=0)

# Predict
output = session.run([output_name], {input_name: X})[0][0, 0]


print(f"Model output: {output:.4f}")
