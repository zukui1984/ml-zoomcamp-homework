## Answer
Full input shape: ['s77', 3, 200, 200]
Format: [batch, channels, height, width]
Target size: 200x200
###


import onnxruntime as ort

session = ort.InferenceSession("hair_classifier_v1.onnx")
input_shape = session.get_inputs()[0].shape

print(f"Full input shape: {input_shape}")

if isinstance(input_shape[1], int) and input_shape[1] == 3:
    height = input_shape[2]
    width = input_shape[3]
    print(f"Format: [batch, channels, height, width]")
elif isinstance(input_shape[3], int) and input_shape[3] == 3:
    height = input_shape[1]
    width = input_shape[2]
    print(f"Format: [batch, height, width, channels]")
else:
    height = input_shape[-2]
    width = input_shape[-1]
    
print(f"Target size: {height}x{width}")

