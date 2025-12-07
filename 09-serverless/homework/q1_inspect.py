### Answer: output

import onnxruntime as ort

# Load the model
session = ort.InferenceSession("hair_classifier_v1.onnx")

# Input name
input_name = session.get_inputs()[0].name
print(f"Input name: {input_name}")

# Output name
output_name = session.get_outputs()[0].name
print(f"Output name: {output_name}")

