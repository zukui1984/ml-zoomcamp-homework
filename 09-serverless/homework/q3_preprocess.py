#### Answer:
Original first pixel [R,G,B]: [ 61. 104.  22.]
After preprocessing [R,G,B]: [-1.073294   -0.21498597 -1.4210021 ]
(R channel): -1.0733
#####

import numpy as np
from images import download_image, prepare_image

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(url)
img = prepare_image(img, target_size=(200, 200))

# Convert to array
x = np.array(img, dtype='float32')
print(f"Original first pixel [R,G,B]: {x[0, 0]}")

# Apply ImageNet preprocessing
x = x / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype='float32').reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype='float32').reshape(1, 1, 3)
x = (x - mean) / std

print(f"After preprocessing [R,G,B]: {x[0, 0]}")
print(f"(R channel): {x[0, 0, 0]:.4f}")

