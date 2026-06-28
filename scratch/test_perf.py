import os
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

model = load_model(os.path.join('Model', 'models', 'drowsiness_model.h5'))

dummy_input = np.random.rand(1, 64, 64, 3).astype('float32')

# Warm up
model.predict(dummy_input, verbose=0)
model(dummy_input, training=False)

start = time.time()
for _ in range(50):
    model.predict(dummy_input, verbose=0)
end = time.time()
print(f"model.predict() 50 calls: {end - start:.4f} seconds")

start = time.time()
for _ in range(50):
    model(dummy_input, training=False)
end = time.time()
print(f"model(input) 50 calls: {end - start:.4f} seconds")
