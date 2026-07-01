"""
Debug real-time: Chay camera 5 giay, in xac suat yawn/no_yawn
cua model tren TUNG FRAME de xem model dang bao gi.
"""
import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model

MODEL_PATH  = os.path.join('Model', 'models', 'drowsiness_model.h5')
CASCADE_DIR = r'Model\haar cascade files'
IMG_SIZE    = 64

model = load_model(MODEL_PATH)
_w = np.zeros((1,64,64,3), dtype='float32')
model(_w, training=False)

face_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, 'haarcascade_frontalface_alt.xml'))

def preprocess(crop):
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong mo duoc camera!")
    exit()

print("="*80)
print("DEBUG: Mo mieng binh thuong (KHONG NGAP) trong 5 giay dau")
print("       Sau do HA MIENG TO (NGAP) trong 5 giay tiep theo")
print("="*80)
print()
print(f"{'Frame':>6} | {'Face?':>5} | {'FaceSize':>10} | {'Closed':>8} | {'Open':>8} | {'yawn':>8} | {'no_yawn':>8} | {'Result':>12}")
print("-"*90)

start = time.time()
frame_count = 0

while time.time() - start < 10:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 5 != 0:  # In moi 5 frame de tranh spam
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Tim khuon mat lon nhat
    all_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
    
    if len(all_faces) == 0:
        print(f"{frame_count:>6} | {'NO':>5} | {'---':>10} | {'---':>8} | {'---':>8} | {'---':>8} | {'---':>8} | {'no face':>12}")
        continue
    
    best = max(all_faces, key=lambda f: f[2]*f[3])
    fx, fy, fw, fh = best
    face_crop = frame[fy:fy+fh, fx:fx+fw]
    
    inp = preprocess(face_crop)
    pred = model(inp, training=False).numpy()[0]
    
    winner = ['Closed','Open','yawn','no_yawn'][np.argmax(pred)]
    
    print(f"{frame_count:>6} | {'YES':>5} | {fw}x{fh:>4} | {pred[0]:>8.4f} | {pred[1]:>8.4f} | {pred[2]:>8.4f} | {pred[3]:>8.4f} | {winner:>12}")

cap.release()
print()
print("="*80)
print("DONE. Xem cot 'yawn' va 'no_yawn' de biet model dang nghieng ve ben nao.")
print("Neu yawn luon > 0.5 du khong ngap => Model bi overfit hoac face crop khong khop.")
print("="*80)
