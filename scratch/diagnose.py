"""
Diagnostic script: Kiem tra pipeline tung buoc, in xac suat model
tren anh train thuc te de phat hien diem sai.
"""
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

MODEL_PATH  = os.path.join('Model', 'models', 'drowsiness_model.h5')
CASCADE_DIR = r'Model\haar cascade files'
IMG_SIZE    = 64
CLASS_LABELS = ['Closed', 'Open', 'yawn', 'no_yawn']

model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, 'haarcascade_frontalface_alt.xml'))
leye_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, 'haarcascade_lefteye_2splits.xml'))
reye_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, 'haarcascade_righteye_2splits.xml'))

TRAIN_DIR = 'archive/dataset_new/train'

def preprocess(crop):
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict_verbose(img_arr, label):
    inp  = preprocess(img_arr)
    pred = model(inp, training=False).numpy()[0]
    result = {CLASS_LABELS[i]: float(f"{pred[i]:.4f}") for i in range(4)}
    winner = CLASS_LABELS[np.argmax(pred)]
    ok = "OK" if winner.lower() in label.lower() else "WRONG"
    print(f"  [{ok}] Expected={label:<12} Got={winner:<15} Probs={result}")
    return winner

print("="*70)
print("TEST 1: Dua thang anh train vao (khong cat gi) - kiem tra model co hoc dung khong")
print("="*70)

for cat in ['Closed', 'Open', 'yawn', 'no_yawn']:
    cat_dir = os.path.join(TRAIN_DIR, cat)
    imgs = os.listdir(cat_dir)[:3]
    print(f"\n--- {cat} (raw image -> resize 64x64 -> model) ---")
    for name in imgs:
        img = cv2.imread(os.path.join(cat_dir, cat, name) if False else os.path.join(cat_dir, name))
        if img is None: continue
        predict_verbose(img, cat)

print()
print("="*70)
print("TEST 2: Cat khuon mat roi predict (kiem tra yawn pipeline)")
print("="*70)

for cat in ['yawn', 'no_yawn']:
    cat_dir = os.path.join(TRAIN_DIR, cat)
    imgs = os.listdir(cat_dir)[:3]
    print(f"\n--- {cat} (face crop via Haar -> resize 64x64 -> model) ---")
    for name in imgs:
        img = cv2.imread(os.path.join(cat_dir, name))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30,30))
        if len(faces) == 0:
            print(f"  [SKIP] {name}: Khong tim thay khuon mat")
            continue
        fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        face_crop = img[fy:fy+fh, fx:fx+fw]
        predict_verbose(face_crop, cat)

print()
print("="*70)
print("TEST 3: Cat mat roi predict (kiem tra Closed/Open pipeline)")
print("="*70)

for cat in ['Closed', 'Open']:
    cat_dir = os.path.join(TRAIN_DIR, cat)
    imgs = os.listdir(cat_dir)[:3]
    print(f"\n--- {cat} (raw eye image -> resize 64x64 -> model, no Haar) ---")
    for name in imgs:
        img = cv2.imread(os.path.join(cat_dir, name))
        if img is None: continue
        predict_verbose(img, cat)

print()
print("="*70)
print("TEST 4: Kiem tra Haar Cascade co tim thay mat trong anh mat goc khong")
print("="*70)

for cat in ['Closed', 'Open']:
    cat_dir = os.path.join(TRAIN_DIR, cat)
    imgs = os.listdir(cat_dir)[:2]
    print(f"\n--- {cat} eye images, size: ---")
    for name in imgs:
        img = cv2.imread(os.path.join(cat_dir, name))
        if img is None: continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Thu tim face trong anh mat
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        leyes = leye_cascade.detectMultiScale(gray, 1.1, 3)
        reyes = reye_cascade.detectMultiScale(gray, 1.1, 3)
        print(f"  {name}: size={w}x{h}, faces={len(faces)}, leye={len(leyes)}, reye={len(reyes)}")
