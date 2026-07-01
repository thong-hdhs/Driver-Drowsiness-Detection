"""
Debug yawn: So sanh xac suat khi dua vao:
  A) Anh train goc (full frame 480x640) -> resize 64x64  (cach model duoc train)
  B) Anh train goc -> cat face crop -> resize 64x64       (cach code realtime dang lam)
Neu B sai nhung A dung => pipeline realtime KHONG KHOP voi pipeline train.
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

TRAIN_DIR = 'archive/dataset_new/train'

def preprocess(crop):
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def pred_str(pred):
    return " | ".join([f"{CLASS_LABELS[i]}:{pred[i]:.4f}" for i in range(4)])

print("="*90)
print("TEST A: Anh train GOC -> resize 64x64 (KHONG cat face) = cach model duoc train")
print("="*90)

for cat in ['yawn', 'no_yawn']:
    cat_dir = os.path.join(TRAIN_DIR, cat)
    imgs = sorted(os.listdir(cat_dir))[:5]
    print(f"\n--- {cat} ---")
    for name in imgs:
        img = cv2.imread(os.path.join(cat_dir, name))
        if img is None: continue
        inp = preprocess(img)
        pred = model(inp, training=False).numpy()[0]
        winner = CLASS_LABELS[np.argmax(pred)]
        ok = "OK" if cat in winner else "WRONG"
        print(f"  [{ok:>5}] {name:<15} -> {winner:<10} | {pred_str(pred)}")

print()
print("="*90)
print("TEST B: Anh train -> CAT FACE (Haar Cascade) -> resize 64x64 = cach code realtime")
print("="*90)

for cat in ['yawn', 'no_yawn']:
    cat_dir = os.path.join(TRAIN_DIR, cat)
    imgs = sorted(os.listdir(cat_dir))[:8]
    print(f"\n--- {cat} ---")
    for name in imgs:
        img = cv2.imread(os.path.join(cat_dir, name))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))
        if len(faces) == 0:
            # Thu relax hon
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30,30))
        
        if len(faces) == 0:
            print(f"  [SKIP ] {name:<15} -> No face found (img size: {img.shape[1]}x{img.shape[0]})")
            continue
        
        best = max(faces, key=lambda f: f[2]*f[3])
        fx, fy, fw, fh = best
        face_crop = img[fy:fy+fh, fx:fx+fw]
        
        inp = preprocess(face_crop)
        pred = model(inp, training=False).numpy()[0]
        winner = CLASS_LABELS[np.argmax(pred)]
        ok = "OK" if cat in winner else "WRONG"
        print(f"  [{ok:>5}] {name:<15} -> {winner:<10} | {pred_str(pred)} | face:{fw}x{fh}")

print()
print("="*90)
print("KET LUAN:")
print("  - Neu TEST A dung nhung TEST B sai => Pipeline realtime KHONG KHOP train")
print("  - Neu ca A va B dung => Van de o camera thuc (anh sang, goc chup, boi canh)")
print("  - Neu ca A va B sai => Model can train lai")
print("="*90)
