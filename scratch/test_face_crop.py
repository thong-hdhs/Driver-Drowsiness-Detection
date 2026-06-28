import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(os.path.join('Model', 'models', 'drowsiness_model.h5'))
class_labels = ['Closed', 'Open', 'yawn', 'no_yawn']
face_cascade = cv2.CascadeClassifier(r'Model\haar cascade files\haarcascade_frontalface_alt.xml')

def test_face_crop(img_path):
    img = cv2.imread(img_path)
    if img is None: return "Fail"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 1) # relaxed
    if len(faces) == 0:
        return "No face found"
    
    # Take largest face
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    
    # Predict
    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    
    pred = model.predict(face_img, verbose=0)
    pred_class = np.argmax(pred)
    prob_str = ", ".join([f"{class_labels[i]}: {pred[0][i]:.4f}" for i in range(len(class_labels))])
    return f"{class_labels[pred_class]} ({prob_str})"

print("Yawn 1.jpg:", test_face_crop('archive/dataset_new/train/yawn/1.jpg'))
print("No Yawn 1.jpg:", test_face_crop('archive/dataset_new/train/no_yawn/1.jpg'))
