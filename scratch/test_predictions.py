import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_path = os.path.join('Model', 'models', 'drowsiness_model.h5')
if not os.path.exists(model_path):
    print("Model file not found!")
    exit(1)

model = load_model(model_path)
# ĐÚNG THỨ TỰ TRONG train_model.ipynb
class_labels = ["Closed", "Open", "yawn", "no_yawn"]

def test_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "Failed to load"
    img_resized = cv2.resize(img, (64, 64))
    img_normalized = img_resized.astype('float32') / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    pred = model.predict(img_expanded, verbose=0)
    pred_class = np.argmax(pred)
    return class_labels[pred_class], pred[0]

categories = ["Closed", "Open", "yawn", "no_yawn"]
for cat in categories:
    cat_dir = os.path.join('archive', 'dataset_new', 'train', cat)
    if os.path.exists(cat_dir):
        files = os.listdir(cat_dir)[:3]
        print(f"\n--- Category: {cat} ---")
        for f in files:
            path = os.path.join(cat_dir, f)
            label, probs = test_image(path)
            prob_str = ", ".join([f"{class_labels[i]}: {probs[i]:.4f}" for i in range(len(class_labels))])
            print(f"{f}: predicted as {label} ({prob_str})")
