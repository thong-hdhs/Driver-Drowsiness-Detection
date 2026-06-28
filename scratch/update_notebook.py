import json

new_code = """import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Khởi tạo Haar Cascade dùng sẵn của OpenCV để dễ dàng chạy trên Colab/Kaggle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 1. Định nghĩa đường dẫn và nhãn
DATA_DIR = "/content/archive/dataset_new/train"  # Kiểm tra lại chính xác đường dẫn thư mục sau khi giải nén
CATEGORIES = ["Closed", "Open", "yawn", "no_yawn"]
IMG_SIZE = 64

data = []
labels = []

# 2. Đọc ảnh và tiền xử lý
print("Bắt đầu xử lý dữ liệu...")
for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    class_num = CATEGORIES.index(category)

    if not os.path.exists(path):
        print(f"Không tìm thấy thư mục: {path}")
        continue
        
    print(f"Đang xử lý thư mục: {category}")
    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            # Đọc ảnh dạng màu (BGR)
            img_array = cv2.imread(img_path)
            if img_array is None:
                continue

            # NẾU LÀ ẢNH NGÁP HOẶC KHÔNG NGÁP -> TÌM VÀ CẮT KHUÔN MẶT
            if category in ["yawn", "no_yawn"]:
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                # Dùng thông số nới lỏng để bắt được các khuôn mặt bị biến dạng khi ngáp
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
                
                if len(faces) > 0:
                    # Ưu tiên lấy khuôn mặt to nhất (tránh nhiễu)
                    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                    x, y, w, h = faces[0]
                    # Cắt lấy vùng khuôn mặt
                    img_array = img_array[y:y+h, x:x+w]

            # Resize ảnh về kích thước chuẩn 64x64
            resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            data.append(resized_array)
            labels.append(class_num)
        except Exception as e:
            pass

# 3. Chuẩn hóa dữ liệu đưa về khoảng [0, 1]
X = np.array(data, dtype="float32") / 255.0
y = np.array(labels)

# One-hot encode nhãn (4 lớp)
y = to_categorical(y, num_classes=4)

# 4. Chia tập dữ liệu: 80% Train, 20% Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Hoàn tất tiền xử lý!")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
"""

with open('train_model.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

# The data preprocessing cell is at index 1
# Split new code by newlines and append \n to each line to match Jupyter format
lines = [line + '\n' for line in new_code.split('\n')]
# remove the trailing \n from the last line to be clean
if lines:
    lines[-1] = lines[-1].rstrip('\n')

data['cells'][1]['source'] = lines

with open('train_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Đã cập nhật train_model.ipynb thành công!")
