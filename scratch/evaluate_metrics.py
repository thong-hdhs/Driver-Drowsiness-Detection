import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# Cấu hình lại stdout để in được tiếng Việt trên Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# Thiết lập đường dẫn tự động dựa trên vị trí của file script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'Model', 'models', 'drowsiness_model.h5')
TEST_DIR = os.path.join(BASE_DIR, 'archive', 'dataset_new', 'test')
WEBCAM_TEST_DIR = os.path.join(BASE_DIR, 'archive', 'dataset_new', 'webcam_test')

# Load Haar Cascades để test TH2 (Webcam)
face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'Model', 'haar cascade files', 'haarcascade_frontalface_alt.xml'))
leye_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'Model', 'haar cascade files', 'haarcascade_lefteye_2splits.xml'))
reye_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'Model', 'haar cascade files', 'haarcascade_righteye_2splits.xml'))

CATEGORIES = ['Closed', 'Open']
IMG_SIZE = 64

def crop_eye_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    if len(faces) == 0:
        return None # Khong tim thay mat
        
    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    
    leyes = leye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
    reyes = reye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
    
    eye_img = None
    if len(leyes) > 0:
        ex, ey, ew, eh = leyes[0]
        eye_img = roi_color[ey:ey+eh, ex:ex+ew]
    elif len(reyes) > 0:
        ex, ey, ew, eh = reyes[0]
        eye_img = roi_color[ey:ey+eh, ex:ex+ew]
        
    return eye_img

def load_and_preprocess_image(filepath, mode="Test Set"):
    img_array = cv2.imread(filepath)
    if img_array is None:
        return None
        
    # NẾU LÀ ẢNH TỪ WEBCAM (TH2) -> PHẢI TÌM VÀ CẮT LẤY CON MẮT TRƯỚC
    if mode == "Webcam Frames":
        eye_crop = crop_eye_from_frame(img_array)
        if eye_crop is not None and eye_crop.size > 0:
            img_array = eye_crop
        else:
            print(f"  [Cảnh báo] Không tìm thấy mắt trong ảnh {os.path.basename(filepath)} -> Bỏ qua ảnh này!")
            return None
            
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    resized_array = resized_array / 255.0
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

def evaluate_on_directory(model, data_dir, mode_name="Test Set"):
    if not os.path.exists(data_dir):
        print(f"\n[LỖI] Không tìm thấy thư mục {data_dir} cho chế độ {mode_name}")
        return

    print(f"\n{'='*50}\nBẮT ĐẦU ĐÁNH GIÁ TRÊN: {mode_name}\n{'='*50}")
    
    y_true = []
    y_pred = []
    
    for category in CATEGORIES:
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            continue
            
        class_idx = CATEGORIES.index(category)
        
        print(f"Đang quét thư mục: {category}...")
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            
            inp = load_and_preprocess_image(img_path, mode=mode_name)
            if inp is None: continue
            
            # Dự đoán - Chỉ lấy xác suất của 2 nhãn đầu tiên (Closed và Open)
            pred_probs = model.predict(inp, verbose=0)[0]
            pred_idx = np.argmax(pred_probs[:2])
            
            y_true.append(class_idx)
            y_pred.append(pred_idx)
            
    if len(y_true) == 0:
        print("Không có ảnh nào để đánh giá!")
        return

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print("\n--- KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN ---")
    print(f"Accuracy (Độ chính xác) : {acc*100:.2f}%")
    print(f"Precision (Độ chuẩn xác): {precision*100:.2f}%")
    print(f"Recall (Độ nhạy)        : {recall*100:.2f}%")
    print(f"F1-Score                : {f1*100:.2f}%\n")
    
    print("--- BÁO CÁO CHI TIẾT THEO TỪNG NHÃN (Classification Report) ---")
    print(classification_report(y_true, y_pred, target_names=CATEGORIES, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'Ma Trận Nhầm Lẫn (Confusion Matrix) - {mode_name}')
    plt.ylabel('Nhãn Thực Tế (Ground Truth)')
    plt.xlabel('Nhãn Dự Đoán (Predicted)')
    plt.tight_layout()
    
    save_path = f'confusion_matrix_{mode_name.replace(" ", "_")}.png'
    plt.savefig(save_path)
    print(f"Đã lưu biểu đồ ma trận nhầm lẫn tại: {save_path}")
    plt.show()

if __name__ == "__main__":
    print("Đang tải model...")
    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Không thể tải model. Lỗi: {e}")
        exit()

    print("Model đã được tải thành công!")
    
    # TH1: Đánh giá qua tập test
    evaluate_on_directory(model, TEST_DIR, "Test Set")
    
    # TH2: Đánh giá qua ảnh lấy từ Webcam
    evaluate_on_directory(model, WEBCAM_TEST_DIR, "Webcam Frames")
