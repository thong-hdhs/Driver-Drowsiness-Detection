import cv2
import os

# Cấu hình đường dẫn lưu ảnh
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBCAM_DIR = os.path.join(BASE_DIR, 'archive', 'dataset_new', 'webcam_test')
CLOSED_DIR = os.path.join(WEBCAM_DIR, 'Closed')
OPEN_DIR = os.path.join(WEBCAM_DIR, 'Open')

# Tạo thư mục nếu chưa có
os.makedirs(CLOSED_DIR, exist_ok=True)
os.makedirs(OPEN_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

print("="*50)
print("📸 CÔNG CỤ CHỤP ẢNH TỪ WEBCAM ĐỂ TEST TH2")
print("="*50)
print("Hướng dẫn:")
print("- Nhấn phím 'c' để chụp và lưu vào thư mục CLOSED (Nhắm mắt)")
print("- Nhấn phím 'o' để chụp và lưu vào thư mục OPEN (Mở mắt)")
print("- Nhấn phím 'q' để thoát")
print("="*50)

closed_count = len(os.listdir(CLOSED_DIR))
open_count = len(os.listdir(OPEN_DIR))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể mở webcam!")
        break
        
    # Hiện hướng dẫn lên màn hình camera
    cv2.putText(frame, f"Closed: {closed_count} (Press 'c')", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Open: {open_count} (Press 'o')", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Thu thap du lieu TH2 (Webcam)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key in [ord('c'), ord('C')]:
        # Lưu vào mục Closed
        img_path = os.path.join(CLOSED_DIR, f"webcam_closed_{closed_count}.jpg")
        cv2.imwrite(img_path, frame)
        closed_count += 1
        print(f"Đã lưu: {img_path}")
    elif key in [ord('o'), ord('O')]:
        # Lưu vào mục Open
        img_path = os.path.join(OPEN_DIR, f"webcam_open_{open_count}.jpg")
        cv2.imwrite(img_path, frame)
        open_count += 1
        print(f"Đã lưu: {img_path}")
    elif key in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()
print("\nĐã thoát. Bây giờ bạn có thể chạy lại lệnh 'python scratch/evaluate_metrics.py' để xem kết quả TH2 nhé!")
