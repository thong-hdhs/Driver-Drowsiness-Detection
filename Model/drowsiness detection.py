import cv2
import os
import numpy as np
import time
from tensorflow.keras.models import load_model

# Try to load pygame mixer for sound
try:
    from pygame import mixer
    mixer.init()
    sound = mixer.Sound('Model/alarm.wav')
    use_sound = True
except:
    use_sound = False

# 1. Tải file bộ não AI
model_path = os.path.join('Model', 'models', 'drowsiness_model.h5')
model = load_model(model_path)

# 2. Tải các file Haar Cascade để cắt khuôn mặt và mắt

face_cascade = cv2.CascadeClassifier(r'Model\haar cascade files\haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier(r'Model\haar cascade files\haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier(r'Model\haar cascade files\haarcascade_righteye_2splits.xml')

# 4 Nhãn tương ứng với model
# Đảm bảo thứ tự nhãn trùng với thứ tự thư mục lúc train
class_labels = ['Closed', 'Open', 'no_yawn', 'yawn']

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2

while(True):
    ret, frame = cap.read()
    if not ret:
        break
        
    height, width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Phát hiện khuôn mặt và mắt
    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    # Vẽ khung nền đen hiển thị trạng thái ở góc dưới
    cv2.rectangle(frame, (0, height-60), (280, height), (0,0,0), thickness=cv2.FILLED)

    state_detected = "Open" # Mặc định
    
    # Xử lý vùng mắt phải 
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        
        # Tiền xử lý chuẩn ảnh 64x64 
        r_eye = cv2.resize(r_eye, (64, 64))
        r_eye = r_eye.astype('float32') / 255.0
        r_eye = np.expand_dims(r_eye, axis=0)
        
        # Dự đoán nhãn
        prediction = model.predict(r_eye, verbose=0)
        pred_class = np.argmax(prediction)
        state_detected = class_labels[pred_class]
        break

    # Xử lý vùng mắt trái (Nếu mắt phải không tìm thấy thì check mắt trái)
    if state_detected == "Open":
        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y+h, x:x+w]
            
            # Tiền xử lý chuẩn ảnh 64x64 
            l_eye = cv2.resize(l_eye, (64, 64))
            l_eye = l_eye.astype('float32') / 255.0
            l_eye = np.expand_dims(l_eye, axis=0)
            
            # Dự đoán nhãn
            prediction = model.predict(l_eye, verbose=0)
            pred_class = np.argmax(prediction)
            state_detected = class_labels[pred_class]
            break

    # Vẽ khung vuông lên mặt tài xế
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Nếu đang ngáp (yawn) thì hiển thị cảnh báo ngay trên mặt
        if state_detected == 'yawn':
            cv2.putText(frame, "YAWNING!", (x, y-10), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Tính toán điểm cảnh báo dựa trên trạng thái nhắm mắt (Closed) hoặc ngáp (yawn)
    if state_detected == 'Closed' or state_detected == 'yawn':
        score += 1
        color = (0, 0, 255) # Màu đỏ cảnh báo
    else:
        score -= 1
        color = (0, 255, 0) # Màu xanh an toàn
        
    if score < 0:
        score = 0   
        
    # Hiển thị trạng thái hiện tại và Điểm buồn ngủ lên màn hình
    cv2.putText(frame, f"State: {state_detected}", (10, height-40), font, 1, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Score: {score}", (10, height-15), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Nếu điểm vượt quá 15 điểm (Nhắm mắt hoặc ngáp quá lâu) -> Hú còi báo động
    if score > 15:
        if use_sound:
            try:
                sound.play()
            except:
                pass
        # Vẽ viền đỏ nhấp nháy xung quanh màn hình camera
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        thicc = thicc + 2 if thicc < 16 else 2

    cv2.imshow('Driver Drowsiness Detection System', frame)
    
    # Nhường 30ms cho Windows vẽ hình và bắt phím 'q' để thoát
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
    # Bấm phím 'q' để tắt ứng dụng camera