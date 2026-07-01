"""
Driver Drowsiness Detection - Multi-threaded Architecture
=========================================================
Kien truc:
  Main Thread : doc camera -> day frame vao Queue -> ve UI (60 FPS)
  AI   Thread : lay frame moi nhat tu Queue -> Haar Cascade -> model() -> ghi ket qua
"""

import cv2
import os
import numpy as np
import threading
import queue
from tensorflow.keras.models import load_model

# ============================================================
#  KHOI TAO TAI NGUYEN
# ============================================================
MODEL_PATH  = os.path.join('Model', 'models', 'drowsiness_model.h5')
IMG_SIZE    = 64          # Khop dung kich thuoc luc train
SCORE_LIMIT = 15          # Nguong canh bao buon ngu
YAWN_CONF   = 0.5         # Nguong tin cay de bao ngap

model = load_model(MODEL_PATH)

# Warm-up model: xoa do tre luc goi dau tien
_warmup = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype='float32')
model(_warmup, training=False)
print("[INFO] Model da san sang (warm-up xong).")

# Haar Cascade
CASCADE_DIR  = r'Model\haar cascade files'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
leye_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, 'haarcascade_lefteye_2splits.xml'))
reye_cascade = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, 'haarcascade_righteye_2splits.xml'))

# Am thanh coi
use_sound = False
sound     = None
try:
    from pygame import mixer
    mixer.init()
    sound     = mixer.Sound('Model/alarm.wav')
    use_sound = True
except Exception:
    pass

# ============================================================
#  SHARED STATE (Queue + Lock)
# ============================================================
frame_q     = queue.Queue(maxsize=1)   # Main -> AI  (chi frame moi nhat)
result_lock = threading.Lock()         # Bao ve result dict
stop_event  = threading.Event()        # Tin hieu dung sach

result = {
    'eye_state':      'Open',
    'yawn_state':     'no_yawn',
    'state_detected': 'Active',
    'score':          0,
    'faces':          [],
    'eye_boxes':      [],
    'thicc':          2,
}

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def preprocess(crop):
    """Tien xu ly anh crop giong het luc train."""
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))          # Resize 64x64
    img = img.astype('float32') / 255.0                   # Chuan hoa [0,1]
    img = np.expand_dims(img, axis=0)                     # -> (1,64,64,3)
    return img

def ai_thread():
    local_score = 0

    while not stop_event.is_set():
        # --- Lay frame moi nhat ---
        frame = None
        while not frame_q.empty():
            try:
                frame = frame_q.get_nowait()
            except queue.Empty:
                break

        if frame is None:
            stop_event.wait(timeout=0.005)
            continue

        # ==========================================================
        # [TỐI ƯU ÁNH SÁNG YẾU TOÀN DIỆN]
        # Chuyen sang khong gian mau LAB de chi can bang kenh Do Sang (L)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Ap dung CLAHE de kich sang vung toi
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        
        # Gop lai va chuyen ve BGR
        limg = cv2.merge((cl, a_channel, b_channel))
        enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Chuyen anh DA DUOC LAM SANG sang xam cho Haar Cascade
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

        eye_state  = 'Open'
        yawn_state = 'no_yawn'
        best_face  = None
        eye_boxes  = []

        # ============================================================
        # BUOC 1: TÌM KHUÔN MẶT ĐỂ KHOANH VÙNG TÌM MẮT (LỌC NHIỄU NỀN)
        # ============================================================
        all_faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 80)
        )

        if len(all_faces) > 0:
            best_face = max(all_faces, key=lambda f: f[2] * f[3])
            fx, fy, fw, fh = best_face
            
            # Cắt lấy NỬA TRÊN khuôn mặt để tìm mắt
            top_y1 = fy
            top_y2 = fy + fh // 2
            top_x1 = fx
            top_x2 = fx + fw
            
            roi_gray_top = gray[top_y1:top_y2, top_x1:top_x2]
            
            # Lấy tất cả mắt trái và mắt phải TRONG nửa trên khuôn mặt
            left_eyes = leye_cascade.detectMultiScale(roi_gray_top, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            right_eyes = reye_cascade.detectMultiScale(roi_gray_top, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            
            all_eyes = list(left_eyes) + list(right_eyes)

            if len(all_eyes) > 0:
                eye_state = 'Closed' # Giả định ban đầu là nhắm, chừng nào thấy 1 mắt mở thì lật lại thành Open

                for (ex, ey, ew, eh) in all_eyes:
                    abs_x = top_x1 + ex
                    abs_y = top_y1 + ey
                    
                    # CẮT SÁT (KHÔNG PADDING)
                    eye_crop = enhanced_frame[abs_y:abs_y+eh, abs_x:abs_x+ew]
                    if eye_crop.size == 0:
                        continue
                        
                    inp  = preprocess(eye_crop)
                    pred = model(inp, training=False).numpy()
                    
                    closed_prob = pred[0][0]
                    open_prob   = pred[0][1]
                    
                    # Ngưỡng tin cậy (Confidence Threshold)
                    if closed_prob > 0.75:
                        local_eye = 'Closed'
                    else:
                        local_eye = 'Open'
                        eye_state = 'Open' # CHỈ CẦN 1 MẮT MỞ LÀ XÁC NHẬN TÀI XẾ ĐANG THỨC
                        
                    prob_str = f"C:{closed_prob:.2f} O:{open_prob:.2f}"
                    eye_boxes.append((abs_x, abs_y, ew, eh, prob_str, local_eye))

        # ============================================================
        # BUOC 2: DOAN NGAP (TẠM THỜI COMMENT ĐỂ TEST MẮT)
        # ============================================================
        # if len(all_faces) > 0:
        #     # CAT ANH TU FRAME DA DUOC LAM SANG
        #     face_crop = enhanced_frame[fy:fy+fh, fx:fx+fw]
        #     if face_crop.size > 0:
        #         inp  = preprocess(face_crop)
        #         pred = model(inp, training=False).numpy()
        #         # index 2=yawn, index 3=no_yawn
        #         if pred[0][2] > pred[0][3] and pred[0][2] > YAWN_CONF:
        #             yawn_state = 'yawn'

        # Tong hop trang thai (Chi quan tam MAT)
        if eye_state == 'Closed':
            state_detected = 'Closed'
        else:
            state_detected = 'Active'

        # Tinh score cuc bo (khong can lock)
        # CƠ CHẾ DECAY SCORING: Tang khi nham, giam tu tu khi mo
        if eye_state == 'Closed':
            local_score += 1
        else:
            local_score -= 1
            if local_score < 0:
                local_score = 0

        # ★ Lock GIU < 1ms: chi ghi ket qua dau ra ★
        with result_lock:
            result['eye_state']      = eye_state
            result['yawn_state']     = yawn_state
            result['state_detected'] = state_detected
            result['score']          = local_score
            result['faces']          = [best_face.tolist()] if best_face is not None else []
            result['eye_boxes']      = eye_boxes


# ============================================================
#  LUONG 1 – MAIN THREAD
# ============================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Khong mo duoc camera!")
        return

    worker = threading.Thread(target=ai_thread, daemon=True, name='AI-Thread')
    worker.start()
    print("[INFO] AI Thread da khoi dong. Nhan 'q' de thoat.")

    alarm_playing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame_q.put_nowait(frame.copy())
        except queue.Full:
            pass

        # Doc ket qua AI (lock <1 ms)
        with result_lock:
            eye_state      = result['eye_state']
            yawn_state     = result['yawn_state']
            state_detected = result['state_detected']
            score          = result['score']
            faces          = result['faces']
            eye_boxes      = result.get('eye_boxes', [])
            thicc          = result['thicc']

        height, width = frame.shape[:2]

        # ---- Ve UI len frame ----
        cv2.rectangle(frame, (0, height - 60), (340, height), (0, 0, 0), cv2.FILLED)

        color = (0, 0, 255) if eye_state == 'Closed' else (0, 255, 0)
        cv2.putText(frame, f'State: {state_detected}', (10, height - 40), font, 1, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f'Score: {score}',          (10, height - 15), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Bounding box khuon mat
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if yawn_state == 'yawn':
                cv2.putText(frame, 'YAWNING!', (x, max(y - 10, 20)), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            
        # Bounding box cho TUNG MAT kem XAC SUAT
        for (x, y, w, h, prob_str, local_eye) in eye_boxes:
            ecolor = (0, 0, 255) if local_eye == 'Closed' else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), ecolor, 2)
            cv2.putText(frame, local_eye, (x, y - 25), font, 0.8, ecolor, 1, cv2.LINE_AA)
            cv2.putText(frame, prob_str, (x, y - 5), font, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        # Canh bao do buon ngu (score > nguong)
        if score > SCORE_LIMIT:
            with result_lock:
                result['thicc'] = thicc + 2 if thicc < 16 else 2

            if use_sound and not alarm_playing:
                try:
                    sound.play(-1)
                    alarm_playing = True
                except Exception:
                    pass
        elif score == 0:
            if use_sound and alarm_playing:
                try:
                    sound.stop()
                    alarm_playing = False
                except Exception:
                    pass
            with result_lock:
                result['thicc'] = 2

        # Ve vien do neu dang trong trang thai bao dong (chuong dang keu)
        if alarm_playing:
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

        cv2.imshow('Driver Drowsiness Detection | Nhan Q de thoat', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    worker.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    if use_sound and alarm_playing:
        try:
            sound.stop()
        except Exception:
            pass
    print("[INFO] Da thoat chuong trinh.")


if __name__ == '__main__':
    main()