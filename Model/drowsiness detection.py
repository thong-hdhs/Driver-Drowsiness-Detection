import cv2
import os
import numpy as np
import time

# Try to load pygame mixer for sound
try:
    from pygame import mixer
    mixer.init()
    sound = mixer.Sound('alarm.wav')
    use_sound = True
except:
    use_sound = False

face = cv2.CascadeClassifier(r'haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

# Simple eye closure detection function (no Keras needed)
def detect_eye_closed(eye_region):
    """Detect if eye is closed using adaptive thresholding"""
    if eye_region is None or eye_region.size == 0:
        return 1
    
    h, w = eye_region.shape[:2]
    if h < 5 or w < 5:
        return 1
    
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY) if len(eye_region.shape) == 3 else eye_region
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Histogram analysis - high dark pixels = closed eye
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (h * w)
    dark_ratio = np.sum(hist[:80])  # pixels with intensity < 80
    
    # Laplacian variance - low variance = blurry = closed
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Return 0 (closed) or 1 (open)
    if dark_ratio > 0.55 and laplacian_var < 80:
        return 0
    return 1
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        rpred = detect_eye_closed(r_eye)
        if(rpred==1):
            lbl='Open' 
        if(rpred==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
