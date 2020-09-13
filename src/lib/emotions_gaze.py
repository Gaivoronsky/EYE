import cv2
import os
from datetime import datetime
import numpy as np
import time
import asyncio

from model_lib.model import model_face
from gaze_tracking import GazeTracking

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class stream_video():
    def __init__(self):
        self.model = model_face()
        self.model.load_weights('model/model.h5')

        self.gaze = GazeTracking()
        
        # cv2.ocl.setUseOpenCL(True)
                
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        self.cap = cv2.VideoCapture(0)
        self.facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
    def take_frame(self):
        _, self.frame = self.cap.read()

        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(self.frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = self.model.predict(cropped_img)

            maxindex = int(np.argmax(prediction))
            cv2.putText(self.frame, self.emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def eye_construction(self):
        self.gaze.refresh(self.frame)
        self.frame = self.gaze.annotated_frame()

        text = ""

        if self.gaze.is_blinking():
            text = "Blinking"
        elif self.gaze.is_right():
            text = "Looking right"
        elif self.gaze.is_left():
            text = "Looking left"
        elif self.gaze.is_center():
            text = "Looking center"

        cv2.putText(self.frame, text, (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, (147, 58, 31), 2)
    
        left_pupil = self.gaze.pupil_left_coords()
        right_pupil = self.gaze.pupil_right_coords()
        cv2.putText(self.frame, "Left pupil:  " + str(left_pupil), (5, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(self.frame, "Right pupil: " + str(right_pupil), (5, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    def print_frame(self):
        cv2.imshow('Video', cv2.resize(self.frame,(1600,960),interpolation = cv2.INTER_CUBIC))
        
    

a = stream_video()
while True: 
    a.take_frame()
    a.eye_construction()
    a.print_frame()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break