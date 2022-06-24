import cv2 as cv 
import mediapipe as mp 
import numpy as np 
import time
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol = volume.GetVolumeRange()

camera = cv.VideoCapture(0)

mhands = mp.solutions.hands
hands = mhands.Hands()
mpdraw = mp.solutions.drawing_utils
#draw = mpdraw.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1) 

pTime = 0

while True:
    isTrue, frame = camera.read()

    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    cv.rectangle(frame, (35, 120), (65, 380), (255,0,0), 3)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 8:
                    cv.circle(frame, (cx,cy), 16, (255,0,255), -1)
                    point1 = [cx, cy]
                if id == 4:
                    cv.circle(frame, (cx,cy), 16, (255,0,255), -1)
                    point2 = [cx, cy]
            
            cx, cy = int((point1[0]+point2[0])/2), int((point1[1]+point2[1])/2) 
            
            length = math.hypot(point1[0]-point2[0], point1[1]-point2[1])
            
            vo = np.interp(length, [10,220], [vol[0], vol[1]])
            volb = np.interp(length, [8,220], [380,120])
            volp = np.interp(length, [8,220], [0,100])

            volume.SetMasterVolumeLevel(vo, None)
            cv.circle(frame, (cx, cy), 15, (255,0,255), -1)
            if length<40:
                cv.circle(frame, (cx, cy), 15, (0,0,255), -1)

            cv.line(frame, point1, point2, (255,0,255), 2)       
            cv.rectangle(frame, (35, int(volb)), (65, 380), (0,255,0), cv.FILLED)
            cv.putText(frame, f'{int(volp)}%', (25, 420), cv.FONT_HERSHEY_PLAIN, 2, (255,0,0),2)
            mpdraw.draw_landmarks(frame, hand, mhands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(frame, f'FPS: {int(fps)}' , (10,40), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
    cv.imshow('Camera', frame)
    
    if cv.waitKey(1) & 0xFF==ord('d'):
      break

camera.release()
cv.destroyAllWindows()