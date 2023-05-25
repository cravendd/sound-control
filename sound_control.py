import cv2 as cv
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#open camera for video capturing
cap = cv.VideoCapture(0)

device = AudioUtilities.GetSpeakers()
interface = device.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL,None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
    
# isOpened() returns true if video capturing has been initialised
if not cap.isOpened():
    print("cannot find camera")
    exit()
    
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            # cap.read() returns true if the frames have been grabbed and returns the frame
            ret, image = cap.read()
            image.flags.writeable = False
            frame = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = hands.process(frame)

            if not ret:
                print("cant recieve frame (stream end?). exiting ...")
                break
            #get landmarks from the live feed
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    lmlist = []
                    for id , lm in enumerate(hand_landmarks.landmark):
                        h,w,c = image.shape
                        cx,cy = int(lm.x*w),int(lm.y*h)
                        lmlist.append([id,cx,cy])
                if lmlist:
                    #get coordinates of thumb and forefinger
                    x1,y1 = lmlist[4][1], lmlist[4][2]
                    x2,y2 = lmlist[8][1], lmlist[8][2]
                    length = math.hypot(x2-x1,y2-y1)
                    if length<50:
                        z1 = (x1+x2)//2
                        z2 = (y1+y2)//2
                        cv.circle(image, (z1,z2),15,(2550,0,7),cv.FILLED)
                #adjust volume displayed in rectanguler bar
                volumeRange = volume.GetVolumeRange()
                minVol = volumeRange[0]
                maxVol = volumeRange[1]
                vol = np.interp(length,[50,300],[minVol,maxVol])
                volBar = np.interp(length,[50,300],[400,150])
                volume.SetMasterVolumeLevel(vol,None)
                cv.rectangle(image,(50,150),(85,400),(254,8,0),3)
                cv.rectangle(image,(50,int(volBar)),(85,400),(0,124,234),cv.FILLED)

            image = cv.flip(image,1)
            cv.imshow('frame',image)
            if cv.waitKey(1) == ord('q'):
                break
cap.release()
cv.destroyAllWindows()
