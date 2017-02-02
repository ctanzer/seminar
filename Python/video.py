import numpy as np
import cv2
import time

last_frame = 0

cap = cv2.VideoCapture('video_wecker.avi')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        cap.release()
        cv2.destroyAllWindows()

    last_frame = frame

    cv2.imshow('frame', last_frame)
    
    while not cv2.waitKey(1) & 0xFF == ord('p'):
        time.sleep(0.0001)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

            

