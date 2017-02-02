import numpy as np
import cv2


last_frame = 0

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    if ret == False:
        continue

    last_frame = np.flipud(frame)
    # last_frame = frame

    cv2.imshow('frame', last_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(last_frame.shape)
