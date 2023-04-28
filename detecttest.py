# main function

import cv2
import time
import PoseDetect as pd

cap = cv2.VideoCapture(0)
pTime = 0
detector = pd.PoseDetect()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.getPosition(img)
    print(lmList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
    cv2.imshow("Frame", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break