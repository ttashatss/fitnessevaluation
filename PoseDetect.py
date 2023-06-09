import cv2
import mediapipe as mp
import time
import math


class PoseDetect:

    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = False, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils # type: ignore
        self.mpPose = mp.solutions.pose # type: ignore
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)


    def findPose(self, img, draw=True):
        # find pose landmarks from the image captured
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img, draw=True):
        # get x y coordinates of each body parts
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
 
    def getAngle(self, a, b, c):
        # calculate angle between 3 points
        ang = math.degrees(math.atan2(c[2]-b[2], c[1]-b[1]) - math.atan2(a[2]-b[2], a[1]-b[1]))
        return ang + 360 if ang < 0 else ang
    
    def getVerticalAngle(self, a , b):
        # calculate angle between a line (2 points) and the vertical axes
        return math.degrees(math.atan2(a[1]-b[1], a[2]-b[2]))
    
    def detectSquat(self, lmList):
        # detect if the user is performing squat and return as an array consisting of [back angle, knee angle, neck angle, is squat]
        sq = [0,0,0,False]
        if len(lmList) > 27:
            back = self.getAngle(lmList[25], lmList[23], lmList[11]) # calculate back angle
            sq[0] = 1 if back > 90 else -1 if back < 80 else 0
            knee = self.getAngle(lmList[23], lmList[25], lmList[27]) # calculate knee angle
            sq[1] = 1 if back > 90 else -1 if back < 80 else 0
            neck = self.getVerticalAngle(lmList[11], lmList[2]) # calculate neck angle
            sq[2] = 1 if neck > 45 else 0
            sq[3] = (sq[0] == 0) and (sq[1] == 0) and (sq[2] == 0) # determine if is squat
        return sq
    
    def detectUp(self, lmList):
        up = False
        if len(lmList) > 27:
            back = self.getAngle(lmList[25], lmList[23], lmList[11])
            knee = self.getAngle(lmList[23], lmList[25], lmList[27])
            up = (back > 120) and (knee > 120)
        return up
        
    
def main():
    cap = cv2.VideoCapture('squat.mp4')
    pTime = 0
    detector = PoseDetect()
    isSquat = [0,0,0,False] 
    prevSquat = False
    count = 0

    while True:
        success, img = cap.read()
        img = detector.findPose(img) # find pose landmarks from the image
        lmList = detector.getPosition(img) # find x y coordinates of the landamarks
        isSquat = detector.detectSquat(lmList) # detect if user is performing squat
       
        if isSquat[0] == 1: cv2.putText(img, "bend your back forward", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 
        if isSquat[1] == 1: cv2.putText(img, "lower yourself by bending your knee", (50, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if isSquat[2] == 1: cv2.putText(img, "keep your head straight", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 


        if isSquat[3]:
            if not prevSquat:
                count += 1
            prevSquat = True
        if detector.detectUp(lmList):
            prevSquat = False
        
        # get fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # cv2.putText(img, "back angle: " + str(isSquat[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(img, "knee angle: " + str(isSquat[1]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(img, "neck angle: " + str(isSquat[2]), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(img, str(isSquat[3]), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "no. of squats: " + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "calorie burn: " + str(count*0.32), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.imshow("Image", img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()