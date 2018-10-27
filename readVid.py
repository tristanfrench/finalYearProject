import numpy as np
import cv2
import time

cap = cv2.VideoCapture("collectCircleTapRand_09041010/video_1.mp4")
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.waitKey(1) #so you can see the video
       
    if ret != 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if ret == 0: #False if video is finished
        cap.release()
        break
    cv2.imshow("frame",gray)
    i+=1
    


#cv2.destroyAllWindows()