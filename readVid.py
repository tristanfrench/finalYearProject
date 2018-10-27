import numpy as np
import cv2
import time
import processImg 

cap = cv2.VideoCapture("collectCircleTapRand_09041010/video_1.mp4")
i=0
while(cap.isOpened()):
    ret, currentFrame = cap.read()
     #so you can see the video
    
    if ret != 0:
        currentImg = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)#grayscale
    if i == 0:
        firstImg = currentImg
        #cv2.imshow("img1",currentImg)
    
    processImg.main(firstImg,currentImg)
        
    if ret == 0: #False if video is finished
        cap.release()
        break
        
    #cv2.imshow("frame",gray)
    i+=1
    




#cv2.destroyAllWindows()