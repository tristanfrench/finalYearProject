import numpy as np
import cv2
import time
import processImg 
import os
from os import listdir
from os.path import isfile, join


def readVideo(videoFile):
    
    cap = cv2.VideoCapture(videoFile)
    i=0
    while(cap.isOpened()):
        ret, currentFrame = cap.read()
        #so you can see the video
        
        if ret != 0:
            currentImg = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)#grayscale
            cv2.imshow('cf',currentImg)
            cv2.waitKey()
        if i == 0:
            firstImg = currentImg
            
        
        difference = processImg.main(firstImg,currentImg)
        print(difference)
        if ret == 0: #False if video is finished
            cap.release()
            break
            
        
        i+=1
        

def main():
    myPath = os.getcwd()+'\\videoFile'
    allVideos = [f for f in listdir(myPath) if isfile(join(myPath, f))]
    #print(allVideos[0])
    result = readVideo("collectCircleTapRand_09041010/video_1.mp4")
    

if __name__ == '__main__':
    main()



#cv2.destroyAllWindows()