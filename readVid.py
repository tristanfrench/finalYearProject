import numpy as np
import cv2
import time
import processImg 
import os
from os import listdir
from os.path import isfile, join


def readVideo(videoFile,write=False):
    
    cap = cv2.VideoCapture(videoFile)
    i=0
    differencePerFrame = []
    while(cap.isOpened()):
        ret, currentFrame = cap.read()
        if ret != 0:
            # grayscale
            currentImg = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
            # To see images
            #cv2.imshow('cf',currentImg)
            #print('frameNumber=',i)
            #cv2.waitKey()
        if i == 0:
            #first frame only
            firstImg = currentImg
            
        
        difference = processImg.main(firstImg,currentImg)
        differencePerFrame.append(difference)
        if ret == 0: # 0 if video is finished
            cap.release()
            break
        i+=1
    if write == True:
        it=0
        differenceArgs = processImg.getArgmax(differencePerFrame)
        cap = cv2.VideoCapture(videoFile)
        while(cap.isOpened()):
            ret, currentFrame = cap.read()
            if it in differenceArgs:
                if ret != 0:
                    currentImg = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
                    outfile = "%s_%s.jpg" %(videoFile,it)
                    print(outfile)
                    cv2.imwrite(outfile,currentImg)
            if ret == 0: # 0 if video is finished
                cap.release()
                break
            it+=1

    return differencePerFrame
    


def main():
    myPath = os.getcwd()+'\\videoFile'
    allVideos = [f for f in listdir(myPath) if isfile(join(myPath, f))]
    print(allVideos[0:2])
    #result = readVideo("collectCircleTapRand_09041010/video_3.mp4",write=False)
    #print(result)
    

if __name__ == '__main__':
    main()



#cv2.destroyAllWindows()