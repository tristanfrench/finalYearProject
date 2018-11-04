import numpy as np
import cv2
import time
import processImg 
import os
from os import listdir
from os.path import isfile, join


def readVideo(mainFolder,videoFolder,vid,write=False):
    videoPath = videoFolder+vid
    cap = cv2.VideoCapture(videoPath)
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
        cap = cv2.VideoCapture(videoPath)
        while(cap.isOpened()):
            ret, currentFrame = cap.read()
            if it in differenceArgs:
                if ret != 0:
                    currentImg = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
                    outfile = "%s/extractedVideos/%s_%s.jpg" %(mainFolder,vid[:-4],it)
                    print(outfile)
                    cv2.imwrite(outfile,currentImg)
            if ret == 0: # 0 if video is finished
                cap.release()
                break
            it+=1

    return differencePerFrame
    


def main():
    mainFolder = 'collectCircleTapRand_08161204'
    videoDir = mainFolder+'/videos/'
    myPath = os.getcwd()+'/'+videoDir
    allVideos = [f for f in listdir(myPath) if isfile(join(myPath, f))]
    for vid in allVideos[0:2]:
        result = readVideo(mainFolder,videoDir,vid,write=True)
    

if __name__ == '__main__':
    main()



#cv2.destroyAllWindows()