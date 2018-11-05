import numpy as np
import cv2
import csv
import time
import processImg 
import pandas as pd
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
    '''
    if write == True:
        it=0
        differenceArgs = processImg.getArgmax(differencePerFrame)
        cap = cv2.VideoCapture(videoPath)
        while(cap.isOpened()):
            ret, currentFrame = cap.read()
            if it in differenceArgs:
                if ret != 0:
                    currentImg = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
                    outfile = "%s/extractedImages/%s_%s.jpg" %(mainFolder,format_string(vid[:-4]),it)
                    cv2.imwrite(outfile,currentImg)
            if ret == 0: # 0 if video is finished
                cap.release()
                break
            it+=1
    '''
    return differencePerFrame
    

def readCsv(csv_file):
    data = pd.read_csv(csv_file)
    newData = data[['pose_1','pose_6']]
    return newData

def format_string(s):
    if len(s) != 10:
        for i in range(0,10-len(s)):
            s = s[:6] + '0' + s[6:] 
    return s


def main():
    mainFolder = 'collectCircleTapRand_08161204'
    videoDir = mainFolder+'/videos/'
    myPath = os.getcwd()+'/'+videoDir
    allVideos = [f for f in listdir(myPath) if isfile(join(myPath, f))]
    it = 0
    for vid in allVideos:
        it+=1
        result = readVideo(mainFolder,videoDir,vid,write=False)
        print(it/20,'%')
    #readCsv('video_targets.csv')
    

if __name__ == '__main__':
    main()


