import cv2
import numpy as np
import math

def main(firstImg,currentImg):
    row, col = np.shape(firstImg)
    
    
    difference = firstImg.astype('int')-currentImg.astype('int')
    #difference = sum(sum(i>50 for i in difference))/(row*col)
    difference = math.sqrt(sum(sum(i**2 for i in difference)))
    
    
    #cv2.imwrite("differenceImg.jpg",difference)
    #newImg = cv2.imread("differenceImg.jpg",0)
    #cv2.imshow("difffer",newImg)
    #cv2.waitKey(0)
    return difference

def imgProcessing(img):
    

    
    return img




