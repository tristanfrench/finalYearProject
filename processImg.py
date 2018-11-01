import cv2
import numpy as np

def main(firstImg,currentImg):
    row, col = np.shape(firstImg)
    
    
    difference = abs(firstImg.astype('int')-currentImg.astype('int')).astype('uint8')
    difference = sum(sum(i>50 for i in difference))/(row*col))
    
    
    #cv2.imwrite("differenceImg.jpg",difference)
    #newImg = cv2.imread("differenceImg.jpg",0)
    #cv2.imshow("difffer",newImg)
    #cv2.waitKey(0)
    return difference

def imgProcessing(img):
    

    
    return img




