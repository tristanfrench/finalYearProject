import cv2
import numpy as np

def main(firstImg,currentImg):
    row, col = np.shape(firstImg)
    
    
    difference = abs(firstImg.astype('int')-currentImg.astype('int')).astype('uint8')
    print(sum(sum(i>50 for i in difference))/(row*col))
    
    #imgProcessing(img)
    
    #cv2.imshow("firstImg",firstImg)
    #cv2.imshow("currentImg",currentImg)
    cv2.imwrite("differenceImg.jpg",difference)
    newImg = cv2.imread("differenceImg.jpg",0)
    cv2.imshow("difffer",newImg)
    cv2.waitKey(0)

def imgProcessing(img):
    

    '''
    for j in range(row):
        if img[j,1] != 0:
            print(j)
            break
    
    img[0:60,:]=255
    '''
    return img




