import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_line(point,dis,angle):
    dis = -dis*4 #convert mm in pixel
    angle = -angle*np.pi/180 #convert deg in rad
    angle_p = angle-90*np.pi/180
    x = 40
    perpen_point = [point[0]+dis*np.cos(angle_p),point[1]+dis*np.sin(angle_p)]
    new_point_1 = [perpen_point[0]+x*np.cos(angle),perpen_point[1]+x*np.sin(angle)]
    new_point_2 = [perpen_point[0]-x*np.cos(angle),perpen_point[1]-x*np.sin(angle)]
    
    return new_point_1,new_point_2



a=[80,80]
angle = -44.9
dis = 8.14
b,c = show_line(a,dis,angle)
img = mpimg.imread('src/cropSampled/video_0002_9_crop.jpg')  
plt.imshow(img,cmap='gray')
plt.plot([c[0],b[0]], [c[1],b[1]], 'r-')
plt.show()