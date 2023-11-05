import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/stairs.jpeg')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_img,50,0.01,10)
corners = np.intp(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),2,255,-1)
plt.imshow(img)
plt.title('Using goodFeaturesToTrack')
plt.show()


img = cv2.imread('/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/stairs.jpeg')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Define the size of the neighborhood for corner detection
block_size = 2
aperture_size = 3

# Harris corner detection
dst = cv2.cornerHarris(gray_img, block_size, aperture_size, 0.04)
threshold =  0.05 * dst.max()
corner_points = np.argwhere(dst > threshold)
print(corner_points)
for i in corner_points:
    y,x = i.ravel()
    cv2.circle(img,(x,y),2,255,-1)
plt.imshow(img)
plt.title('Using CornerHarris')
plt.show()
