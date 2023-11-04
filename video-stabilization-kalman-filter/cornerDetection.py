import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/stairs.jpeg')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray_img,50,0.01,10)
corners = np.intp(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
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

# Apply a threshold to highlight corners
threshold = 0.01 * dst.max()
image_with_corners = img.copy()
image_with_corners[dst > threshold] = [255, 0, 0]  # Mark corners in red
plt.figure()
plt.imshow(image_with_corners)
plt.title('Using CornerHarris')
plt.show()