import cv2
import numpy as np
import matplotlib.pyplot as plt
def smoothening(frame):
    #Gaussian Low Pass Filterning the given image
    gaussian_kernel_size = (3,3)
    filtered_frame = cv2.GaussianBlur(filtered_frame, gaussian_kernel_size, sigmaX=0.2)
    return filtered_frame
def sharpening(frame):
    #High Pass Filterning the given image
    filter_3x3_hpf = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    filtered_frame = cv2.filter2D(frame,-1,filter_3x3_hpf)
    return filtered_frame

def preProcessing(frame):
    #luminance_frame = brightness(frame,1.1)
    #contrast_frame = contrast(luminance_frame, 0.9)
    sharpened_frame = sharpening(frame)
    filtered_frame = sharpened_frame
    return filtered_frame

img = cv2.imread('/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/stairs.jpeg')
img = preProcessing(img)
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Frame 1')
frame = cv2.imread('/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/stairs.jpeg')
frame = preProcessing(frame)
angle = 10
center = (frame.shape[1] // 2, frame.shape[0] // 2)
# Apply the rotation transformation
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
noisy_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
gray_frame = cv2.cvtColor(noisy_frame,cv2.COLOR_BGR2GRAY)
plt.subplot(2,2,2)
plt.imshow(noisy_frame)
plt.title('Frame 2')
corners = cv2.goodFeaturesToTrack(gray_frame,50,0.01,10)
corners = np.intp(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(noisy_frame,(x,y),2,255,-1)
plt.subplot(2,2,4)
plt.imshow(noisy_frame)
plt.title('goodFeaturesToTrack on Frame 2')
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray_img,50,0.01,10)
corners = np.intp(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),2,255,-1)
plt.subplot(2,2,3)
plt.imshow(img)
plt.title('goodFeaturesToTrack on Frame 1')
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
