import cv2
import matplotlib.pyplot as plt
import numpy as np

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

# Load the two consecutive frames
prev_idx = 9
curr_idx = 10
prev_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
curr_frame = gray_frame
original_image = img.copy()
reference_frame = frame.copy()

'''plt.figure()
plt.imshow(prev_frame, cmap='gray')
plt.title('Previous Frame')
plt.show()

plt.figure()
plt.imshow(curr_frame, cmap='gray')
plt.title('Current Frame')
plt.show()'''

feature_points = cv2.goodFeaturesToTrack(prev_frame,300,0.01,10)
corners = np.intp(feature_points)
for i in corners:
    x,y = i.ravel()
    cv2.circle(prev_frame,(x,y),2,0,-1)
#plt.imshow(prev_frame, cmap='gray')
#plt.title('Using goodFeaturesToTrack')
#plt.show()
plt.figure()
plt.subplot(1,3,1)
plt.imshow(prev_frame,cmap='gray')
plt.title('Previous Frame')


next_feature_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, feature_points, None)
curr_features = []
for point in next_feature_points:
    curr_features.append([int(point[0][0]),int(point[0][1])])
#next_feature_points = np.array([(int(point[0][0]), int(point[0][0])) for point in next_feature_points])
curr_frame_cpy = curr_frame.copy()
for i in curr_features:
    x,y = [i[0],i[1]]
    cv2.circle(curr_frame_cpy,(x,y),2,0,-1)
plt.subplot(1,3,2)
plt.imshow(curr_frame_cpy,cmap='gray')
plt.title('Current Frame Frame')

curr_frame = img
plt.subplot(1,3,3)
for i in range(len(feature_points)):
    if status[i]:
        pt1 = tuple(corners[i][0])
        pt2 = tuple(curr_features[i])
        print(pt1,pt2)
        cv2.arrowedLine(noisy_frame, pt1, pt2, (0, 255, 0), 1)
        #plt.quiver(pt1[1], pt1[0], pt2[1], pt2[0],linewidth = 0.5)

plt.imshow(noisy_frame, cmap='gray')
plt.title('Optical Flow Vectors')
plt.show()