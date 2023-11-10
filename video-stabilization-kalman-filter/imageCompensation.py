import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/Foreman360p.mp4')
if (cap.isOpened()== False):
    print("Error openingfile")
#read 40 frames and store it in a list
frames = []
for _ in range(0,40):
    ret, frame = cap.read()
    if ret:
        gray_frame = gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

# Load the two consecutive frames
prev_idx = 4
curr_idx = 5
prev_frame = frames[prev_idx]
curr_frame = frames[curr_idx]
original_image = frames[prev_idx].copy()
reference_frame = frames[curr_idx].copy()
'''plt.figure()
plt.imshow(prev_frame, cmap='gray')
plt.title('Previous Frame')
plt.show()

plt.figure()
plt.imshow(curr_frame, cmap='gray')
plt.title('Current Frame')
plt.show()'''

feature_points_raw = cv2.goodFeaturesToTrack(prev_frame,300,0.01,10)
feature_points = np.intp(feature_points_raw)
'''block_size = 2
aperture_size = 3

dst = cv2.cornerHarris(prev_frame, block_size, aperture_size, 0.04)
threshold =  0.05 * dst.max()
feature_points = np.argwhere(dst > threshold)'''
#plt.imshow(prev_frame, cmap='gray')
#plt.title('Using goodFeaturesToTrack')
#plt.show()


next_feature_points, status, error = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, feature_points_raw, None)
curr_features = []
for point in next_feature_points:
    curr_features.append([int(point[0][0]),int(point[0][1])])
#next_feature_points = np.array([(int(point[0][0]), int(point[0][0])) for point in next_feature_points])
curr_frame_cpy = curr_frame.copy()

curr_frame = frames[curr_idx]
filtered_features1 = []
filtered_features2 = []
for i in range(len(feature_points)):
    if status[i]:
        pt1 = tuple(feature_points[i][0])
        pt2 = tuple(curr_features[i])
        filtered_features1.append(pt1)
        filtered_features2.append(pt2)


filtered_features1 = np.array(filtered_features1)
filtered_features2 = np.array(filtered_features2)
print(filtered_features1[1],filtered_features2[1])
affine,_ = cv2.estimateAffine2D(filtered_features1, filtered_features2)

dx = affine[0, 2]
dy = affine[1, 2]
da = np.arctan2(affine[1, 0], affine[0, 0])
ds_x = affine[0, 0] / np.cos(da)
ds_y = affine[1, 1] / np.cos(da)

sx = ds_x
sy = ds_y

smoothedMat = np.zeros((2, 3), dtype=np.float64)

diff_scaleX = sx - ds_x
diff_scaleY = sy - ds_y
diff_transX = dx - dx
diff_transY = dy - dy
diff_thetha = da - da

ds_x = ds_x + diff_scaleX
ds_y = ds_y + diff_scaleY
dx = dx + diff_transX
dy = dy + diff_transY
da = da + diff_thetha

smoothedMat[0, 0] = sx * np.cos(da)
smoothedMat[0, 1] = sx * -np.sin(da)
smoothedMat[1, 0] = sy * np.sin(da)
smoothedMat[1, 1] = sy * np.cos(da)

smoothedMat[0, 2] = dx
smoothedMat[1, 2] = dy

smoothed_frame = cv2.warpAffine(prev_frame, smoothedMat, curr_frame.shape[::-1])

plt.figure()
plt.subplot(1,3,1)
plt.imshow(prev_frame, cmap='gray')
plt.title('Previous Frame')

plt.subplot(1,3,2)
plt.imshow(curr_frame, cmap='gray')
plt.title('Current Frame')

plt.subplot(1,3,3)
plt.imshow(smoothed_frame, cmap='gray')
plt.title('Smoothened Frame')
plt.show()
