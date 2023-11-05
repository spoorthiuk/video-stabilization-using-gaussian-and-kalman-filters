import cv2
import matplotlib.pyplot as plt
import numpy as np

def msr_psnr(img1,img2):
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)

    # Calculate the maximum possible pixel value
    max_pixel_value = 255.0  # For 8-bit images

    # Calculate the PSNR
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    print(f"MSR: {mse} PSNR: {psnr} dB")

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
prev_idx = 9
curr_idx = 10
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
for i in feature_points:
    x,y = i.ravel()
    cv2.circle(prev_frame,(x,y),2,0,-1)
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
for i in curr_features:
    x,y = [i[0],i[1]]
    cv2.circle(curr_frame_cpy,(x,y),2,0,-1)

curr_frame = frames[curr_idx]
filtered_features1 = []
filtered_features2 = []
for i in range(len(feature_points)):
    if status[i]:
        pt1 = tuple(feature_points[i][0])
        pt2 = tuple(curr_features[i])
        filtered_features1.append(pt1)
        filtered_features2.append(pt2)
        cv2.arrowedLine(curr_frame, pt1, pt2, (0, 255, 0), 1)
            #plt.quiver(pt1[1], pt1[0], pt2[1], pt2[0],linewidth = 0.5)


filtered_features1 = np.array(filtered_features1)
filtered_features2 = np.array(filtered_features2)
M,_ = cv2.estimateAffine2D(filtered_features1, filtered_features2)

transformed_image = cv2.warpAffine(original_image, M, (original_image.shape[1], original_image.shape[0]))
print(transformed_image)
plt.subplot(1,3,1)
plt.imshow(original_image,cmap='gray')
plt.title('Previous Frame')
plt.subplot(1,3,2)
plt.imshow(reference_frame,cmap='gray')
plt.title('Current Frame')
plt.subplot(1,3,3)
plt.imshow(transformed_image,cmap='gray')
plt.title('Current Frame constructed from \nAffine Tranformation matrix')
plt.suptitle('Motion Estimation')
plt.show()

msr_psnr(original_image,transformed_image)
msr_psnr(reference_frame,transformed_image)