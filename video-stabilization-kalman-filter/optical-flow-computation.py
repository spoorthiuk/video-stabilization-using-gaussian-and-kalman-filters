import cv2
import matplotlib.pyplot as plt
import numpy as np

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
prev_frame = frames[9]
curr_frame = frames[10]

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

curr_frame = frames[10]
plt.subplot(1,3,3)
for i in range(len(feature_points)):
    if status[i]:
        pt1 = tuple(corners[i][0])
        pt2 = tuple(curr_features[i])
        print(pt1,pt2)
        cv2.arrowedLine(curr_frame, pt1, pt2, (0, 255, 0), 1)
        #plt.quiver(pt1[1], pt1[0], pt2[1], pt2[0],linewidth = 0.5)

plt.imshow(curr_frame, cmap='gray')
plt.title('Optical Flow Vectors')
plt.show()