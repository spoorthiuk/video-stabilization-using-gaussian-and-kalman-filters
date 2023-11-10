import cv2
import numpy as np
import matplotlib.pyplot as plt

Q1 = 0.004
R1 = 0.5

class VideoStabilization():
    def __init__(self) -> None:
        self.k = 1
        self.errorScaleX = 1
        self.errorScaleY = 1
        self.errorTheta = 1
        self.errorTransX = 1
        self.errorTransY = 1

        self.qScaleX = Q1
        self.qScaleY = Q1
        self.qTheta = Q1
        self.qTransX = Q1
        self.qTransY = Q1

        self.rScaleX = R1
        self.rScaleY = R1
        self.rTheta = R1
        self.rTransX = R1
        self.rTransY = R1

        self.sumScaleX = 0
        self.sumScaleY = 0
        self.sumTheta = 0
        self.sumTransX = 0
        self.sumTransY = 0

        self.scaleX = 0
        self.scaleY = 0
        self.theta = 0
        self.transX = 0
        self.transY = 0

        self.horizontalBorder = 70

        self.smoothedMat = np.zeros((2, 3), dtype=np.float64)
        pass

    def kalman_filter(self):
        frame1ScaleX = self.scaleX
        frame1ScaleY = self.scaleY
        frame1Theta = self.theta
        frame1TransX = self.transX
        frame1TransY = self.transY

        frame1ErrScaleX = self.errorScaleX + self.qScaleX
        frame1ErrScaleY = self.errorScaleY + self.qScaleY
        frame1ErrTheta = self.errorTheta + self.qTheta
        frame1ErrTransX = self.errorTransX + self.qTransX
        frame1ErrTransY = self.errorTransY + self.qTransY

        gainScaleX = frame1ErrScaleX / (frame1ErrScaleX + self.rScaleX)
        gainScaleY = frame1ErrScaleY / (frame1ErrScaleY + self.rScaleY)
        gainTheta = frame1ErrTheta / (frame1ErrTheta + self.rTheta)
        gainTransX = frame1ErrTransX / (frame1ErrTransX + self.rTransX)
        gainTransY = frame1ErrTransY / (frame1ErrTransY + self.rTransY)

        self.scaleX = frame1ScaleX + gainScaleX * (self.sumScaleX - frame1ScaleX)
        self.scaleY = frame1ScaleY + gainScaleY * (self.sumScaleY - frame1ScaleY)
        self.theta = frame1Theta + gainTheta * (self.sumTheta - frame1Theta)
        self.transX = frame1TransX + gainTransX * (self.sumTransX - frame1TransX)
        self.transY = frame1TransY + gainTransY * (self.sumTransY - frame1TransY)

        self.errorScaleX = ( 1 - gainScaleX ) * frame1ErrScaleX
        self.errorScaleY = ( 1 - gainScaleY ) * frame1ErrScaleY
        self.errorTheta = ( 1 - gainTheta ) * frame1ErrTheta
        self.errorTransX = ( 1 - gainTransX) * frame1ErrTransX
        self.errorTransY = ( 1 - gainTransY) * frame1ErrTransY

        pass

    def stabilize(self, frame1, frame2):
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        rows, cols = frame1.shape
        verticalBorder = self.horizontalBorder * rows / cols
        
        #track features between frames
        noOfFramesToTrack = 200
        featureTrackingThreshold = 0.01
        minDistanceBetweenPoints = 10
        features1 = cv2.goodFeaturesToTrack(frame1, noOfFramesToTrack, featureTrackingThreshold, minDistanceBetweenPoints)
        features2, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame2, features1, None)
        goodFeatures1 = []
        goodFeatures2 = []
        for i in range(0,len(status)):
            pt1 = tuple(features1[i][0])
            pt2 = tuple(features2[i])
            goodFeatures1.append(pt1)
            goodFeatures2.append(pt2)
        goodFeatures1 = np.array(goodFeatures1)
        goodFeatures2 = np.array(goodFeatures2)
        affine,_ = cv2.estimateAffine2D(goodFeatures1, goodFeatures2)
        dx = affine[0, 2]
        dy = affine[1, 2]
        da = np.arctan2(affine[1, 0], affine[0, 0])
        ds_x = affine[0, 0] / np.cos(da)
        ds_y = affine[1, 1] / np.cos(da)

        sx = ds_x
        sy = ds_y

        self.sumTransX += dx
        self.sumTransY += dy
        self.sumTheta += da
        self.sumScaleX += ds_x
        self.sumScaleY += ds_y

        #skipping the predicted state for the first iteration
        if self.k == 1:
            self.k +=1
        else:
            self.kalman_filter()

        diffScaleX = self.scaleX - self.sumScaleX
        diffScaleY = self.scaleY - self.sumScaleY
        diffTheta = self.theta - self.sumTheta
        diffTransX = self.transX - self.sumTransX
        diffTransY = self.transY - self.sumTransY

        ds_x = ds_x + diffScaleX
        ds_y = ds_y + diffScaleY
        dx = dx + diffTransX
        dy = dy + diffTransY

        self.smoothedMat[0,0] = sx * np.cos(da)
        self.smoothedMat[0,1] = sx * -np.sin(da)
        self.smoothedMat[1,0] = sy * np.sin(da)
        self.smoothedMat[1,1] = sy * np.cos(da)

        self.smoothedMat[0,2] = dx
        self.smoothedMat[1,2] = dy

        smoothedFrame = cv2.warpAffine(frame1, self.smoothedMat, frame2.shape[::-1])
        print(int(verticalBorder),int(smoothedFrame.shape[0]-verticalBorder))
        smoothedFrame = smoothedFrame[int(verticalBorder):int(smoothedFrame.shape[0]-verticalBorder),self.horizontalBorder:smoothedFrame.shape[1]-self.horizontalBorder]

        return smoothedFrame

cap = cv2.VideoCapture('/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/32.mp4')
if (cap.isOpened()== False):
    print("Error openingfile")
#read 40 frames and store it in a list
frames = []
for _ in range(0,100):
    ret, frame = cap.read()
    if ret:
        #gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        #frames.append(gray_frame)

VS = VideoStabilization()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
smoothFrame = VS.stabilize(frames[0],frames[1])
width, height = smoothFrame.shape
output_video = '/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/stabilized_video_32.mp4'
video_writer = cv2.VideoWriter(output_video, fourcc, 30, (width, height))
for i in range(1,100):
    smoothFrame = VS.stabilize(frames[i-1],frames[i])
    plt.figure()
    plt.imshow(cv2.cvtColor(smoothFrame, cv2.COLOR_GRAY2BGR))
    plt.show()
    video_writer.write(cv2.cvtColor(smoothFrame, cv2.COLOR_GRAY2BGR))
    #video_writer.write(cv2.cvtColor(smoothFrame, cv2.COLOR_YCR_CB2BGR))

print("Video saved to:", output_video)