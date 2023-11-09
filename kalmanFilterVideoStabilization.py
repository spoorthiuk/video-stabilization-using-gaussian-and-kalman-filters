import cv2
import numpy as np
import matplotlib.pyplot as plt

Q1 = 0.004
R1 = 0.5

class VideoStabilization():
    def __init__(self, horzontalBorder) -> None:
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

        self.horizontalBorder = horzontalBorder

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
        rows, cols = frame1.shape()
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
            goodFeatures1.append(tuple(features1[i][0]))
            goodFeatures2.append(tuple(features2[i]))
        
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