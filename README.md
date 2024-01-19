# Adaptive Video Stabilization With Kalman Filter and Gaussian Filter

## 1 Introduction
Our project focuses on mitigating motion jitters in UAVs and drones with low-cost cameras. Using a software- based video processing solution, we employ the Kalman and Gaussian Filters to enhance corner detection, estimate motion, and dynamically modify an affine matrix for smooth transitions. Stabilized videos are quantitatively compared through the average Structural Similarity Index (SSIM) to assess our solution's effectiveness in reducing undesired motion jitters.

### 1.1 Preprocessing
In our project's first step, we make the frames clearer by using a 3x3 box filter to sharpen them. This helps to highlight features that are easier to track, especially smaller movements. Our results in Table 1 confirm this improvement, showing higher average SSIM for the frames that went through this sharpening process. This step is crucial for making our tracking more efficient and the footage better overall.
![preprocessing](https://github.com/spoorthiuk/video-stabilization-using-gaussian-and-kalman-filters/blob/main/assets/preprocessing.png)

Figure 2: The above figures show how we are unable to track the same features as a trackable feature without preprocessing (a) but can be solved by sharpening the frames beforehand (b).
## 2 Video Stabilization Using Kalman Filter
We start by preprocessing frames to enhance and identify easily trackable features. Calculating optical flow reveals motion between frames, aiding in establishing an affine matrix that captures frame relationships. Utilizing the Kalman filter minimizes abrupt motions, ensuring smoother transitions. Our approach, effective with information from the previous frame alone, is well-suited for real-time video stabilization.

### 2.1 Corner Detection and Optical Flow Computation
In estimating the motion and optical flow between consecutive frames, we rely on identifying corner points within the images. OpenCV provides two useful functions, namely goodFeaturesToTrack and CornerHarris, for corner detection. In our project, we opt for goodFeaturesToTrack to identify the crucial corner points.
After determining the features to track, we compute the optical flow between the two consecutive frames using the Pyramid Lucas Kanade Tracker algorithm such that the given two frames are similar.
![corner detection](https://github.com/spoorthiuk/video-stabilization-using-gaussian-and-kalman-filters/blob/main/assets/opticalflow.png)

###  2.2 Motion Estimation
A 2D Affine transform model representing the change of scale in the X and Y axis, rotation and Translation in the X and Y axis is determined such that the following relationship between consecutive frames is maintained

### 2.3 Kalman Filter
The Kalman filter is used in estimating and correcting the motion parameters, such as translation and rotation, to stabilize shaky or jittery video sequences. The Kalman Filter involves two major stages:
1) State Prediction Stage: The next state of the system is determined based on the dynamics of the system. 𝑥′ is the estimated state of the system at time k, in our case it is the scale in the X and Y axis, translation in the X and Y axis, and rotation angle theta.
2) Update Stage: In this stage, the measurement from the current frame is used to update and correct the
value of 𝑥′. We calculate
the Kalman Gain 𝐾 , based on the previous covariance 𝑃 , observation 𝐻 and measurement noise
𝑅, measurement, this Kalman gain is then used along with the current measured state 𝑍 to update the
estimated state to arrive at the current estimate 𝑥 and to update the estimated value of covariance to
arrive to get 𝑃 .
Since the Kalman filter algorithm is based only on the previous frame and not on a sequence of frames, it can be used in smoothening out the jitters in real-time video streams.
## 3 Video Stabilization Using Gaussian Filter
Our video stabilization pipeline, shown in Figure 6, comprises three key steps. We start with a comprehensive frame review, aiming to enhance sharpness in the pre-processing stage. The next step involves estimating the camera path using the Euclidean motion model, smoothing the path by adjusting each image. Finally, the introduction of the Gaussian Filter enhances stability, resulting in a smooth, stabilized video that reduces the shakiness of the original footage.
### 3.1 Motion Estimation
Motion Estimation is a crucial phase in image stabilization, involving key steps to determine camera movement in a video sequence. The initial step calculates the warp matrix, employing Euclidean motion assumptions for frame-to-frame transformations. The "Parametric Image Alignment" technique enhances computing efficiency in matrix construction. Building a stack of warp matrices then reveals the camera's velocity and trajectory, showing the motion direction and speed throughout the video. Applying these warp matrices to frames using OpenCV's warpPerspective function stabilizes camera motion by aligning pictures with the desired path, minimizing motion- induced distortions.
### 3.2 Gaussian Filter
The undesirable movements in video sequences can be efficiently smoothed out and eliminated by applying a low-frequency Gaussian kernel filter to the warp matrix. To prevent the accumulating error caused by the original and smoothed transform chain cascade, the neighboring frame’s local displacement is smoothed to
produce a compensated motion.
The kernel parameters, such as window size and sigma, are chosen based on the level of smoothing desired. The resulting smoothed frames represent the camera's intended motion, while high-frequency noise is suppressed.
## 4 Results 
### 4.1 SSIM
To quantitatively measure the performance of our video stabilization system, we use the SSIM metrics to measure the similarity between consecutive frames in the original video and the stabilized video. 
![SSIM](https://github.com/spoorthiuk/video-stabilization-using-gaussian-and-kalman-filters/blob/main/assets/SSIM.png)
Table 1: Above shows the average SSIM comparison for the two videos using the Kalman Filter (left) and the Gaussian filter (right). It can be observed that in both cases SSIM was increased with preprocessing and the Gaussian filter gave better results.
### 4.2 Plots
We can also observe how well our algorithms performed by plotting X and Y translation graphs and SSIM graphs.
![plo1](https://github.com/spoorthiuk/video-stabilization-using-gaussian-and-kalman-filters/blob/main/assets/Plot1.png)
Figure 7: X and Y translation graphs show the smoothening performed by the Kalman filter on Selfie.mp4 video and the SSIM graph shows the improvement in SSIM value between consecutive frames.
![plo1](https://github.com/spoorthiuk/video-stabilization-using-gaussian-and-kalman-filters/blob/main/assets/Plot2.png)
Figure 8: X and Y translation graphs show the smoothening performed by the Gaussian filter on Selfie.mp4 video and the SSIM graph shows the improvement in SSIM value between consecutive frames.
![frames](https://github.com/spoorthiuk/video-stabilization-using-gaussian-and-kalman-filters/blob/main/assets/Frames.png)

Figure 9: The above figure shows how we crop the warped frame to achieve motion stabilized frame
