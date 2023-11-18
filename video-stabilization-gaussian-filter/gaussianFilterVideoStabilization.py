import cv2
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import convolve
from skimage.metrics import structural_similarity
from matplotlib.animation import FuncAnimation

Q1 = 0.004
R1 = 0.5
ORG_TRANSX = []
ORG_TRANSY = []

GAUSSIAN_TRANSX = []
GAUSSIAN_TRANSY = []

ORG_SSIM = []
STB_SSIM = []

def brightness(image, value):
    return np.clip(image * value, 0, 255).astype(np.uint8)

def contrast(image, value):
    mean = np.mean(image)
    return np.clip((image - mean) * value + mean, 0, 255).astype(np.uint8) 

def sharpening(frame):
    #High Pass Filterning and Gaussian Low Pass Filter
    filter_3x3_hpf = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    filtered_frame = cv2.filter2D(frame,-1,filter_3x3_hpf)
    gaussian_kernel_size = (5,5)
    filtered_frame = cv2.GaussianBlur(filtered_frame, gaussian_kernel_size, sigmaX=1.2)
    return filtered_frame

def preProcessing(frame):
    luminance_frame = brightness(frame, 0.9)
    contrast_frame = contrast(luminance_frame, 1.2)
    filtered_frame = sharpening(contrast_frame)
    return filtered_frame

#function for getting the Mean Squared Error of two images
def MSE(img1, img2):
    if img1.size != img2.size:
        print('Images have different sizes')
        return
    img1_pixels = list(img1.flatten())
    img2_pixels = list(img2.flatten())
    sum_pixels = 0
    for p1, p2 in zip(img1_pixels,img2_pixels):
        sum_pixels += (p1-p2)**2
    mse = sum_pixels/len(img1_pixels)
    return round(mse,2)

#function for getting the PSNR value
def PSNR(mse):
    #defining maximum pixel value to be 255 for an 8 bit grayscale image
    max_pixel_val =255
    psnr = 20 * np.log10(max_pixel_val / np.sqrt(mse))
    return round(psnr,2)

def SSIM(img1, img2):
    return round(structural_similarity(img1, img2, win_size= 5),2)

#Function to plot normal line graph
def plot_line_graph():
    plt.figure()
    plt.plot(ORG_TRANSX)
    plt.plot(GAUSSIAN_TRANSX)
    plt.legend(['Orginal video','Kalman Filter stabilized video'])
    plt.ylim([-100,100])
    plt.title('X Transform')
    plt.show()

    plt.plot(ORG_TRANSY)
    plt.plot(GAUSSIAN_TRANSY)
    plt.legend(['Orginal video','Kalman Filter stabilized video'])
    plt.ylim([-100,100])
    plt.title('Y Transform')
    plt.show()

x_vals = []
y_vals = []

def plot_ssim():
    plt.figure()
    plt.ylim([0,1])
    plt.ylabel('SSIM')
    plt.plot(ORG_SSIM)
    plt.plot(STB_SSIM)
    plt.legend(['Original video\'s SSIM', 'Kalman Filter stabilized video\'s SSIM'])
    plt.show()
    
def plot_line_animation():
    def animate(i):
        #ax.cla()
        #ax.set_ylim([-100,100])
        #ax.set_xlim([0,len(t)])
        #plt.plot(t[:i],data1[:i])
        #plt.plot(t[:i],data2[:i])
        ax1.cla()
        ax2.cla()
        ax1.set_ylim([-100,100])
        ax2.set_ylim([-100,100])
        ax1.set_xlim([0,len(t)])
        ax2.set_xlim([0,len(t)])
        ax1.set_title('X Transform')
        ax2.set_title('Y Transform')
        ax1.plot(t[:i],ORG_TRANSX[:i])
        ax1.plot(t[:i],GAUSSIAN_TRANSX[:i])
        ax1.legend(['Orginal video','Kalman Filter stabilized video'])
        ax2.plot(t[:i],ORG_TRANSY[:i])
        ax2.plot(t[:i],GAUSSIAN_TRANSY[:i])
        ax2.legend(['Orginal video','Kalman Filter stabilized video'])
    t = range(len(ORG_TRANSX))
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches([20,10])
    anim = FuncAnimation(fig, animate, frames = len(t), interval = 100)
    anim.save('animation.gif', writer='imagemagick', fps=30)
    plt.show()

class VideoStabilization():
    def __init__(self) -> None:
        frames = []
        top = 0
        bottom = 0
        left = 0
        right = 0
        pass

    def get_border_pads(self, img_shape, warp_stack):
        maxmin = []
        corners = np.array([[0,0,1], [img_shape[1], 0, 1], [0, img_shape[0],1], [img_shape[1], img_shape[0], 1]]).T
        warp_prev = np.eye(3)
        for warp in warp_stack:
            warp = np.concatenate([warp, [[0,0,1]]])
            warp = np.matmul(warp, warp_prev)
            warp_invs = np.linalg.inv(warp)
            new_corners = np.matmul(warp_invs, corners)
            xmax,xmin = new_corners[0].max(), new_corners[0].min()
            ymax,ymin = new_corners[1].max(), new_corners[1].min()
            maxmin += [[ymax,xmax], [ymin,xmin]]
            warp_prev = warp.copy()
        maxmin = np.array(maxmin)
        bottom = maxmin[:,0].max()
        print('bottom', maxmin[:,0].argmax()//2)
        top = maxmin[:,0].min()
        print('top', maxmin[:,0].argmin()//2)
        left = maxmin[:,1].min()
        print('right', maxmin[:,1].argmax()//2)
        right = maxmin[:,1].max()
        print('left', maxmin[:,1].argmin()//2)
        self.top = int(-top)
        self.bottom = int(bottom-img_shape[0])
        self.left = int(-left)
        self.right = int(right-img_shape[1])
        return int(-top), int(bottom-img_shape[0]), int(-left), int(right-img_shape[1])
    def get_homography(self, img1, img2, motion = cv2.MOTION_EUCLIDEAN):
        imga = img1.copy().astype(np.float32)
        imgb = img2.copy().astype(np.float32)
        if len(imga.shape) == 3:
            imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
        if len(imgb.shape) == 3:
            imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
        if motion == cv2.MOTION_HOMOGRAPHY:
            warpMatrix=np.eye(3, 3, dtype=np.float32)
        else:
            warpMatrix=np.eye(2, 3, dtype=np.float32)
        warp_matrix = cv2.findTransformECC(templateImage=imga,inputImage=imgb,warpMatrix=warpMatrix, motionType=motion)[1]
        return warp_matrix 
    
    def homography_gen(self, warp_stack):
        H_tot = np.eye(3)
        wsp = np.dstack([warp_stack[:,0,:], warp_stack[:,1,:], np.array([[0,0,1]]*warp_stack.shape[0])])
        for i in range(len(warp_stack)):
            H_tot = np.matmul(wsp[i].T, H_tot)
            yield np.linalg.inv(H_tot)#[:2]
    
    def gauss_convolve(self, trajectory, window, sigma):
        kernel = signal.gaussian(window, std=sigma)
        kernel = kernel/np.sum(kernel)
        return convolve(trajectory, kernel, mode='reflect')

    def create_warp_stack(self):
        warp_stack = []
        for i, img in enumerate(self.frames[:-1]):
            warp_stack += [self.get_homography(img, self.frames[i+1])]
        return np.array(warp_stack)
    
    def moving_average(self, warp_stack, sigma_mat):
        x,y = warp_stack.shape[1:]
        original_trajectory = np.cumsum(warp_stack, axis=0)
        smoothed_trajectory = np.zeros(original_trajectory.shape)
        for i in range(x):
            for j in range(y):
                kernel = signal.gaussian(1000, sigma_mat[i,j])
                kernel = kernel/np.sum(kernel)
                smoothed_trajectory[:,i,j] = convolve(original_trajectory[:,i,j], kernel, mode='reflect')
        smoothed_warp = np.apply_along_axis(lambda m: convolve(m, [0,1,-1], mode='reflect'), axis=0, arr=smoothed_trajectory)
        smoothed_warp[:,0,0] = 0
        smoothed_warp[:,1,1] = 0
        return smoothed_warp, smoothed_trajectory, original_trajectory
    
    def apply_warping_fullview(self,warp_stack):
        top, bottom, left, right = self.get_border_pads(img_shape=self.frames[0].shape, warp_stack=warp_stack)
        H = self.homography_gen(warp_stack)
        imgs = []
        for i, img in enumerate(self.frames[1:]):
            H_tot = next(H)+np.array([[0,0,left],[0,0,top],[0,0,0]])
            img_warp = cv2.warpPerspective(img, H_tot, (img.shape[1]+left+right, img.shape[0]+top+bottom))
            imgs += [img_warp]
        return imgs
    
    def stabilize(self, frame_stack):
        self.frames = frame_stack
        warp_stack = self.create_warp_stack()
        smoothed_warp, smoothed_trajectory, original_trajectory =self.moving_average(warp_stack, sigma_mat= np.array([[1000,15, 10],[15,1000, 10]]))
        print('Shape Trajectory:',smoothed_trajectory.shape)
        smoothened_frames= self.apply_warping_fullview(warp_stack=warp_stack-smoothed_warp)
        return smoothened_frames

input_output_path = {
    'rover1' : ['/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/rover1.mp4','/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/results/gaussian_filter_rover1.mp4'],
    'drone' : []
}
video = 'rover1'
#cap = cv2.VideoCapture('/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/32.mp4')
input_video = input_output_path[video][0]
cap = cv2.VideoCapture(input_video)
if (cap.isOpened()== False):
    print("Error openingfile")
frames = []
for _ in range(0,1000):
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

#output_video = '/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/Gaussian_Drone_footage_enhanced.mp4'
output_video = input_output_path[video][1]
VS = VideoStabilization()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#output_video = "stabilized_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'H264')
fps = 30
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
#preprocessing the frames
frame_stack = []
for i in range(0,200):
    frame = preProcessing(frames[i])
    frame_stack.append(frame)

color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
smoothFrames = VS.stabilize(frame_stack)
padding_ver = 50
padding_hor = int(padding_ver * 606/1000)
smoothFramesResized = []
for smoothFrame in smoothFrames:
    print(frame.shape,smoothFrame.shape)
    smoothFrame = cv2.resize(smoothFrame[int(VS.left)+padding_ver:int(smoothFrame.shape[0]-VS.right-padding_ver),VS.top+padding_hor:smoothFrame.shape[1]-VS.bottom-padding_hor], (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    smoothFramesResized.append(smoothFrame)
    video_writer.write(smoothFrame)
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("Video saved to:", output_video)

#plotting the tranformation for original and stabilized video
#original video
noOfFramesToTrack = 300
featureTrackingThreshold = 0.01
minDistanceBetweenPoints = 10
#for i in range(1, len(frames)):
for i in range(1, len(smoothFramesResized)):
    try:
        frame1 = frames[i-1]
        frame2 = frames[i]
        #ORG_SSIM.append(SSIM(frames[i-1],frames[i]))
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        ORG_SSIM.append(SSIM(frame1,frame2))
        features1 = cv2.goodFeaturesToTrack(frame1, noOfFramesToTrack, featureTrackingThreshold, minDistanceBetweenPoints)
        features2, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame2, features1, None)
        goodFeatures1 = []
        goodFeatures2 = []
        for i in range(0,len(status)):
            if(status[i]):
                pt1 = tuple(features1[i][0])
                pt2 = tuple(features2[i])
                goodFeatures1.append(pt1)
                goodFeatures2.append(pt2)
        goodFeatures1 = np.array(goodFeatures1)
        goodFeatures2 = np.array(goodFeatures2)
        affine,_ = cv2.estimateAffine2D(goodFeatures1, goodFeatures2)
        dx = affine[0, 2]
        dy = affine[1, 2]
        ORG_TRANSX.append(dx)
        ORG_TRANSY.append(dy)
    except:
        pass
#Stabilized video
for i in range(1, len(smoothFramesResized)):
    try:
        frame1 = smoothFramesResized[i-1]
        frame2 = smoothFramesResized[i]
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        STB_SSIM.append(SSIM(frame1,frame2))
        features1 = cv2.goodFeaturesToTrack(frame1, noOfFramesToTrack, featureTrackingThreshold, minDistanceBetweenPoints)
        features2, status, error = cv2.calcOpticalFlowPyrLK(frame1, frame2, features1, None)
        goodFeatures1 = []
        goodFeatures2 = []
        for i in range(0,len(status)):
            if(status[i]):
                pt1 = tuple(features1[i][0])
                pt2 = tuple(features2[i])
                goodFeatures1.append(pt1)
                goodFeatures2.append(pt2)
        goodFeatures1 = np.array(goodFeatures1)
        goodFeatures2 = np.array(goodFeatures2)
        affine,_ = cv2.estimateAffine2D(goodFeatures1, goodFeatures2)
        dx = affine[0, 2]
        dy = affine[1, 2]
        GAUSSIAN_TRANSX.append(dx)
        GAUSSIAN_TRANSY.append(dy)
    except:
        pass

plt.style.use('ggplot')
#plot_line_animation()
plot_line_graph()
plot_ssim()

print(f'Average SSIM:\n1) Original Video:{np.average(ORG_SSIM)}\n2) Kalman Filter stabilized Video:{np.average(STB_SSIM)}')