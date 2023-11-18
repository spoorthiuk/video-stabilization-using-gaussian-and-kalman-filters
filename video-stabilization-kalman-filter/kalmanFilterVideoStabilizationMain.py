import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from matplotlib.animation import FuncAnimation

Q1 = 0.004
R1 = 0.5
ORG_TRANSX = []
ORG_TRANSY = []
KALMAN_TRANSX = []
KALMAN_TRANSY = []

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
    plt.plot(KALMAN_TRANSX)
    plt.legend(['Orginal video','Kalman Filter stabilized video'])
    plt.ylim([-100,100])
    plt.title('X Transform')
    plt.show()

    plt.plot(ORG_TRANSY)
    plt.plot(KALMAN_TRANSY)
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
        ax1.plot(t[:i],KALMAN_TRANSX[:i])
        ax1.legend(['Orginal video','Kalman Filter stabilized video'])
        ax2.plot(t[:i],ORG_TRANSY[:i])
        ax2.plot(t[:i],KALMAN_TRANSY[:i])
        ax2.legend(['Orginal video','Kalman Filter stabilized video'])
    t = range(len(ORG_TRANSX))
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_size_inches([20,10])
    anim = FuncAnimation(fig, animate, frames = len(t), interval = 100)
    anim.save('animation.gif', writer='imagemagick', fps=30)
    plt.show()

class VideoStabilization():
    def __init__(self) -> None:
        self.k = 1
        self.prev_frame = ''
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

        self.horizontalBorder = 10

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
        #print(frame1[0].shape)
        colour_frame1 = frame1.copy()
        colour_frame2 = frame2.copy()
        frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        rows, cols = frame1.shape
        verticalBorder = self.horizontalBorder * rows / cols
        
        #track features between frames
        noOfFramesToTrack = 300
        featureTrackingThreshold = 0.01
        minDistanceBetweenPoints = 5
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
        da = np.arctan2(affine[1, 0], affine[0, 0])
        ds_x = affine[0, 0] / np.cos(da)
        ds_y = affine[1, 1] / np.cos(da)

        ORG_TRANSX.append(dx)
        ORG_TRANSY.append(dy)

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
        da = da + diffTheta

        self.smoothedMat[0,0] = sx * np.cos(da)
        self.smoothedMat[0,1] = sx * -np.sin(da)
        self.smoothedMat[1,0] = sy * np.sin(da)
        self.smoothedMat[1,1] = sy * np.cos(da)

        self.smoothedMat[0,2] = dx
        self.smoothedMat[1,2] = dy

        #print(colour_frame2.shape)
        #smoothenedFrame = cv2.warpAffine(frame1, self.smoothedMat, frame2.shape[::-1])
        smoothenedFrame = cv2.warpAffine(colour_frame1, self.smoothedMat, frame2.shape[::-1])
        smoothenedFrame = smoothenedFrame[int(verticalBorder):int(smoothenedFrame.shape[0]-verticalBorder),self.horizontalBorder:smoothenedFrame.shape[1]-self.horizontalBorder]
        smoothenedFrame_gray = cv2.warpAffine(frame1, self.smoothedMat, frame2.shape[::-1])
        smoothenedFrame_gray = smoothenedFrame_gray[int(verticalBorder):int(smoothenedFrame_gray.shape[0]-verticalBorder),self.horizontalBorder:smoothenedFrame_gray.shape[1]-self.horizontalBorder]
            
        if(self.prev_frame != ''):
            #track features between frames
            noOfFramesToTrack = 200
            featureTrackingThreshold = 0.01
            minDistanceBetweenPoints = 10
            features1 = cv2.goodFeaturesToTrack(self.prev_frame, noOfFramesToTrack, featureTrackingThreshold, minDistanceBetweenPoints)
            features2, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame, smoothenedFrame_gray, features1, None)
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
            da = np.arctan2(affine[1, 0], affine[0, 0])
            ds_x = affine[0, 0] / np.cos(da)
            ds_y = affine[1, 1] / np.cos(da)
            KALMAN_TRANSX.append(dx)
            KALMAN_TRANSY.append(dy)
            #print(f'KALMAN MSE = {MSE(self.prev_frame,smoothenedFrame_gray)}, PSNR = {PSNR(MSE(self.prev_frame,smoothenedFrame_gray))}, SSIM = {SSIM(self.prev_frame,smoothenedFrame_gray)}')
            STB_SSIM.append(SSIM(self.prev_frame,smoothenedFrame_gray))
        self.prev_frame = smoothenedFrame_gray
        return smoothenedFrame

input_output_path = {
    'rover1' : ['/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/rover1.mp4','/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/results/kalman_filter_rover1.mp4'],
    'basketball' : ['/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/basketball.mp4','/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/results/kalman_filter_basketball.mp4'],
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
#output_video = '/Users/spoorthiuk/ASU/digital-video-processing/video-stabalization/assets/Drone_footage_enhanced.mp4'
output_video = input_output_path[video][1]
VS = VideoStabilization()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#output_video = "stabilized_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'H264')
fps = 30
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
for i in range(1,200):
    prev_frame = preProcessing(frames[i-1])
    cur_frame = preProcessing(frames[i])
    smoothFrame = VS.stabilize(prev_frame,cur_frame)
    smoothFrame = cv2.resize(smoothFrame,(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    prev_frame_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)
    cur_frame_gray = cv2.cvtColor(cur_frame,cv2.COLOR_BGR2GRAY)
    #print(f'{i}: Original MSE = {MSE(prev_frame_gray,cur_frame_gray)}, PSNR = {PSNR(MSE(prev_frame_gray,cur_frame_gray))}, SSIM = {SSIM(prev_frame_gray,cur_frame_gray)}')
    ORG_SSIM.append(SSIM(prev_frame_gray,cur_frame_gray))
    video_writer.write(smoothFrame)
cap.release()
video_writer.release()
cv2.destroyAllWindows()
print("Video saved to:", output_video)

plt.style.use('ggplot')
#plot_line_animation()
plot_line_graph()
plot_ssim()

print(f'Average SSIM:\n1) Original Video:{np.average(ORG_SSIM)}\n2) Kalman Filter stabilized Video:{np.average(STB_SSIM)}')