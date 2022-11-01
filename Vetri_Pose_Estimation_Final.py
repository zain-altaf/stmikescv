# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import pdb
import time
import math
import pathlib
from threading import Thread
import importlib.util
import datetime


import time

# Define the codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
filename = "test"+".m4v"
outputDir = os.path.join("/home/pi/Desktop/Output", filename)


# set the frame count to zero
count = 0

# Define and parse input arguments
#***Please make sure to change the default path for modeldir, output_path and bg based on where you store the file***
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default="/home/pi/Desktop/Final_Project/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected keypoints (specify between 0 and 1).',
                    default=0.4)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--output_path', help="Where to save processed imges from pi.",
                    default = "/home/pi/Desktop/Final_Project/Images")
parser.add_argument("-bg", "--background", default="/home/pi/Desktop/Final_Project/bg.jpg", 
    help="Path to background")
args = parser.parse_args()

#Setting necessary variables from arguments
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

#Opening Background image
bg = cv2.imread(args.background)


#Class for starting the videostream - allows us to capture from camera
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        #breakpoint()
        
        self.stream = cv2.VideoCapture(0)
        self.video_writer = cv2.VideoWriter(outputDir, fourcc, 30.0, (640, 480))
        
        print("Camera initiated.")
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME)


# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
#set stride to 32 based on model size
output_stride = 32

led_on = False
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

def mod(a, b):
    """find a % b"""
    floored = np.floor_divide(a, b)
    return np.subtract(a, np.multiply(floored, b))

def sigmoid(x):
    """apply sigmoid actiation to numpy array"""
    return 1/ (1 + np.exp(-x))
    
def sigmoid_and_argmax2d(inputs, threshold):
    """return y,x coordinates from heatmap"""
    #v1 is 9x9x17 heatmap
    v1 = interpreter.get_tensor(output_details[0]['index'])[0]
    height = v1.shape[0]
    width = v1.shape[1]
    depth = v1.shape[2]
    reshaped = np.reshape(v1, [height * width, depth])
    reshaped = sigmoid(reshaped)
    #apply threshold
    reshaped = (reshaped > threshold) * reshaped
    coords = np.argmax(reshaped, axis=0)
    yCoords = np.round(np.expand_dims(np.divide(coords, width), 1)) 
    xCoords = np.expand_dims(mod(coords, width), 1) 
    return np.concatenate([yCoords, xCoords], 1)

def get_offset_point(y, x, offsets, keypoint, num_key_points):
    """get offset vector from coordinate"""
    y_off = offsets[y,x, keypoint]
    x_off = offsets[y,x, keypoint+num_key_points]
    return np.array([y_off, x_off])
    

def get_offsets(output_details, coords, num_key_points=17):
    """get offset vectors from all coordinates"""
    offsets = interpreter.get_tensor(output_details[1]['index'])[0]
    offset_vectors = np.array([]).reshape(-1,2)
    for i in range(len(coords)):
        heatmap_y = int(coords[i][0])
        heatmap_x = int(coords[i][1])
        #make sure indices aren't out of range
        if heatmap_y >8:
            heatmap_y = heatmap_y -1
        if heatmap_x > 8:
            heatmap_x = heatmap_x -1
        offset_vectors = np.vstack((offset_vectors, get_offset_point(heatmap_y, heatmap_x, offsets, i, num_key_points)))  
    return offset_vectors

def draw_lines(keypoints, image, bad_pts):
    """connect important body part keypoints with lines"""
    #Colour of the drawing
    #color = (255, 0, 0)
    color = (0, 255, 0)
    thickness = 2
    #refernce for keypoint indexing: https://www.tensorflow.org/lite/models/pose_estimation/overview
    body_map = [[5,6], [5,7], [7,9], [5,11], [6,8], [8,10], [6,12], [11,12], [11,13], [13,15], [12,14], [14,16]]
    for map_pair in body_map:
        #print(f'Map pair {map_pair}')
        if map_pair[0] in bad_pts or map_pair[1] in bad_pts:
            continue
        start_pos = (int(keypoints[map_pair[0]][1]), int(keypoints[map_pair[0]][0]))
        end_pos = (int(keypoints[map_pair[1]][1]), int(keypoints[map_pair[1]][0]))
        image = cv2.line(image, start_pos, end_pos, color, thickness)
    return image

#flag for debugging
debug = True 
# Logging
verbose = True     # False= Non True=Display showMessage

# Motion Settings
threshold = 20     # How Much a pixel has to change
sensitivity = 300  # How Many pixels need to change for motion detection

# Camera Settings
testWidth = 128
testHeight = 80
nightShut = 5.5    # seconds Night shutter Exposure Time default = 5.5  Do not exceed 6 since camera may lock up
nightISO = 800
if nightShut > 6:
    nightShut = 5.9
SECONDS2MICRO = 1000000  # Constant for converting Shutter Speed in Seconds to Microseconds    
nightMaxShut = int(nightShut * SECONDS2MICRO)
nightMaxISO = int(nightISO)
nightSleepSec = 8   # Seconds of long exposure for camera to adjust to low light 

def checkForMotion(data1, data2):
    # Find motion between two data streams based on sensitivity and threshold
    motionDetected = False
    pixColor = 1 # red=0 green=1 blue=2
    pixChanges = 0;
    for w in range(0, testWidth):
        for h in range(0, testHeight):
            # get the diff of the pixel. Conversion to int
            # is required to avoid unsigned short overflow.
            pixDiff = abs(int(data1[h][w][pixColor]) - int(data2[h][w][pixColor]))
            if  pixDiff > threshold:
                pixChanges += 1
            if pixChanges > sensitivity:
                break; # break inner loop
        if pixChanges > sensitivity:
            break; #break outer loop.
    if pixChanges > sensitivity:
        motionDetected = True
    return motionDetected

def userMotionCode():
    count = 0
    t0 = time.time()
    
    
    while True:
        if True:
            #timestamp an output directory for each capture
            outdir = pathlib.Path(args.output_path) / time.strftime('%Y-%m-%d_%H-%M-%S-%Z')
            outdir.mkdir(parents=True)
            time.sleep(.1)
            led_on = True
            f = []
            # Initialize frame rate calculation
            frame_rate_calc = 1
            freq = cv2.getTickFrequency()
            videostream1 = VideoStream(resolution=(imW,imH),framerate=30).start()
            time.sleep(1)

            #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
            #elapsed = 0
            #while elapsed<10:
            
            while True:
                t1 = time.time()
                elapsed = t1-t0
                print('running loop') #To show that the loop is working - can remove for actual usage
                # Start timer (for calculating frame rate)
                t1 = cv2.getTickCount()
                
                # Grab frame from video stream
                frame1 = videostream1.read()
                # Acquire frame and resize to expected shape [1xHxWx3]
                frame = frame1.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (width, height))
                input_data = np.expand_dims(frame_resized, axis=0)
                
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                bg = cv2.imread(args.background)
                bg_resized = cv2.resize(bg, (width, height))

                # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
                if floating_model:
                    input_data = (np.float32(input_data) - input_mean) / input_std

                # Perform the actual detection by running the model with the image as input
                interpreter.set_tensor(input_details[0]['index'],input_data)
                interpreter.invoke()
                
                #get y,x positions from heatmap
                coords = sigmoid_and_argmax2d(output_details, min_conf_threshold)
                #keep track of keypoints that don't meet threshold
                drop_pts = list(np.unique(np.where(coords ==0)[0]))
                #get offets from postions
                offset_vectors = get_offsets(output_details, coords)
                #use stide to get coordinates in image coordinates
                keypoint_positions = coords * output_stride + offset_vectors
            
                # Loop over all detections and draw detection box if confidence is above minimum threshold
                confidence = 0
                for i in range(len(keypoint_positions)):
                    #don't draw low confidence points
                    if i in drop_pts:
                        continue
                    # Center coordinates
                    confidence += 1
                    x = int(keypoint_positions[i][1])
                    y = int(keypoint_positions[i][0])
                    center_coordinates = (x, y)
                    radius = 2
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.circle(bg_resized, center_coordinates, radius, color, thickness)
                    if debug:
                        cv2.putText(bg_resized, str(i), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1) # Draw label text
     
                frame_final = draw_lines(keypoint_positions, bg_resized, drop_pts)

                # Draw framerate in corner of frame - remove for small image display
                #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
                #cv2.putText(frame_resized,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

                # Calculate framerate
                t2 = cv2.getTickCount()
                time1 = (t2-t1)/freq
                frame_rate_calc= 1/time1
                f.append(frame_rate_calc)
    
                #save image with time stamp to directory
                #path = str(outdir) + '/'  + str(datetime.datetime.now()) + ".jpg"
                
                
                path = str(outdir) + '/'  + time.strftime('%Y-%m-%d_%H-%M-%S')+".jpg"
                
                #If at least one point of confidence exists, then the image is saved
                if confidence>0:
                    status = cv2.imwrite(path, frame_final)
                #If no confidence points are detected(nobody is in the image), a counter starts
                else:
                    #If the counter is greater than 50, the bg is changed and the loop is stopped - goes back to motion detect mode
                    if count>50:
                        cv2.imwrite(args.background, frame_resized)
                        count = 0
                        break
                    else:
                        count += 1
                cv2.imshow("Frame", frame_final)  # show the frame to our screen
                key = cv2.waitKey(33) & 0xFF
                videostream1.video_writer.write(frame_final)  # Write the video to the file system

                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    print(f"Saved images to: {outdir}")
                    led_on = False
                    # Clean up
                    cv2.destroyAllWindows()
                    #videostream1.stream.release()
                    videostream1.stop()
                    videostream1.video_writer.release()
                    time.sleep(2)
                    break
                #press b to change bg pic - make sure nobody is in the image
                if cv2.waitKey(1) == ord('b'):
                    #Change wait time if you want
                    print("Taking picture in...")
                    time.sleep(2)
                    print("3")
                    time.sleep(2)
                    print("2")
                    time.sleep(2)
                    print("1")
                    time.sleep(2)
                    frame = videostream1.read()
                    cv2.imwrite(args.background, frame)
                    print("bg changed")
                    
        
        #videostream1.stream.release()
        videostream1.stop()
        videostream1.video_writer.release()
        cv2.destroyAllWindows()
        print('Stopped video stream.')
        break
def main():
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    frame1 = videostream.read() 
    while True:
        frame2 = videostream.read()
        if checkForMotion(frame1, frame2):
            print ("There is motion")
            #videostream.stream.release()
            videostream.stop()
            
            #Pose-estimation is activated
            userMotionCode()
            
            time.sleep(2)
            frame1 = frame2
            #cv2.imwrite(args.background, frame1)
            #print("bg changed")
            videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
            
    return
main()