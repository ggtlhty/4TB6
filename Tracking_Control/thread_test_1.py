######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import threading

#Import Package from Control
from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
#Package Import ends



# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
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
    
# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()


MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
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
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
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

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

#-----------------------------------------------------------------------------------#
# initial step for control
picar.setup()

# Control Panel for the Robot
scan_enable         = True
rear_wheels_enable  = False
front_wheels_enable = True
pan_tilt_enable     = True

#kernel = np.ones((5,5),np.uint8)

SCREEN_WIDTH = 352
SCREEN_HIGHT = 288
CENTER_X = SCREEN_WIDTH/2
CENTER_Y = SCREEN_HIGHT/2
BALL_SIZE_MIN = 0 #SCREEN_HIGHT/5
BALL_SIZE_MAX = 2000  #SCREEN_HIGHT/3

#PID Control for Servo Control
Constant_P = 2
Constant_I = 0.1
Constant_D = 0.08

# Filter setting, DONOT CHANGE
hmn = 12
hmx = 37
smn = 96
smx = 255
vmn = 186
vmx = 255


CAMERA_STEP = 2
CAMERA_X_ANGLE = 20
CAMERA_Y_ANGLE = 20

MIDDLE_TOLERANT = 5
PAN_ANGLE_MAX   = 170
PAN_ANGLE_MIN   = 10
TILT_ANGLE_MAX  = 150
TILT_ANGLE_MIN  = 70
FW_ANGLE_MAX    = 90+50
FW_ANGLE_MIN    = 90-50

SCAN_POS = [[20, TILT_ANGLE_MIN], [50, TILT_ANGLE_MIN], [90, TILT_ANGLE_MIN], [130, TILT_ANGLE_MIN], [160, TILT_ANGLE_MIN], 
            [160, 80], [130, 80], [90, 80], [50, 80], [20, 80]]

bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()
pan_servo = Servo.Servo(1)
tilt_servo = Servo.Servo(2)
picar.setup()

fw.offset = 0
pan_servo.offset = 0
tilt_servo.offset = 0

bw.speed = 0
fw.turn(105)
pan_servo.write(90)                          
tilt_servo.write(20)

motor_speed = 40

def nothing(x):
    pass

pan_angle = 90              # initial angle for pan
tilt_angle = 20             # initial angle for tilt
fw_angle = 105
pan_speed = 0                # Discrete speed of pan servo  
tilt_speed =0                # discrete speed of pan servo

x = 0			#initial x position of the center
y = 0 			#initial y position of the center
r = 0			#initial area of the rectangle

scan_count = 0
print("Begin!")

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
   def run(self):
      print "Starting " + self.name
      control_module_thread()
      print "Exiting " + self.name

def control_module_thread():
    # scan:
    if r < BALL_SIZE_MIN:	#x=0, y=0 and a counter
       bw.stop()
       if scan_enable:
          #bw.stop()
          pan_angle = SCAN_POS[scan_count][0]
          tilt_angle = SCAN_POS[scan_count][1]
          if pan_tilt_enable:
             pan_servo.write(pan_angle)
             tilt_servo.write(tilt_angle)
          scan_count += 1
          if scan_count >= len(SCAN_POS):
             scan_count = 0
          else:
             sleep(0.1)
            
    elif r < BALL_SIZE_MAX:
       delta_x = CENTER_X - x
       delta_y = CENTER_Y - y
       print("x = %s, delta_x = %s" % (x, delta_x))
       print("y = %s, delta_y = %s" % (y, delta_y))
       delta_pan = int(Constant_P * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * delta_x + Constant_I * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * x + Constant_D * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * pan_speed)
       if x==0 and y ==0: delta_pan = 0
       print("delta_pan = %s" % delta_pan)
       pan_angle += delta_pan
       pan_speed = delta_pan
       delta_tilt = int(Constant_P * float(CAMERA_Y_ANGLE) / SCREEN_HIGHT * delta_y + Constant_I * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * y + Constant_D * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * tilt_speed)
       if x==0 and y ==0: delta_tilt = 0
       print("delta_tilt = %s" % delta_tilt)
       tilt_angle += delta_tilt
       tilt_speed = delta_tilt

       if pan_angle > PAN_ANGLE_MAX:
           pan_angle = PAN_ANGLE_MAX
       elif pan_angle < PAN_ANGLE_MIN:
           pan_angle = PAN_ANGLE_MIN
       if tilt_angle > TILT_ANGLE_MAX:
           tilt_angle = TILT_ANGLE_MAX
       elif tilt_angle < TILT_ANGLE_MIN:
           tilt_angle = TILT_ANGLE_MIN
            
       if pan_tilt_enable:
           pan_servo.write(pan_angle)
           tilt_servo.write(tilt_angle)
       sleep(0.01)
       
# Distancing Maintaining 
       if r == 0:#counter needed
           bw.stop()
           sleep(2)
           if scan_enable:
              print("SCAN")
          #bw.stop()
              pan_angle = SCAN_POS[scan_count][0]
              tilt_angle = SCAN_POS[scan_count][1]
              if pan_tilt_enable:
                 pan_servo.write(pan_angle)
                 tilt_servo.write(tilt_angle)
              scan_count += 1
              sleep(1)
              if scan_count >= len(SCAN_POS):
                 scan_count = 0
           else:
              sleep(0.1)
       elif r < 1200:
           fw_angle = 195-pan_angle
           if fw_angle < FW_ANGLE_MIN or fw_angle > FW_ANGLE_MAX:
              fw_angle = ((180 - fw_angle) - 90)/2 + 90
          #fw.angle = 105
#              if front_wheels_enable:
              fw.turn(fw_angle)
#          if rear_wheels_enable:
              bw.speed = 30
              bw.forward()                                                              
           else:
#          if front_wheels_enable:
              fw.turn(fw_angle)
#          if rear_wheels_enable:
              bw.speed = 30
              bw.backward()
#       elif r < 1400:
#           print("BBBBBBBBBBBBB")
#           bw.speed = 25
#           bw.backward()
       else:
           bw.speed = 30
           bw.forward()
		
       
    else:
        bw.stop()
    # Press 'q' to quit	


thread1 = myThread(1, "Thread-1", 1)
thread1.start()

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
    ymin = 0
    xmin = 0
    ymax = 0
    xmax = 0
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
	    # for control: return (ymax-ymin)*(xmax-xmin)
	    # resolution: 352 x 288 => center point: 352/2, 288/2
	    # x-offset: (xmin+xmax)/2 - 352/2
	    # y-offset: ...
            print("y max box coordinate" + str(ymax))
            print("y min box cooridnate" + str(ymin))
	    
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    #-------------------------------------------------------------------------------------------------------------------------
    #Control for the robot
    x = 0             # x initial in the middle
    y = 0             # y initial in the middle
    r = 0             # ball radius initial to 0(no balls if r < ball_size

    x = (xmax+xmin)/2
    y = (ymax+ymin)/2
    r = (xmax-xmin)*(ymax-ymin)

    print(x, y, r)


    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
bw.stop()
cv2.destroyAllWindows()
videostream.stop()
