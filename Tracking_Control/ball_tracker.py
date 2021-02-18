from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import picar
import os
from threading import Thread
import importlib.util
import sys
import argparse
import time


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


#-------------------------ported from the machinelearningcamera.py
# Define and parse input arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
#                     required=True)
# parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
#                     default='detect.tflite')
# parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
#                     default='labelmap.txt')
# parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
#                     default=0.5)
# parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
#                     default='1280x720')
# parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
#                     action='store_true')

# args = parser.parse_args()

MODEL_NAME = '3BarModelV4_1'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = float(0.05)
resW, resH = 352, 288
imW, imH = int(resW), int(resH)
use_TPU = False

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


#-------------------------ported from the machinelearningcamera.py------end



picar.setup()
# Show image captured by camera, True to turn on, you will need #DISPLAY and it also slows the speed of tracking
show_image_enable   = True
draw_circle_enable  = True
scan_enable         = False
rear_wheels_enable  = False
front_wheels_enable = True
pan_tilt_enable     = True

if (show_image_enable or draw_circle_enable) and "DISPLAY" not in os.environ:
    print('Warning: Display not found, turn off "show_image_enable" and "draw_circle_enable"')
    show_image_enable   = False
    draw_circle_enable  = False

kernel = np.ones((5,5),np.uint8)
img = cv2.VideoCapture(-1)

SCREEN_WIDTH = 160
SCREEN_HIGHT = 120
img.set(3,SCREEN_WIDTH)
img.set(4,SCREEN_HIGHT)
CENTER_X = SCREEN_WIDTH/2
CENTER_Y = SCREEN_HIGHT/2
BALL_SIZE_MIN = SCREEN_HIGHT/5
BALL_SIZE_MAX = SCREEN_HIGHT/3

#PID Control for Servo Control
Constant_P = 2
Constant_I = 0.1
Constant_D = 0.08

#Intercommunicate Variables
# Ymin, Ymax, Xmin, Xmax

# Filter setting, DONOT CHANGE
hmn = 12
hmx = 37
smn = 96
smx = 255
vmn = 186
vmx = 255

# camera follow mode:
# 0 = step by step(slow, stable), 
# 1 = calculate the step(fast, unstable)
follow_mode = 1

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
fw.turn(90)
pan_servo.write(90)
tilt_servo.write(90)

motor_speed = 40

def nothing(x):
    pass

def main():
    MODEL_NAME = '3BarModelV4_1'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    min_conf_threshold = float(0.05)
    resW, resH = 352, 288
    imW, imH = int(resW), int(resH)
    use_TPU = False
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
    pan_angle = 90              # initial angle for pan
    tilt_angle = 0            # initial angle for tilt
    fw_angle = 90
    pan_speed = 0                # Discrete speed of pan servo  
    tilt_speed =0                # discrete speed of pan servo

    scan_count = 0
    print("Begin!")
    
#-----------------------------------------------------------------------------#    
   # Need to conduct the calibartion to determine the initial status
    while True:
       #------------------------------------------------ported machine learning code start ---------------- 
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

        #rectangle variable for control 
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
        
  #------------------------------------------------------------------ported ended      

	#x = 0             # x initial in the middle
     #   y = 0             # y initial in the middle
      #  r = 0             # ball radius initial to 0(no balls if r < ball_size)
# Variable explanation:
# x : current center x coordinate y: current center y coordinate r: area of the rectangle
        x = (xmin+xmax)/2
        y = (ymin+ymax)/2
        r = (xmax-xmin)*(ymax-ymin)
             
        print(x, y, r)

        # scan:
	#if r < BALL_SIZE_MIN:
        if r < 0:
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
            
        elif r > 0:
            if follow_mode == 0:
                if abs(x - CENTER_X) > MIDDLE_TOLERANT:
                    if x < CENTER_X:                              # Ball is on left
                        pan_angle += CAMERA_STEP
                        #print("Left   ", )
                        if pan_angle > PAN_ANGLE_MAX:
                            pan_angle = PAN_ANGLE_MAX
                    else:                                         # Ball is on right
                        pan_angle -= CAMERA_STEP
                        #print("Right  ",)
                        if pan_angle < PAN_ANGLE_MIN:
                            pan_angle = PAN_ANGLE_MIN
                if abs(y - CENTER_Y) > MIDDLE_TOLERANT:
                    if y < CENTER_Y :                             # Ball is on top
                        tilt_angle += CAMERA_STEP
                        #print("Top    " )
                        if tilt_angle > TILT_ANGLE_MAX:
                            tilt_angle = TILT_ANGLE_MAX
                    else:                                         # Ball is on bottom
                        tilt_angle -= CAMERA_STEP
                        #print("Bottom ")
                        if tilt_angle < TILT_ANGLE_MIN:
                            tilt_angle = TILT_ANGLE_MIN
            else:
                delta_x = CENTER_X - x
                delta_y = CENTER_Y - y
                print("x = %s, delta_x = %s" % (x, delta_x))
                print("y = %s, delta_y = %s" % (y, delta_y))
                delta_pan = int(Constant_P * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * delta_x + Constant_I * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * x + Constant_D * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * pan_speed)
                #print("delta_pan = %s" % delta_pan)
                pan_angle += delta_pan
                pan_speed = delta_pan
                delta_tilt = int(Constant_P * float(CAMERA_Y_ANGLE) / SCREEN_HIGHT * delta_y + Constant_I * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * y + Constant_D * float(CAMERA_X_ANGLE) / SCREEN_WIDTH * tilt_speed)
                #print("delta_tilt = %s" % delta_tilt)
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
            fw_angle = 180 - pan_angle
            if fw_angle < FW_ANGLE_MIN or fw_angle > FW_ANGLE_MAX:
                fw_angle = ((180 - fw_angle) - 90)/2 + 90
                if front_wheels_enable:
                    fw.turn(fw_angle)
                if rear_wheels_enable:
                    bw.speed = motor_speed
                    bw.backward()
            else:
                if front_wheels_enable:
                    fw.turn(fw_angle)
                if rear_wheels_enable:
                    bw.speed = motor_speed
                    bw.forward()
        else:
            bw.stop()
        
def destroy():
    bw.stop()
    img.release()
    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()

def test():
    fw.turn(90)

def find_blob() :
    radius = 0
    # Load input image
    _, bgr_image = img.read()

    orig_image = bgr_image

    bgr_image = cv2.medianBlur(bgr_image, 3)

    # Convert input image to HSV
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image, keep only the red pixels
    lower_red_hue_range = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
    upper_red_hue_range = cv2.inRange(hsv_image, (160, 100, 100), (179, 255, 255))
    # Combine the above two images
    red_hue_image = cv2.addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0)

    red_hue_image = cv2.GaussianBlur(red_hue_image, (9, 9), 2, 2)

    # Use the Hough transform to detect circles in the combined threshold image
    circles = cv2.HoughCircles(red_hue_image, cv2.HOUGH_GRADIENT, 1, 120, 100, 20, 10, 0)
    circles = np.uint16(np.around(circles))
    # Loop over all detected circles and outline them on the original image
    all_r = np.array([])
    # print("circles: %s"%circles)
    if circles is not None:
        try:
            for i in circles[0,:]:
                # print("i: %s"%i)
                all_r = np.append(all_r, int(round(i[2])))
            closest_ball = all_r.argmax()
            center=(int(round(circles[0][closest_ball][0])), int(round(circles[0][closest_ball][1])))
            radius=int(round(circles[0][closest_ball][2]))
            if draw_circle_enable:
                cv2.circle(orig_image, center, radius, (0, 255, 0), 5)
        except IndexError:
            pass
            # print("circles: %s"%circles)

    # Show images
    if show_image_enable:
        cv2.namedWindow("Threshold lower image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Threshold lower image", lower_red_hue_range)
        cv2.namedWindow("Threshold upper image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Threshold upper image", upper_red_hue_range)
        cv2.namedWindow("Combined threshold images", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Combined threshold images", red_hue_image)
        cv2.namedWindow("Detected red circles on the input image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Detected red circles on the input image", orig_image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        return (0, 0), 0
    if radius > 3:
        return center, radius
    else:
        return (0, 0), 0


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        destroy()
