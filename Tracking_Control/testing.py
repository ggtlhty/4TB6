# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

#Import Package from Control
from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep

picar.setup()
# Control Panel for the Robot
scan_enable         = False
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
fw_angle = 90
pan_speed = 0                # Discrete speed of pan servo  
tilt_speed =0                # discrete speed of pan servo

x = 0			#initial x position of the center
y = 0 			#initial y position of the center
r = 0			#initial area of the rectangle

scan_count = 0
print("Begin!")

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
   fw_angle = 95
   fw.turn(fw_angle)
   bw.speed = 30
   bw.backward()
bw.stop()
cv2.destroyAllWindows()
videostream.stop()
