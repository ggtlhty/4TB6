#in terminal do
#pip3 install opencv-python
#pip3 install 
#then run the script
#press "space" to take a picture
#press "q" or "ctrl+z" to quit the program

import cv2
import imutils
import time

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("camera", frame)
    if(cv2.waitKey(1)%256 == 32):
        # SPACE pressed
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
    elif(cv2.waitKey(1)%256 == ord("q")):
        break
cam.release()
cv2.destroyAllWindows()

