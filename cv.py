import numpy as np
import cv2
import time
import random as rng


# Video source - can be camera index number given by 'ls /dev/video*
# or can be a video file, e.g. '~/Video.avi'
cap = cv2.VideoCapture(0)

lower_range = np.array([0,0, 255], dtype = np.uint8)
upper_range = np.array([0, 0, 255], dtype = np.uint8)
while(True):
    # Capture frame-by-frames
    ret, frame = cap.read()

    frame = cv2.resize(frame,(1280,640))
    # Our operations on the frame come here
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(color, lower_range, upper_range)

    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    #mask1 = cv2.inRange(color, (0,50,20), (5,255,255))
    #mask2 = cv2.inRange(color, (175,50,20), (180,255,255))

    ## Merge the mask and crop the red regions
    #mask = cv2.bitwise_or(mask1, mask2 )
    ##res = cv2.bitwise_and(frame, frame, mask=mask)
        
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) 
    gray = cv2.blur(gray, (3, 3))
##    rows = gray.shape[0]
##    #print(rows)
##    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,param1=50, param2=30, minRadius=0, maxRadius=0)
##    
##    if circles is not None:
##        circles = np.uint16(np.around(circles))
##        for i in circles[0, :]:        
##            center = (i[0], i[1])
##            # circle center
##            cv2.circle(res, center, 1, (0, 100, 100), 3)
##            # circle outline
##            radius = i[2]
##            cv2.circle(res, center, radius, (255, 0, 255), 3)
    

    #gray_blurred = cv2.blur(gray, (3, 3))
    #edged = cv2.Canny(gray, 30, 200)
    #gray_blurred = cv2.Canny(gray, 30, 200)
    #detected_circles = cv2.HoughCircles(gray_blurred,  cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 150, maxRadius = 999)


    # Draw circles that are detected. 
##    if detected_circles is not None: 
##      
##        # Convert the circle parameters a, b and r to integers. 
##        detected_circles = np.uint16(np.around(detected_circles)) 
##      
##        for pt in detected_circles[0, :]: 
##            a, b, r = pt[0], pt[1], pt[2] 
##      
##            # Draw the circumference of the circle. 
##            cv2.circle(res, (a, b), r, (0, 255, 0), 2) 
##      
##            # Draw a small circle (of radius 1) to show the center. 
##            cv2.circle(res, (a, b), 1, (0, 0, 255), 3) 
##            cv2.imshow("Detected Circle", res)

    #edged = cv2.Canny(gray, 30, 200)


    edged = gray
    contours, hierarchy = cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    
    ##print("Number of Contours found = " + str(len(contours))) 
    
    #time.sleep( 5 )
    #756 231 53 51
    #expected position    
    #cv2.rectangle(res, (ex_x,ex_y), (ex_x+ex_w, ex_y+ex_h), (255,0,0),2) 

    cv2.drawContours(res, contours, -1, (0, 255, 0), 3) 
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.circle(res,(640,320),10,(0,0,255),2)

    #cv2.imshow('Contours', image) 
    
    # Display the resulting frame


    # Set our filtering parameters 
    # Initialize parameter settiing using cv2.SimpleBlobDetector 
##    params = cv2.DSimpleBlobDetector_Params() 
##      
##    # Set Area filtering parameters 
##    params.filterByArea = True
##    params.minArea = 40
##      
##    # Set Circularity filtering parameters 
##    params.filterByCircularity = True 
##    params.minCircularity = 0.9
##      
##    # Set Convexity filtering parameters 
##    params.filterByConvexity = True
##    params.minConvexity = 0.2
##          
##    # Set inertia filtering parameters 
##    params.filterByInertia = True
##    params.minInertiaRatio = 0.01
##      
##    # Create a detector with the parameters 
##    detector = cv2.SimpleBlobDetector_create(params) 
##          
##    # Detect blobs 
##    keypoints = detector.detect(gray_blurred) 
##      
##    # Draw blobs on our image as red circles 
##    blank = np.zeros((1, 1))  
##    blobs = cv2.drawKeypoints(gray_blurred, keypoints, blank, (0, 0, 255), 
##                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
##      
##    number_of_blobs = len(keypoints)
##    print(number_of_blobs)
##    text = "Number of Circular Blobs: " + str(len(keypoints)) 
##    cv2.putText(blobs, text, (20, 550), 
##                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
##

    center_x = (x+w)/2
    center_y = (y+h)/2
    x_error = abs(640-x)
    y_error = abs(320-y)


    cv2.imshow('frame',res)
    #print (x,y,w,h)
    if(x_error < 30 and y_error < 30): print("at the front")
    else: print("pivit by", 640-x," ",320-y)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
