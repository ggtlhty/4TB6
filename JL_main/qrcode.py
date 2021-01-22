import cv2
import time
import serial


class QRCode:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.detector = cv2.QRCodeDetector()
        self.detectedData = None
        self.waiting = False
        self.LOOP = 2
        self.counter = 0
        self.serial = serial.Serial("/dev/serial0", 38400)
    
    def checkQRCode(self):
        # get the image
        _, img = self.cam.read()
        # get bounding box coords and data
        data, bbox, _ = self.detector.detectAndDecode(img)
        
        # if there is a bounding box, draw one, along with the data
        if(bbox is not None) and not self.waiting:
            #for i in range(len(bbox)):
            #    cv2.line(img, tuple(bbox[i][0]),
            #             tuple(bbox[(i+1) % len(bbox)][0]),
            #             color=(255, 0, 255), thickness=2)
            cv2.putText(img, data, (int(bbox[0][0][0]),
                                    int(bbox[0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            if data:
                # print("data found: ", data)
                self.detectedData = data
                self.waiting = True
                self.counter = 0
                # print("waiting")
                cv2.imshow("code detector", img)
                # msg = "price: " + data
                # self.serial.write(msg.encode())
                return True
        elif self.waiting:
            if self.counter >=self.LOOP:
                self.waiting = False
                self.counter = 0
            else:
                self.counter += 1
                print("counting:", self.counter)
        else:
            pass
        # display the image preview
        cv2.imshow("code detector", img)
        return False
    
    
    def processData(self):
        print(self.data)
    
    