import serial
import time

class Bluetooth:
    def __init__(self):
        self.baudrate = 38400
        self.port = "/dev/serial0"
        self.serial = serial.Serial(self.port, self.baudrate)
        self.msgStr = ""
    
    def sendAisleInfo(self, aisleDict):
        msg = ""
        for aisle in aisleDict:
            try:
                # aisle = int(aisle)
                msg += "aisle {}: ".format(int(aisle))
            except ValueError:
                msg += "{}: ".format(aisle)
            products = aisleDict[aisle]
            for product in products:
                msg += "{}, ".format(product)
            msg = msg[:-2] + "\n"
        msg = msg[:-1]
        print(msg)
        self.serial.write(msg.encode())
            
    
    def updatePrice(self, price, totalPrice):
        msg = str(price) + " " + str(totalPrice)
        self.serial.write(msg.encode())
        print('message "{}" sent'.format(msg))
    
    
    def clearMsgStr(self):
        self.msgStr = ""
        