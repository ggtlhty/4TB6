import serial
import time

class Bluetooth:
    def __init__(self):
        self.baudrate = 38400
        self.port = "/dev/serial0"
        self.serial = serial.Serial(self.port, self.baudrate)
        self.msg_str = ""
    
    def send_aisle_info(self, aisle_dict):
        msg = ""
        for aisle in aisle_dict:
            try:
                # aisle = int(aisle)
                msg += "aisle {}: ".format(int(aisle))
            except ValueError:
                msg += "{}: ".format(aisle)
            products = aisle_dict[aisle]
            for product in products:
                msg += "{}, ".format(product)
            msg = msg[:-2] + "\n"
        msg = msg[:-1]
        print(msg)
        self.serial.write(msg.encode())
            
    
    def clear_msg_str(self):
        self.msg_str = ""
        