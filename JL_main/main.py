import serial
import time
import os
from store import *
from bluetooth import *


#initliazation
Bluetooth = Bluetooth()
Store = Store("sobeys_1")


while True:
    #bluetooth module
    msg_rcv = Bluetooth.serial.read().decode()
    if msg_rcv == "]":
    # print(Bluetooth.msg_str)
        aisle_info = Store.locate_products(Bluetooth.msg_str)
        Bluetooth.send_aisle_info(aisle_info)
        Bluetooth.clear_msg_str()
    else:
        Bluetooth.msg_str += msg_rcv
    Bluetooth.serial.flush()