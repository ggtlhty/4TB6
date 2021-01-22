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
        aisle_info = Store.locateProducts(Bluetooth.msgStr)
        Bluetooth.sendAisleInfo(aisle_info)
        Bluetooth.clearMsgStr()
    else:
        Bluetooth.msgStr += msg_rcv
    Bluetooth.serial.flush()

    
    
    
    
    
    