from bluetooth import *
from qrcode import *

QRCode = QRCode()
Bluetooth = Bluetooth()
totalPrice = 0

while True:
    # get the image
    if QRCode.checkQRCode():
        price = QRCode.detectedData
        print(price)
        totalPrice += int(price)
        Bluetooth.updatePrice(price, totalPrice)
    if(cv2.waitKey(1) == ord("q")):
        break
    
    
# free camera object and exit
QRCode.cam.release()
cv2.destroyAllWindows()
    