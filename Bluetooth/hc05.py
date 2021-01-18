import serial
import time
ser = serial.Serial("/dev/serial0", 38400)

while True:
    data = ser.read()
    print(data.decode())
    ser.flush()
    # ser.write(data)
    # ser.write(str(counter%3).encode("utf-8"))
# time.sleep(3)
# print("connection lost")
# ser.write("connection lost".encode("utf-8"))
