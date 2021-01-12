import socket, time

UDP_IP = "192.168.0.100"
UDP_PORT = 5005
to_address = (UDP_IP, UDP_PORT)
counter = 0
msg = str(counter).encode()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
  sock.sendto(msg, to_address)
  counter += 1
  msg = str(counter).encode()
  time.sleep(1)
