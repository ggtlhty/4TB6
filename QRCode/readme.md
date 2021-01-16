to generate QR Code in terminal using Python:

reference:
https://www.hackster.io/gatoninja236/scan-qr-codes-in-real-time-with-raspberry-pi-a5268b

in terminal, do:
pip3 install opencv-python
pip3 install qrcode

go into python3 in terminal by typing "python3"
in python3 termina, do:

import qrcode;
qrcode.make('hello').save('hello.png')
