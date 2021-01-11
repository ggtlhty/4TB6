follow instructions 1a to 1c in the below website:
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md

open Raspberry Pi terminal

do:
sudo apt-get update
sudo apt-get dist-upgrade

enable camera:
top left corner of Raspberry Pi desktop -> Preferences -> Pi Configuration -> Interfaces -> Enable Camera

back to the terminal, do:
git clone https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi.git
mv TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi tflite1
cd tflite1
sudo pip3 install virtualenv
python3 -m venv tflite1-env
source tflite1-env/bin/activate
bash get_pi_requirements.sh

python3 TFLite_detection_webcam.py --modeldir=3BarModelV2 --resolution=300x300 (run the python file)
