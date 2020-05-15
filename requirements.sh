#!/bin/bash

sudo apt-get update -y
sudo apt-get install build-essential cmake -y
sudo apt-get install libopenblas-dev liblapack-dev -y
sudo apt-get install libx11-dev libgtk-3-dev -y
sudo apt-get install python python-dev python-pip -y
sudo apt-get install python3 python3-dev python3-pip -y
sudo apt-get upgrade -y

pip3 install -r requirements.txt

# Downloading requirements
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1twcUs1sZSMwD3XHAtPtPfLivqwgIUSC2' -O 'encodings.pickle'
wget https://www.dropbox.com/s/15oxq46b0qzflkw/yolo.zip
unzip yolo.zip && rm yolo.zip

# additional package for some rare dependency error on firebase
python -m pip install pycrypto