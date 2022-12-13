#!/bin/zsh
echo '-----------------------------HELLO TEAM PHOLOTINO------------------------------------'
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install scikit-learn
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
pip install -r requirements.txt
echo '-----------------------------DOWNLOADING WEIGHTS------------------------------------'
#segmentation = './weights/segmentation'
#detection = './weights/detection'
#recognition = './weights/recognition'
#url_segmentation =
#url_detection =
#url_recognition =
#mkdir $segmentation
#mkdir $detection
#mkdir $segmentation
#gdown
#gdown
#gdown
python client.py
