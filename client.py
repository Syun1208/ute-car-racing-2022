import os
import socket
import sys
import json
import base64
from pathlib import Path
from weights.load_models import loadModels
from deploy.controller import *
import logging
import yaml
import torch
from pygame import mixer
from multiprocessing import Process
from deploy.traffic_signs_detection import *
from deploy.image_processing import *
import argparse
import requests

torch.cuda.set_device(0)
# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
port = 54321

# connect to the server on local computer
s.connect(('127.0.0.1', port))
global angle, speed
angle = 0
speed = 200
Signal_Traffic = 'straight'
pre_Signal = 'straight'
noneArray = np.zeros(50)
fpsArray = np.zeros(50)
carArray = np.zeros(50)
reset_seconds = 1.0
fps = 20
carFlag = 0
frame = 0
out_sign = "straight"
flag_timer = 0
file = open('./config/ute_car_v1.yaml', 'r')
data_yaml = yaml.full_load(file)
'''--------------------------------------------------------------------'''
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


def set_angle_speed(sendBackAngle, sendBackSpeed):
    global angle, speed
    angle = sendBackAngle
    speed = sendBackSpeed


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-seg', type=str, help='initial weights path',
                        default=data_yaml['segmentation']['weight_path'])
    parser.add_argument('--weight-det', type=str, help='initial weights path',
                        default=data_yaml['detection']['weight_path'])
    parser.add_argument('--weight-rec', type=str, help='initial weights path',
                        default=data_yaml['recognition']['weight_path'])
    parser.add_argument('--url', type=str, help='url downloading music',
                        default='https://stream.nixcdn.com/NhacCuaTui2029/DoanTuyetNangDiFrexsRemix-PhatHuyT4-7022889.mp3?st=SQkSrqlEECJcdathICmHdg&e=1666795517')
    return parser.parse_args()


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {:.2f}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


def playMusicPHOLOTINO(url, name_music):
    try:
        downloaded_file_location = ROOT / name_music
        r = requests.get(url)
        with open(downloaded_file_location, 'wb') as f:
            f.write(r.content)
        mixer.init()
        mixer.music.load(ROOT / name_music)
        mixer.music.set_volume(0.5)
        mixer.music.play()
    except Exception as bug:
        logging.error(bug)
        mixer.init()
        mixer.music.load(ROOT / name_music)
        mixer.music.set_volume(0.5)
        mixer.music.play()


def load_weights():
    args = parse_arg()
    trainedModel = loadModels()
    trainedSegmentation = trainedModel.loadUNET(args.weight_seg)
    print('[INFO]: DONE IN LOADING SEGMENTATION')
    # trainedDetection = trainedModel.loadYOLO(args.weight_det)
    # print('[INFO]: DONE IN LOADING DETECTION')
    # trainedRecognition = trainedModel.loadCNN(args.weight_rec)
    # print('[INFO]: DONE IN LOADING RECOGNITION')
    print('[INFO]: CHIẾN THÔI !')
    return trainedSegmentation


def main():
    global angle, speed
    data_recv = {}
    trainedSegmentation = load_weights()
    try:
        while True:
            # Gửi góc lái và tốc độ để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Receive data from server
            data = s.recv(1000000000)
            # print(data)
            try:
                data_recv = json.loads(data)
            except Exception as error:
                logging.error(error)
                continue

            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            # print("angle: ", current_angle)
            # print("speed: ", current_speed)
            # print("---------------------------------------")
            # Img data recv from server
            '''-------------------------GET IMAGE FROM THE MAP-----------------------'''
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
            print('-----------------------VÀ ĐÂY LÀ PHOLOTINO-------------------------')
            '''----------------------------IMAGE PROCESSING--------------------------'''
            start = time.time()
            modelSegmentation = segmentation(image)
            mask = modelSegmentation(trainedSegmentation)
            # Enhance mask after segmentation
            try:
                enhancedMask = imageProcessing(mask)
                mask = enhancedMask()
            except Exception as error:
                logging.error(error)
                pass
            # Controller
            controller = Controller(mask, float(current_speed))
            angle, speed, minLane, maxLane, center = controller()
            set_angle_speed(angle, speed)
            end = time.time()
            if data_yaml['parameters']['show_image']:
                try:
                    fps = 1 / (end - start)
                    image = show_fps(image, fps)
                    cv2.circle(mask, (minLane, 50), radius=5, color=(0, 0, 0), thickness=5)
                    cv2.circle(mask, (maxLane, 50), radius=5, color=(0, 0, 0), thickness=5)
                    cv2.line(mask, (center, 50), (mask.shape[1] // 2, mask.shape[0]), color=(0, 0, 0), thickness=5)
                    cv2.imshow("IMG", image)
                    cv2.imshow("Mask", mask)
                    key = cv2.waitKey(1)
                except Exception as bug:
                    logging.error(bug)
                    pass
    finally:
        print('closing socket')
        s.close()


if __name__ == "__main__":
    args = parse_arg()
    if data_yaml['mode']['music']:
        try:
            p1 = Process(target=playMusicPHOLOTINO(args.url, 'pholotino.mp3'))
            p1.start()
            p2 = Process(target=main())
            p2.start()
            p1.join()
            p2.join()
            p1.terminate()
            p2.terminate()
        except Exception as error:
            logging.error(error)
            main()
    else:
        main()
