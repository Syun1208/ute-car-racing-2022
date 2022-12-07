import os
import socket
import sys
import json
import base64
from pathlib import Path
from weights.load_models import loadModels
from utils.controller import *
from utils.traffic_signs_detection import *
from utils.image_processing import *
import argparse

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
port = 54321

# connect to the server on local computer
s.connect(('127.0.0.1', port))
angle = 10
speed = 100

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)
global angle, speed


def set_angle_speed(sendBackAngle, sendBackSpeed):
    global angle, speed
    angle = sendBackAngle
    speed = sendBackSpeed
    return angle, speed


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-seg', type=str, help='initial weights path', default='weights/segmentation/unet_v2')
    parser.add_argument('--weight-det', type=str, help='initial weights path', default='weights/detection/best_m.pt')
    parser.add_argument('--weight-rec', type=str, help='initial weights path',
                        default='weights/recognition/weight6_sum.pth')
    return parser.parse_args()


def load_weights():
    args = parse_arg()
    trainedModel = loadModels()
    trainedSegmentation = trainedModel.loadUNET(args.weight_seg)
    trainedDetection = trainedModel.loadYOLO(args.weight_det)
    trainedRecognition = trainedModel.loadCNN(args.weight_rec)
    return trainedSegmentation, trainedDetection, trainedRecognition


def main():
    # count = 0
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
    global angle, speed
    trainedSegmentation, trainedDetection, trainedRecognition = load_weights()
    try:
        while True:
            # Gửi góc lái và tốc độ để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Receive data from server
            data = s.recv(100000)
            # print(data)
            data_recv = json.loads(data)

            # Angle and speed recv from server
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            print("angle: ", current_angle)
            print("speed: ", current_speed)
            print("---------------------------------------")
            # Img data recv from server
            '''-------------------------GET IMAGE FROM THE MAP-----------------------'''
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            image = cv2.imdecode(jpg_as_np, flags=1)
            cv2.imshow("IMG", image)
            key = cv2.waitKey(1)
            print("Img Shape: ", image.shape)
            '''----------------------------IMAGE PROCESSING--------------------------'''
            # Save image
            # if not os.path.exists("/home/long/Desktop/UTECar-PHOLOTINO/datasets/"):
            #     os.makedirs("/home/long/Desktop/UTECar-PHOLOTINO/datasets/")
            # image_name = "/home/long/Desktop/UTECar-PHOLOTINO/datasets/ute_frame_{}.jpg".format(count)
            # if count % 10 == 0:
            #     cv2.imwrite(image_name, image)
            # count += 1
            # key = cv2.waitKey(1)
            if not flag_timer:
                start = time.time()
                # Segmentation
                modelSegmentation = segmentation(image)
                mask = modelSegmentation(trainedSegmentation)
                # Enhance mask after segmentation
                enhancedMask = imageProcessing(mask)
                mask = enhancedMask()
                # Real-time processing
                frame += 1
                if frame % 1 == 0:
                    # Detection
                    modelDetection = detection(image)
                    out_sign = modelDetection(trainedDetection, trainedRecognition)
                if carFlag == 0:
                    if 50 <= frame < 100:
                        fpsArray[frame - 50] = fps
                    elif 100 <= frame < 120:
                        noneArray = np.zeros(int(np.mean(fpsArray) * reset_seconds))
                        carArray = noneArray[1:int(len(noneArray) / 2)]
                    elif frame > 150:
                        if out_sign == "none" or out_sign is None:
                            noneArray[1:] = noneArray[0:-1]
                            noneArray[0] = 0

                        else:
                            noneArray[1:] = noneArray[0:-1]
                            noneArray[0] = 1

                        if np.sum(noneArray) == 0:
                            out_sign = "straight"
                elif carFlag == 1:
                    if out_sign == "none" or out_sign is None or out_sign == "unknown":
                        carArray[1:] = carArray[0:-1]
                        carArray[0] = 0

                    else:
                        carArray[1:] = carArray[0:-1]
                        carArray[0] = 1

                    if np.sum(carArray) == 0:
                        out_sign = "straight"
            pre_Signal = Signal_Traffic
            if out_sign != "unknown" and out_sign is not None and out_sign != "none":
                if out_sign == "car_left" or out_sign == "car_right":
                    carFlag = 1
                else:
                    carFlag = 0
                Signal_Traffic = out_sign
            '''---------------------------CONTROLLER---------------------------'''
            # Code anh Tuong
            Signal_Traffic, speed, error, flag_timer = Control_Car(enhancedMask, out_sign, Signal_Traffic,
                                                                   current_speed)
            angle = -PID(error, 0.40, 0, 0.05)
            angle, speed = set_angle_speed(angle, speed)
            end = time.time()
            fps = 1 / (end - start)
    finally:
        print('closing socket')
        s.close()


if __name__ == "__main__":
    main()
