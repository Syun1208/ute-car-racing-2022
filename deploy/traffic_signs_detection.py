import torch
import cv2
import numpy as np

signArray = np.zeros(15)


def check_sign(signName, indexMinSign):
    classes = ['straight', 'turn_left', 'turn_right', 'no_turn_left', 'no_turn_right', 'no_straight', 'unknown']
    new_cls_id = [i + 1 for i, value in enumerate(classes) if signName == classes[i]]
    signArray[1:] = signArray[0:-1]
    signArray[0] = int(new_cls_id[0])
    num_cls_id = np.zeros(6)
    for i in range(6):
        num_cls_id[i] = np.count_nonzero(signArray == (i + 1))

    max_num = num_cls_id[0]
    pos_max = 0
    for i in range(6):
        if max_num < num_cls_id[i]:
            max_num = num_cls_id[i]
            pos_max = i

    if max_num >= indexMinSign:
        signName = classes[pos_max]
    else:
        signName = "none"
    return signName


class recognition:
    def __init__(self, image):
        self.image = image
        self.classes = ['straight', 'turn_left', 'turn_right', 'no_turn_left', 'no_turn_right', 'no_straight',
                        'unknown']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, model, *args, **kwargs):
        self.image = cv2.resize(self.image, (64, 64))
        self.image = self.image / 255
        self.image = self.image.astype('float32')
        self.image = self.image.transpose(2, 0, 1)
        self.image = torch.from_numpy(self.image).unsqueeze(0)
        with torch.no_grad():
            self.image = self.image.to(self.device)
            mask = model(self.image)
            _, pred = torch.max(mask, 1)
            pred = pred.data.cpu().numpy()
            class_pred = self.classes[pred[0]]
            mask = mask[0].cpu()
        return class_pred


class detection:
    def __init__(self, image):
        self.image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (320, 320))

    def __call__(self, modelYOLO, modelCNN, *args, **kwargs):
        # Results in YOLOv5
        results = modelYOLO(self.image)
        if len(results.pandas().xyxy[0]) != 0:
            if (float(results.pandas().xyxy[0].confidence[0])) >= 0.85:

                x_min = int(results.xyxy[0][0][0])
                y_min = int(results.xyxy[0][0][1])
                x_max = int(results.xyxy[0][0][2])
                y_max = int(results.xyxy[0][0][3])

                x_c = int(x_min + (x_max - x_min) / 2)
                y_c = int(y_min + (y_max - y_min) / 2)
                # Compute area of bounding box
                s_bbox = (x_max - x_min) * (y_max - y_min)
                # Cutting ROI based on bounding box
                img_classifier = self.image[y_min:y_max, x_min:x_max]
                # Classify traffic signs
                modelClassification = recognition(img_classifier)
                sign = modelClassification(modelCNN)

                if 30 < s_bbox < 250:
                    if results.pandas().xyxy[0].name[0] == 'unknown' or sign == "unknown":
                        return "none"
                    else:
                        return "decrease"


                elif 250 <= s_bbox <= 1200 and y_min > 10 and float(
                        results.pandas().xyxy[0].confidence[0]) > 0.88:  # and y_min > 10 and x_max < 270:
                    cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
                    if sign != "unknown":
                        sign_checked = check_sign(sign, 2)
                        if sign_checked != "none":
                            return sign_checked
                        else:
                            return "unknown"
                    else:
                        return "unknown"

                elif s_bbox >= 1000:  # 1200
                    if results.pandas().xyxy[0].name[0] == "car":
                        s_max = 0
                        x_max = 0
                        y_max = 0
                        for i in range(len(results.xyxy[0])):
                            x1 = int(results.xyxy[0][0][0])
                            y1 = int(results.xyxy[0][0][1])
                            x2 = int(results.xyxy[0][0][2])
                            y2 = int(results.xyxy[0][0][3])
                            s = (x2 - x1) * (y2 - y1)
                            if s > s_max:
                                s_max = s
                                x_max = x2
                                y_max = y2
                        if x_max >= 160 and y_max > 150:
                            return "car_right"
                        else:
                            return "car_left"

            else:
                return "none"
        else:
            signArray[1:] = signArray[0:-1]
            signArray[0] = 0
            return "none"
