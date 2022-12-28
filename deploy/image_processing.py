import torch
import cv2
import numpy as np

global h_lines
global h_max
global r_lines
global r_max
global l_lines
global l_max


class imageProcessing:
    def __init__(self, mask):
        self.mask = mask

    def __removeSmallContours(self):
        image_binary = np.zeros((self.mask.shape[0], self.mask.shape[1]), np.uint8)
        contours = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        masked = cv2.drawContours(image_binary, [max(contours, key=cv2.contourArea)], -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(self.mask, self.mask, mask=masked)
        return image_remove

    def __reduceNoise(self):
        # self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        self.mask = self.__fill_small_zeros_mask_area()
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        bg = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, se)
        self.mask = cv2.divide(self.mask, bg, scale=255)
        # self.mask = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
        # self.mask = self.__removeSmallContours()
        return self.mask

    def __fill_small_zeros_mask_area(self):
        self.mask = cv2.GaussianBlur(self.mask, (3, 3), 0)
        self.mask = cv2.bitwise_not(self.mask)
        mask_remove = self.__removeSmallContours()
        output = cv2.bitwise_not(mask_remove)
        return output

    def __call__(self, *args, **kwargs):
        self.mask = self.__reduceNoise()
        return self.mask


class segmentation:
    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, model, *args, **kwargs):
        self.image = self.image[125:, :]
        self.image = cv2.resize(self.image, (160, 80))
        x = torch.from_numpy(self.image)
        x = x.to(self.device)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x / 255.0
        x = x.unsqueeze(0).float()
        with torch.no_grad():
            # pretrainedUNET = weights(x)
            predictImage = model(x)
            predictImage = torch.sigmoid(predictImage)
            predictImage = predictImage[0]
            predictImage = predictImage.squeeze()
            predictImage = predictImage > 0.5
            predictImage = predictImage.cpu().numpy()
            predictImage = np.array(predictImage, dtype=np.uint8)
            predictImage = predictImage * 255
        return predictImage
