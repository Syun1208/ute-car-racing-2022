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

    def __removeSmallContours(self, max_area=300):
        image_binary = np.zeros((self.mask.shape[0], self.mask.shape[1]), np.uint8)
        contours = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        filteredContours = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > max_area:
                filteredContours.append(i)

        mask = cv2.drawContours(image_binary, filteredContours, -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(self.mask, self.mask, mask=mask)
        return image_remove

    @staticmethod
    def remove_small_contours(image, max_area=300):
        image_binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        filteredContours = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > max_area:
                filteredContours.append(i)

        mask = cv2.drawContours(image_binary, filteredContours, -1, (255, 255, 255), -1)
        image_remove = cv2.bitwise_and(image, image, mask=mask)
        return image_remove

    def fill_small_zeros_mask_area(self, mask):
        invert_mask = cv2.bitwise_not(mask)
        mask_remove = self.remove_small_contours(invert_mask)
        output = cv2.bitwise_not(mask_remove)
        return output

    @staticmethod
    def reset():
        global h_lines
        global h_max
        global r_lines
        global r_max
        global l_lines
        global l_max
        h_max = 0
        h_lines = 0
        r_max = 0
        r_lines = 0
        l_max = 0
        l_lines = 0

    @staticmethod
    def find_line(p1, p2, img):
        global h_lines
        global h_max
        global r_lines
        global r_max
        global l_lines
        global l_max
        if p1[0] != p2[0]:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            intercept = p1[1] - p1[0] * slope
            angle = np.arctan(slope) * (180 / np.pi)
            d = np.sqrt(pow((p1[0] - p2[0]), 2) + pow((p1[1] - p2[1]), 2))
            if intercept != 79:
                # find horizon lines:
                if d > h_max and 15 > angle > -15:
                    h_max = d
                    h_lines = [(slope, intercept, angle, d), (p1, p2)]

                if d > r_max and 15 < angle < 60:
                    r_max = d
                    r_lines = [(slope, intercept, angle, d), (p1, p2)]

                if d > l_max and -60 < angle < -15:
                    l_max = d
                    l_lines = [(slope, intercept, angle, d), (p1, p2)]

    @staticmethod
    def __(mask, line):
        if line != 0:
            _, ph = line
            slope = _[0]
            if slope == 0:
                slope = 1e-5
            intercept = _[1]
            y1 = 0
            x1 = int(-intercept / slope)
            y2 = 80
            x2 = int((80 - intercept) / slope)
            cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), 2)
            return mask

    def crop_mask(self, mask, h_lines, r_lines, l_lines, sign):
        o_mask = mask.copy()
        if sign == 'straight':
            # mask = __(mask, l_lines)
            # mask = __(mask, r_lines)
            mask = mask
        elif sign == 'turn_right':
            mask = self.__(mask, l_lines)
            mask = self.__(mask, h_lines)
        elif sign == 'turn_left':
            mask = self.__(mask, r_lines)
            mask = self.__(mask, h_lines)
        elif sign == 'no_turn_right':
            mask = self.__(mask, r_lines)
        elif sign == 'no_turn_left':
            mask = self.__(mask, l_lines)
        elif sign == 'no_straight':
            mask = self.__(mask, h_lines)

        if np.array_equal(o_mask, mask):
            if sign == "turn_right" or sign == "no_turn_left":
                cv2.line(mask, (35, 0), (35, 80), (0, 0, 0), 5)
            if sign == "turn_left" or sign == "no_turn_right":
                cv2.line(mask, (130, 0), (130, 80), (0, 0, 0), 5)
        mask = self.remove_small_contours(mask, 3000)
        return mask

    def remove_noise_mask(self, _mask, sign, e=0.005):
        self.reset()
        mask = _mask.copy()
        # mask = cv2.imread("corner_mask_4" + ".png", 0)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = self.fill_small_zeros_mask_area(mask)
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]

        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        epsilon = e * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        points = []
        for p in approx:
            points.append((p[0][0], p[0][1]))

        for p in range(len(points) - 1):
            if p == 0:
                self.find_line(points[0], points[len(points) - 1], mask)
            else:
                self.find_line(points[p], points[p + 1], mask)

        new_mask = self.crop_mask(mask, h_lines, r_lines, l_lines, sign)
        return new_mask

    def __call__(self, *args, **kwargs):
        self.mask = self.__removeSmallContours()
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
