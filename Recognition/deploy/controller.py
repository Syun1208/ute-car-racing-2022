from deploy.image_processing import imageProcessing
import time
from sklearn.ensemble import RandomForestRegressor
import numpy as np

t = time.time()
error_arr = np.zeros(5)
list_angle = np.zeros(5)


class Controller(imageProcessing):
    def __init__(self, image, maxSpeed, trafficSigns, Car):
        # super(Controller, self).__init__(image, trafficSigns)
        if not trafficSigns:
            trafficSigns = [-1]
        self.trafficSigns = list(['camtrai', 'camphai', 'camthang', 'trai', 'phai', 'thang', 'none'])[
            int(trafficSigns[0])]
        imageProcessing.__init__(self, image, self.trafficSigns, Car)
        self.mask, self.scale = imageProcessing.__call__(self)
        # self.mask = image
        self.current_speed = Car.getSpeed_rad()
        self.Car = Car
        self.maxSpeed = maxSpeed

    def __reduceSpeed(self, speed):
        if self.current_speed > 20:
            return 0
        else:
            return speed

    def findingLane(self, scale=20):
        arr_normal = []
        height = self.mask.shape[0] - scale
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y.any() == 255:
                arr_normal.append(x)
        if not arr_normal:
            arr_normal = [self.mask.shape[1] * 1 // 3, self.mask.shape[1] * 2 // 3]
        minLane = min(arr_normal)
        maxLane = max(arr_normal)
        center = int((minLane + maxLane) / 2)
        width = maxLane - minLane
        if width < 55:
            if center < int(self.mask.shape[1] / 2):
                center -= 55 - width
            else:
                center += 55 - width
        error = int(self.mask.shape[1] / 2) - center
        return error

    def computeError(self, center):
        return int(self.mask.shape[1] / 2) - center

    @staticmethod
    def __PID(error, scale=1, p=0.43, i=0, d=0.05):
        global t
        global error_arr
        error_arr[1:] = error_arr[0:-1]
        error_arr[0] = error
        P = error * p
        delta_t = time.time() - t
        t = time.time()
        D = (error - error_arr[1]) / delta_t * d
        I = np.sum(error_arr) * delta_t * i
        angle = P + I + D
        # angle = self.__optimizeFuzzy(angle)
        if abs(angle) > 50:
            angle = np.sign(angle) * 50
        return - int(angle) * scale

    def __conditionalSpeed(self, error):
        list_angle[1:] = list_angle[0:-1]
        list_angle[0] = abs(error)
        # list_angle_train = np.array(list_angle).reshape((-1, 1))
        predSpeed = np.dot(list_angle, - 0.1) + self.maxSpeed
        # reg = LinearRegression().fit(list_angle_train, speed)
        # reg = RandomForestRegressor(n_estimators=40, random_state=1).fit(list_angle_train, predSpeed)
        # predSpeed = reg.predict(np.array(list_angle_train))
        return np.average(predSpeed)

    def __call__(self, *args, **kwargs):
        error = self.findingLane()
        print('Traffic Sign: ', self.trafficSigns)
        self.Car.OLED_Print('Traffic Sign: {}'.format(self.trafficSigns), 3)
        if not self.trafficSigns or self.trafficSigns != 'none' or self.trafficSigns != 'thang':
            error = self.findingLane(scale=30)
        angle = self.__PID(error, self.scale)
        speed = self.__conditionalSpeed(error)
        speed = self.__reduceSpeed(speed)
        # angle = angle * 60 / 25
        return angle, speed
