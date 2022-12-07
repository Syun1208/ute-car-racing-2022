import torch
from models.UNET import build_unet
from models.CNN import Network


class loadModels:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def loadUNET(self, weight):
        predictedUNET = build_unet()
        predictedUNET = predictedUNET.to(self.device)
        predictedUNET.load_state_dict(torch.load(weight, map_location=self.device))
        predictedUNET.eval()
        return predictedUNET

    def loadYOLO(self, weight):
        predictedYOLO = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
        predictedYOLO.to(self.device)
        predictedYOLO.eval()
        return predictedYOLO

    def loadCNN(self, weight):
        modelCNN = Network()
        modelCNN.to(self.device)
        modelCNN.load_state_dict(torch.load(weight, map_location=self.device))
        modelCNN.eval()
        return modelCNN
