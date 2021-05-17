import rospy, cv2
import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

class NumberCruncher:
    
    # Constructor
    # @self    this object every object function has to have self.
    def __init__(self):
        super(NumberCruncher, self).__init()

        