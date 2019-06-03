import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import sampler
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import numpy as np
import torchvision.models as models

#Notes: Add layers to the CNN architecture, play with number of filters and kernel size, etc. 
class CNN(nn.Module):
    def __init__(self, p=0.5):
        super(CNN,self).__init__()
        self.p = p
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,96,kernel_size=7,stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2))
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(384,440,kernel_size=3,padding=1),
#             nn.BatchNorm2d(440),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3,stride=1, padding=1))
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(440,512,kernel_size=3,padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3,stride=1, padding = 1))
        self.fc1 = nn.Linear(384*6*6,512) #was originally 384*6*6
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,2)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)
        out = out.view(out.size(0),-1)
        out_2 = F.dropout(F.relu(self.fc1(out)), p = self.p)
        penultimate_weights = F.dropout(F.relu(self.fc2(out_2)), p = self.p)
        out = self.fc3(penultimate_weights)
        return out, penultimate_weights, out_2

    
class NN(nn.Module):
    def __init__(self, features):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(features, 100) 
        #features Needs to be 1000 for resnet #Needs to be 4096 for vgg #Needs to be 512 when doing CNN
        self.fc2 = nn.Linear(100,2)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        
        return out

NUM_CLASSES = 2
class MyVgg(nn.Module):
    def __init__(self):
        super(MyVgg, self).__init__()
        vgg16 = models.vgg16(pretrained = True)
        num_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1]
        vgg16.classifier = nn.Sequential(*features)
        self.myModel = vgg16
        self.fc = nn.Linear(num_features,NUM_CLASSES, bias = True)
        
    def forward(self, images):
        out = self.myModel(images)
        x = self.fc(out)
        return x, out
    
class MyResnet(nn.Module):
    def __init__(self):
        super(MyResnet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = True
        self.num_ftrs = self.resnet.fc.out_features
        self.fc1 = nn.Sequential(nn.Linear(self.num_ftrs, 1000),
                                 nn.ReLU(),
                                 nn.Linear(1000,2))
        
    def forward(self, images):
        pen_weights = self.resnet(images)
        out = self.fc1(pen_weights)
        return out, pen_weights
        
    
    
    
    
    
    
 