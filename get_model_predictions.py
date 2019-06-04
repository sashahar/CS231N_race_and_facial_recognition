import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import sampler
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import numpy as np
import os
from CNN_architecture import CNN
from custom_dataset_loader import gender_race_dataset
import pandas as pd

use_gpu = torch.cuda.is_available()

#which model do you want the predictions for?
model = "cnn"
outfile = "adversarial_cnn_v1.csv"


def generate_predictions(cnn,data_loader):
    preds = np.array([])
    correct = np.array([])
    filenames = np.array([])
    races = np.array([])
    num_correct,num_sample = 0, 0
    for gender_labels, race_labels, img_names, images in data_loader:
        gender_labels = torch.from_numpy(np.asarray(gender_labels))
        race_labels = np.asarray(race_labels)
        races = np.append(races, race_labels)
        filenames = np.append(filenames, np.asarray(img_names))
        if use_gpu:
            images = Variable(images).cuda()
            labels = gender_labels.cuda()
        outputs,_ = cnn(images)
        _,pred = torch.max(outputs.data,1)
        num_sample += labels.size(0)
        num_correct += (pred == labels).sum()
        correct = np.append(correct, labels.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
        print("Validation Predictions - accuracy: ", sum(preds==correct)/len(correct))

    df = pd.DataFrame(np.concatenate((np.expand_dims(filenames, axis = 1), np.expand_dims(correct, axis=1), np.expand_dims(preds, axis=1), np.expand_dims(races, axis = 1)), axis =1), columns = ["Filenames", "label", "pred", "race"])
    df.to_csv(outfile, header=True)


test_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(227),
	transforms.ToTensor()
	])

print('Loading images...')
batch_size = 50
root='UTKFace/val'

val_data = gender_race_dataset("val_labels_all.csv", root, test_transform)
val_loader = torch.utils.data.DataLoader(val_data,
	batch_size=batch_size,shuffle=False)

if model == "cnn":
    print("Using CNN")

    cnn = CNN()
    if use_gpu:
        cnn.cuda()
    optimizer = torch.optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)

    SAVED_MODEL_PATH = 'cnn_model_best_SGD_adversary.pth.tar'
    checkpoint = torch.load(SAVED_MODEL_PATH)
    cnn.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    
    print("best model saved from epoch: ", epoch)

    print("best val_acc = ", best_val_acc)

    val_acc = generate_predictions(cnn,val_loader)
    
elif model == "vgg":
    print("Using VGG")

    NUM_CLASSES = 2
    vgg16 = models.vgg16(pretrained = True)
    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, NUM_CLASSES)]) # Add our layer with 4 outputs
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    if use_gpu:
        vgg16.cuda()

    optimizer = torch.optim.SGD(vgg16.parameters(),lr=0.001,momentum=0.9)

    SAVED_MODEL_PATH = 'vgg_checkpoint.pth.tar'
    checkpoint = torch.load(SAVED_MODEL_PATH)
    vgg16.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']

    print("best model saved from epoch: ", epoch)

    print("best val_acc = ", best_val_acc)

    val_acc = generate_predictions(vgg16,val_loader)
