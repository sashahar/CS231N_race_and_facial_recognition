import torch
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import sampler
from torch.optim import lr_scheduler
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import numpy as np
from custom_dataset_loader import gender_race_dataset
#from gender_vgg import check_acc

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("USING GPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()
	])
test_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor()
	])

def save_checkpoint(state,is_best,file_name = 'resnet_checkpoint_v1.pth.tar'):
	torch.save(state,file_name)
	if is_best:
		shutil.copyfile(file_name,'resnet_model_best_v1.pth.tar')

batch_size = 50
validation_split = .2
shuffle_dataset = True
random_seed= 42

print('Loading images...')
train_data = gender_race_dataset("train_labels_all.csv", "UTKFace/train", train_transform)
test_data = gender_race_dataset("val_labels_all.csv", "UTKFace/val", test_transform)
train_loader = torch.utils.data.DataLoader(train_data,
	batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data,
	batch_size=batch_size,shuffle=False, num_workers=4)

NUM_CLASSES = 2
print("Number of Training Classes: {}".format(NUM_CLASSES))

model = models.resnet18(pretrained=True)
for param in model.parameters():
  param.requires_grad = False

print(model)

#Source of code below:

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

dataloaders = {"train": train_loader, "val": test_loader}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for gender_labels, race_labels, img_names, images in dataloaders[phase]:
                inputs = images.to(device)
                labels = gender_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                is_best = epoch_acc > best_acc
                best_val_acc = max(epoch_acc,best_acc)
                save_checkpoint(
                    {'epoch':epoch+1,
                    'state_dict':cnn.state_dict(),
                    'best_val_acc':best_acc,
                    'optimizer':optimizer.state_dict()},is_best)
                if is_best:
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#MAIN
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
