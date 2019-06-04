import torch
import time
import copy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import sampler
from torch.optim import lr_scheduler
from torch.autograd import Variable

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import numpy as np
from custom_dataset_loader import gender_race_dataset
from CNN_architecture import NN, MyResnet
#from gender_vgg import check_acc

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("USING GPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

##########################
# For Grad Cam
def get_classtable():
    classes = []
    with open("samples/gender_labels.txt") as lines:
        for line in lines:
            classes.append(line)
    return classes

def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

###################
def save_checkpoint(state,is_best,file_name = 'resnet_checkpoint.pth.tar'):
	torch.save(state,file_name)
	if is_best:
		shutil.copyfile(file_name,'resnet_model_best.pth.tar')

batch_size = 50
validation_split = .2
shuffle_dataset = True
random_seed= 42

print('Loading images...')
train_data = gender_race_dataset("train_labels_all.csv", "UTKFace/train", train_transform)
test_data = gender_race_dataset("val_labels_all.csv", "UTKFace/val", test_transform)

dataset_sizes = {"train": len(train_data), "val": len(test_data)}

train_loader = torch.utils.data.DataLoader(train_data,
	batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_data,
	batch_size=batch_size,shuffle=False, num_workers=4)

NUM_CLASSES = 2
print("Number of Training Classes: {}".format(NUM_CLASSES))

# model = models.resnet18(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = True

#print(model)

#Source of code below:

# Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model.fc.out_features
# model = nn.Sequential(model, nn.Linear(num_ftrs, 1000), nn.ReLU(), nn.Linear(1000, 2))

model = MyResnet()
model = model.to(device)

criterion = nn.CrossEntropyLoss()

adversary = NN()
adversary = adversary.to(device)
nn_criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

dataloaders = {"train": train_loader, "val": test_loader}

outfile = "resnet_v1.csv"

def generate_predictions(cnn,data_loader):
    preds = np.array([])
    correct = np.array([])
    filenames = np.array([])
    races = np.array([])
    num_correct,num_sample = 0, 0
    for gender_labels, race_labels, img_names, images in data_loader:
        print("BATCH")
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
    
    #########################
    # GRAD CAM
    classes = get_classtable()
    
    #Get image paths
    image_paths = []
    q = 0
    for female_file in os.listdir(os.path.join(root, "female")):
        image_paths.append(os.path.join(root, os.path.join("female",str(female_file))))
        q += 1
        if (q == 10):
            break
    q = 0 
    for male_file in os.listdir(os.path.join(root, "male")):
        image_paths.append(os.path.join(root, os.path.join("male",str(male_file))))
        q += 1
        if (q == 10):
            break
    print(len(image_paths))
    
    # preprocess each image
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images)  #.to(device)

    # Here we choose the last convolution layer TODO: This will likely be wrong!
    target_layer = "exit_flow.conv4"
    target_class = 1 

    # run grad cam on all images!
    gcam = GradCAM(model=cnn)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images))  #.to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            # Make the target class male. The second half of the images are all of men. Please excuse the hack. 
            if j > (len(images)/2):
                target_class = 0
                
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "VGG16Adv", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

def train_model(model, criterion, adversary, nn_criterion, learning_rate, num_epochs=35):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    alpha = 1.0
    
    for epoch in range(num_epochs):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
        nn_optimizer = torch.optim.SGD(adversary.parameters(), lr=learning_rate, momentum=0.9)
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            i = 0
            for gender_labels, race_labels, img_names, images in dataloaders[phase]:
                inputs = images.to(device)
                labels = gender_labels.to(device)
                race_labels = race_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, pen_weights = model(inputs)
                    race_preds = adversary(pen_weights)
                    nn_loss = nn_criterion(race_preds, race_labels)
                    
                    _, preds = torch.max(outputs, 1)
                    resnet_loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss = resnet_loss - alpha*nn_loss
                        loss.backward(retain_graph = True)
                        optimizer.step()
                        
                        nn_optimizer.zero_grad()
                        nn_loss.backward()
                        nn_optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if (i+1) % 20 == 0:
                    print ('%s Epoch [%d/%d], Iter [%d/%d] Resnet Loss: %.4f Adversary Loss: %.4f'
                        %(phase, epoch+1, num_epochs, i+1, 380, resnet_loss, nn_loss))
                i=i+1
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if epoch % 5 == 0:
                learning_rate = learning_rate * 0.9

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                 phase, epoch_loss, epoch_acc))

                # deep copy the model
            if phase == 'val':
                is_best = epoch_acc > best_acc
                best_acc = max(epoch_acc,best_acc)
                save_checkpoint(
                        {'epoch':epoch+1,
                        'state_dict':model.state_dict(),
                        'best_val_acc':best_acc,
                        'optimizer':optimizer.state_dict()}, is_best)
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
learning_rates = [0.002]
    
for lr in learning_rates:
    print("Testing learning rate ", lr)
    model = train_model(model, criterion, adversary, nn_criterion, lr)
    
generate_predictions(model,test_loader)