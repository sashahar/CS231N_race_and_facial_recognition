#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import os
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
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
        self.fc1 = nn.Linear(384*6*6,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,2)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        #print out.size()
        out = F.dropout(F.relu(self.fc1(out)))
        out = F.dropout(F.relu(self.fc2(out)))
        out = self.fc3(out)

        return out

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

    # if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False

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
    
def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable():
    classes = []
    '''
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    '''
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


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)

def save_checkpoint(state,is_best,file_name = 'vgg_checkpoint.pth.tar'):
	torch.save(state,file_name)
	if is_best:
		shutil.copyfile(file_name,'vgg_model_best.pth.tar')

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


# torchvision models
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo1(image_paths, topk, output_dir, cuda):
    
    # Get image paths
    root='../UTKFace/val'
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

    
    """
    CNN Gradcam 
    """
    """
    
    # Synset words
    classes = get_classtable()
    
    print("Using CNN")

    cnn = CNN()
    #if use_gpu:
    #    cnn.cuda()
    optimizer = torch.optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)

    SAVED_MODEL_PATH = '../cnn_model_best_final.pth.tar'
    checkpoint = torch.load(SAVED_MODEL_PATH,  map_location='cpu')
    cnn.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    
    print("best model saved from epoch: ", epoch)

    print("best val_acc = ", best_val_acc)

    cnn.eval()

    # The four residual layers
    target_layers = ["layer3"]
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
            
            #print(
            #    "\t#{}: {} ({:.5f})".format(
            #        j, classes[target_class], float(probs[ids == target_class])
            #    )
           # )
            if j > (len(images)/2):
                target_class = 0
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "CNN", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )
    """
    """
    # ============================================================================================
    Adversary CNN 
    """
    """
    classes = get_classtable()
    
    print("Using Adversary CNN")
    
    cnn = CNN()
    optimizer = torch.optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)
     
    SAVED_MODEL_PATH ='../cnn_model_best_SGD_adversary.pth.tar'
    checkpoint = torch.load(SAVED_MODEL_PATH,  map_location='cpu')
    cnn.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    
    print("best model saved from epoch: ", epoch)

    print("best val_acc = ", best_val_acc)

    cnn.eval()

    # The four residual layers
    target_layers = ["layer3"]
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
            
            #print(
            #    "\t#{}: {} ({:.5f})".format(
            #        j, classes[target_class], float(probs[ids == target_class])
            #    )
           # )
            # Make the target class male. The second half of the images are all of men. Please excuse the hack. 
            if j > (len(images)/2):
                target_class = 0
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "CNNAdv", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )
    """
    """
    # =============================================================================================
    VGG16 Gradcam 
    """
    """
    # TODO: waiting on .tar file. 
    #device = get_device(cuda)

    # Synset words
    classes = get_classtable()
    
    # make model 
    NUM_CLASSES = 2
    vgg16 = models.vgg16(pretrained = True)
    # Freeze training for all layers NOTE: may or may not need this
    for param in vgg16.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, NUM_CLASSES)]) # Add our layer with 4 outputs
    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    optimizer = torch.optim.SGD(vgg16.parameters(),lr=0.001,momentum=0.9) # dont know why you need this when you're just recreating a model that's been already trained... 

    SAVED_MODEL_PATH = '../vgg_checkpoint.pth.tar'
    checkpoint = torch.load(SAVED_MODEL_PATH)
    vgg16.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    vgg16.eval() # the gcam has this eval in their code. idk if we need it though
    
    # Check available layer names
    print("Layers:")
    for m in model.named_modules():
        print("\t", m[0])

    # Here we choose the last convolution layer TODO: This will likely be wrong!
    target_layer = "exit_flow.conv4"
    target_class = 0 # TODO: CHANGE THIS!!!
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
            """
            #print(
            #    "\t#{}: {} ({:.5f})".format(
            #        j, classes[target_class], float(probs[ids == target_class])
            #    )
            #)
    """
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "VGG16", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

    
    """
    """
    # =============================================================================================
    VGG16 Adversarial Gradcam
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    # =============================================================================================
    Resnet Gradcam
    """
    """
    # Synset words
    classes = get_classtable()
    
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True

    #print(model)

    #Source of code below:

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.out_features
    model = nn.Sequential(model, nn.Linear(num_ftrs, 1000), nn.ReLU(), nn.Linear(1000, 2))
    learning_rate = 0.002
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
    
    SAVED_MODEL_PATH = '../resnet_model_best_v1.pth.tar'
    checkpoint = torch.load(SAVED_MODEL_PATH,  map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    
    print("best model saved from epoch: ", epoch)

    print("best val_acc = ", best_val_acc)
    
    model.eval() # the gcam has this eval in their code. idk if we need it though 
    
    # Check available layer names
    print("Layers:")
    for m in model.named_modules():
        print("\t", m[0])

    # Here we choose the last convolution layer TODO: This will likely be wrong!
    target_layers = ["0.layer4.1.bn2"] # ["layer4"] # why did they originally do "relu", "layer1", "layer2", "layer3", 
    target_class = 1 
    # run grad cam on all images!
    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images))  #.to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            """
            #print(
            #    "\t#{}: {} ({:.5f})".format(
            #        j, classes[target_class], float(probs[ids == target_class])
            #    )
            #)
    """
            # Make the target class male. The second half of the images are all of men. Please excuse the hack. 
            if j > (len(images)/2):
                target_class = 0
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "RESNET", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )
    """
    """
    # =============================================================================================
    Resnet Adversarial Gradcam
    """
    
    learning_rate = 0.002
    model = MyResnet()
    criterion = nn.CrossEntropyLoss()
    adversary = NN(1000)
    nn_criterion = nn.CrossEntropyLoss()
#     resnet = models.resnet18(pretrained=True)
#     for param in resnet.parameters():
#         param.requires_grad = True
#     num_ftrs = resnet.fc.out_features
#     fc1 = nn.Sequential(nn.Linear(num_ftrs, 1000),
#                              nn.ReLU(),
#                              nn.Linear(1000,2))

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)

    # Synset words
    classes = get_classtable()
    
    SAVED_MODEL_PATH = '../resnet_model_best.pth.tar'
    checkpoint = torch.load(SAVED_MODEL_PATH,  map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']
    
    print("best model saved from epoch: ", epoch)

    print("best val_acc = ", best_val_acc)
    
    model.eval() # the gcam has this eval in their code. idk if we need it though 
    
    # Check available layer names
    print("Layers:")
    for m in model.named_modules():
        print("\t", m[0])

    # Here we choose the last convolution layer TODO: This will likely be wrong!
    target_layers = ["resnet.layer4.1.bn2"] # ["layer4"] # why did they originally do "relu", "layer1", "layer2", "layer3", 
    target_class = 1
    # run grad cam on all images!
    gcam = GradCAM(model=model)
    print(images[0])
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
                        j, "RESNET", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )
    
    
    
    
    
    

   
    """
    Common usage:
    1. Wrap your model with visualization classes defined in grad_cam.py
    2. Run forward() with images
    3. Run backward() with a list of specific classes
    4. Run generate() to export results
    """
'''
    # =========================================================================
    print("Vanilla Backpropagation:")

    bp = BackPropagation(model=model)
    print("model ", model)
    probs, ids = bp.forward(images)
    print("probs ",probs) 
    print("ids ", ids)
    for i in range(topk):
        # In this example, we specify the high confidence classes
        
        bp.backward(ids=ids[:, [i]])
        gradients = bp.generate()

        # Save results as image files
        for j in range(len(images)):
            print("ids[j, i] ", ids[j, i])
            print("probs[j, i] ",probs[j, i])
            print(" classes[ids[j, i]] ",  classes[ids[j, i]])
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-vanilla-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    # Remove all the hook function in the "model"
    bp.remove_hook()

    # =========================================================================
    print("Deconvolution:")

    deconv = Deconvnet(model=model)
    _ = deconv.forward(images)

    for i in range(topk):
        deconv.backward(ids=ids[:, [i]])
        gradients = deconv.generate()

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-deconvnet-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

    deconv.remove_hook()
'''
    # =========================================================================
"""
    print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

    gcam = GradCAM(model=vgg16)
    _ = gcam.forward(images)

    gbp = GuidedBackPropagation(model=vgg16)
    _ = gbp.forward(images)

    for i in range(topk):
        # Guided Backpropagation
        gbp.backward(ids=ids[:, [i]])
        gradients = gbp.generate()

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Guided Backpropagation
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided-{}.png".format(j, arch, classes[ids[j, i]]),
                ),
                gradient=gradients[j],
            )

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )

            # Guided Grad-CAM
            save_gradient(
                filename=osp.join(
                    output_dir,
                    "{}-{}-guided_gradcam-{}-{}.png".format(
                        j, arch, target_layer, classes[ids[j, i]]
                    ),
                ),
                gradient=torch.mul(regions, gradients)[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo2(image_paths, output_dir, cuda):
    """
   #  Generate Grad-CAM at different layers of ResNet-152
"""

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = 243  # "bull mastif"

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo3(image_paths, topk, output_dir, cuda):
    """
    # Generate Grad-CAM with original models
"""

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Third-party model from my other repository, e.g. Xception v1 ported from Keras
    model = torch.hub.load(
        "kazuto1011/pytorch-ported-models", "xception_v1", pretrained=True
    )
    model.to(device)
    model.eval()

    # Check available layer names
    print("Layers:")
    for m in model.named_modules():
        print("\t", m[0])

    # Here we choose the last convolution layer
    target_layer = "exit_flow.conv4"

    # Preprocessing
    def _preprocess(image_path):
        raw_image = cv2.imread(image_path)
        raw_image = cv2.resize(raw_image, model.image_shape)
        image = torch.FloatTensor(raw_image[..., ::-1].copy())
        image -= model.mean
        image /= model.std
        image = image.permute(2, 0, 1)
        return image, raw_image

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = _preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    print("Grad-CAM:")

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)

    for i in range(topk):

        # Grad-CAM
        gcam.backward(ids=ids[:, [i]])
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

            # Grad-CAM
            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "xception_v1", target_layer, classes[ids[j, i]]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


@main.command()
@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-k", "--topk", type=int, default=3)
@click.option("-s", "--stride", type=int, default=1)
@click.option("-b", "--n-batches", type=int, default=128)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo4(image_paths, arch, topk, stride, n_batches, output_dir, cuda):
    """
    # Generate occlusion sensitivity maps
"""

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # Model from torchvision
    model = models.__dict__[arch](pretrained=True)
    model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()

    # Images
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    images = torch.stack(images).to(device)

    print("Occlusion Sensitivity:")

    patche_sizes = [10, 15, 25, 35, 45, 90]

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    probs, ids = probs.sort(dim=1, descending=True)

    for i in range(topk):
        for p in patche_sizes:
            print("Patch:", p)
            sensitivity = occlusion_sensitivity(
                model, images, ids[:, [i]], patch=p, stride=stride, n_batches=n_batches
            )

            # Save results as image files
            for j in range(len(images)):
                print("\t#{}: {} ({:.5f})".format(j, classes[ids[j, i]], probs[j, i]))

                save_sensitivity(
                    filename=osp.join(
                        output_dir,
                        "{}-{}-sensitivity-{}-{}.png".format(
                            j, arch, p, classes[ids[j, i]]
                        ),
                    ),
                    maps=sensitivity[j],
                )
"""

if __name__ == "__main__":
    main()
