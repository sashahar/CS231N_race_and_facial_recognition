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
from CNN_architecture import CNN, MyVgg
from custom_dataset_loader import gender_race_dataset
import pandas as pd

# TODO: We might need to move grad_cam.py or a copy of it into the main CS231N_race_and_facial_recognition folder
from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

use_gpu = torch.cuda.is_available()

#which model do you want the predictions for?
model = "cnn"
outfile = "predictions_adversarial_cnn_best.csv"

##########################
# For Grad Cam
def get_classtable():
    classes = []
    with open("grad-cam-pytorch/samples/gender_labels.txt") as lines:
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
        outputs,_,_ = cnn(images)
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

test_data = gender_race_dataset("test_labels_all.csv", 'UTKFace/test', test_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=False)

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

#     val_acc = generate_predictions(cnn,val_loader)
    test_acc = generate_predictions(cnn, test_loader)
    
elif model == "vgg":
    print("Using VGG")

#     NUM_CLASSES = 2
#     vgg16 = models.vgg16(pretrained = True)
#     # Newly created modules have require_grad=True by default
#     num_features = vgg16.classifier[6].in_features
#     features = list(vgg16.classifier.children())[:-1] # Remove last layer
#     features.extend([nn.Linear(num_features, NUM_CLASSES)]) # Add our layer with 4 outputs
#     vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
    myVGG = MyVgg()
    
    if use_gpu:
        myVGG.cuda()

    optimizer = torch.optim.SGD(myVGG.parameters(),lr=0.001,momentum=0.9)

    SAVED_MODEL_PATH = 'vgg_adversary_best_model.pth.tar'
    checkpoint = torch.load(SAVED_MODEL_PATH)
    myVGG.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_val_acc']

    print("best model saved from epoch: ", epoch)

    print("best val_acc = ", best_val_acc)

    val_acc = generate_predictions(myVGG,val_loader)
    
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
    gcam = GradCAM(model=myVGG)
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
