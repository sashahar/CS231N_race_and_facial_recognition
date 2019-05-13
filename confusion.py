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
import os
#from gender import CNN

use_gpu = torch.cuda.is_available()


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

def confusion_matrix(cnn,data_loader, filenames):
	preds = np.array([])
	correct = np.array([])
	num_correct,num_sample = 0, 0
	for images,labels in data_loader:
		images = Variable(images).cuda()
		labels = labels.cuda()
		outputs = cnn(images)
		_,pred = torch.max(outputs.data,1)
		num_sample += labels.size(0)
		num_correct += (pred == labels).sum()
		correct = np.append(correct, labels.cpu().numpy())
		preds = np.append(preds, pred.cpu().numpy())
		print(len(correct))
		print(len(preds))
	np.savetxt("predictions.csv", (filenames, correct, preds), fmt = '%s', delimiter=',')


test_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(227),
	transforms.ToTensor()
	])

print('Loading images...')
batch_size = 50
root='UTKFace/val'
filenames = []
for male_file in os.listdir(os.path.join(root, "male")):
	filenames.append(str(male_file))
for female_file in os.listdir(os.path.join(root, "female")):
	filenames.append(str(female_file))
print(len(filenames))
val_data = dsets.ImageFolder(root=root, transform =test_transform)

val_loader = torch.utils.data.DataLoader(val_data,
	batch_size=batch_size,shuffle=False)

cnn = CNN()
if use_gpu:
	cnn.cuda()
optimizer = torch.optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)

SAVED_MODEL_PATH = 'model_best.pth.tar'
checkpoint = torch.load(SAVED_MODEL_PATH)
cnn.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
best_val_acc = checkpoint['best_val_acc']

val_acc = confusion_matrix(cnn,val_loader, np.array(filenames)) #val_data)
