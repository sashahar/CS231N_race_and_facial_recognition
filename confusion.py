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
import gender

use_gpu = torch.cuda.is_available()

def confusion_matrix(cnn,data_loader):
	#four groups: white women, non-white women, white men, non-white men
	#for each group
	num_sample = np.zeros(4)

	num_correct,num_sample = 0, 0
	index = 0
	for images,labels in data_loader:
		if index < 10:
			print(labels)
		images = Variable(images).cuda()
		labels = labels.cuda()
		outputs = cnn(images)

		_,pred = torch.max(outputs.data,1)
		num_sample += labels.size(0)
		num_correct += (pred == labels).sum()
	print("WHITE WOMEN")
	print("WHITE MEN")

test_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(227),
	transforms.ToTensor()
	])

print('Loading images...')
val_data = dsets.ImageFolder(root='UTKFace/val',transform =test_transform)
for img in val_data:
	print(img)
val_loader = torch.utils.data.DataLoader(val_data,
	batch_size=batch_size,shuffle=False)

cnn = CNN()
optimizer = torch.optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)

SAVED_MODEL_PATH = 'model_best.pth.tar'
checkpoint = torch.load(SAVED_MODEL_PATH)
cnn.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
best_val_acc = checkpoint[best_val_acc]

val_acc = check_acc(cnn,val_loader)
