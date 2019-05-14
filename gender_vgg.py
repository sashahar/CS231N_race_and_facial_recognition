import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from  torchvision.models import resnet18
from torch.utils.data import sampler
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import numpy as np

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_acc(model,data_loader):
	num_correct,num_sample = 0, 0
	for images,labels in data_loader:
		images = Variable(images).cuda()
		labels = labels.cuda()
		outputs = model(images)

		_,pred = torch.max(outputs.data,1)
		num_sample += labels.size(0)
		num_correct += (pred == labels).sum()
	return float(num_correct)/num_sample

def plot_performance_curves(train_acc_history,val_acc_history,epoch_history):
	plt.figure()
	plt.plot(np.array(epoch_history),np.array(train_acc_history),label = 'Training accuracy')
	plt.plot(np.array(epoch_history),np.array(val_acc_history),label = 'Validation accuracy')
	plt.title('Accuracy on training and validation')
	plt.ylabel('Accuracy')
	plt.xlabel('Number of epochs')
	plt.legend()
	plt.savefig('acc_recode.png')

def save_checkpoint(state,is_best,file_name = 'resnet_checkpoint.pth.tar'):
	torch.save(state,file_name)
	if is_best:
		shutil.copyfile(file_name,'resnet_model_best.pth.tar')

train_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomCrop(227),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()
	])
test_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(227),
	transforms.ToTensor()
	])


print('Loading images...')
train_data = dsets.ImageFolder(root='UTKFace/train',transform = train_transform)
test_data = dsets.ImageFolder(root='UTKFace/val',transform =test_transform)

batch_size = 50
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
#dataset_size = len(dataset)
#indices = list(range(dataset_size))
#split = int(np.floor(validation_split * dataset_size))
#if shuffle_dataset :
    #np.random.seed(random_seed)
    #np.random.shuffle(indices)
#train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
#train_sampler = SubsetRandomSampler(train_indices)
#valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_data,
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
	batch_size=batch_size,shuffle=False)

NUM_CLASS = len(train_loader.dataset.classes)
print("Number of Training Classes: {}".format(NUM_CLASS))



model = resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASS)


model= model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

loss_history = []
num_epochs = 30
train_acc_history = []
val_acc_history = []
epoch_history = []
learning_rate = 0.001
best_val_acc = 0.0


for epoch in range(num_epochs):
	optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
	print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
	print('Learning Rate for this epoch: {}'.format(learning_rate))

	for i,(images,labels) in enumerate(train_loader):
		images = Variable(images)
		labels = Variable(labels)
		if use_gpu:
			images,labels = images.cuda(),labels.cuda()

		pred_labels = model(images)
		loss = criterion(pred_labels,labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 5 == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
            	%(epoch+1, num_epochs, i+1, len(train_data)//50, loss.data))

	if True or epoch % 7 ==0 or epoch == num_epochs-1:
		learning_rate = learning_rate * 0.1

		train_acc = check_acc(model,train_loader)
		train_acc_history.append(train_acc)
		print('Train accuracy for epoch {}: {} '.format(epoch + 1,train_acc))

		val_acc = check_acc(model,test_loader)
		val_acc_history.append(val_acc)
		print('Validation accuracy for epoch {} : {} '.format(epoch + 1,val_acc))
		epoch_history.append(epoch+1)
		plot_performance_curves(train_acc_history,val_acc_history,epoch_history)

		is_best = val_acc > best_val_acc
		best_val_acc = max(val_acc,best_val_acc)
		save_checkpoint(
			{'epoch':epoch+1,
			'state_dict':model.state_dict(),
			'best_val_acc':best_val_acc,
			'optimizer':optimizer.state_dict()},is_best)
