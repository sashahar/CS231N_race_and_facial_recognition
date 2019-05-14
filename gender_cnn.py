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

use_gpu = torch.cuda.is_available()

def confusion_matrix(cnn,data_loader):
	#four groups: white women, non-white women, white men, non-white men
	#for each group
	num_sample = np.zeros(4)

	num_correct,num_sample = 0, 0
	for images,labels in data_loader:
		images = Variable(images).cuda()
		labels = labels.cuda()
		outputs = cnn(images)

		_,pred = torch.max(outputs.data,1)
		num_sample += labels.size(0)
		num_correct += (pred == labels).sum()
	print("WHITE WOMEN")
	print("WHITE MEN")

def check_acc(cnn,data_loader):
	num_correct,num_sample = 0, 0
	for images,labels in data_loader:
		images = Variable(images).cuda()
		labels = labels.cuda()
		outputs = cnn(images)

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
def save_checkpoint(state,is_best,file_name = 'checkpoint.pth.tar'):
	torch.save(state,file_name)
	if is_best:
		shutil.copyfile(file_name,'model_best.pth.tar')

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
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_data,
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
	batch_size=batch_size,shuffle=False)

NUM_CLASS = len(train_loader.dataset.classes)
print("Number of Training Classes: {}".format(NUM_CLASS))



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

cnn = CNN()
if use_gpu:
	cnn.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)

loss_history = []
num_epochs = 100
train_acc_history = []
val_acc_history = []
epoch_history = []
learning_rate = 0.001
best_val_acc = 0.0


for epoch in range(num_epochs):
	optimizer = torch.optim.SGD(cnn.parameters(),lr=learning_rate,momentum=0.9)
	print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
	print('Learning Rate for this epoch: {}'.format(learning_rate))

	for i,(images,labels) in enumerate(train_loader):
		images = Variable(images)
		labels = Variable(labels)
		if use_gpu:
			images,labels = images.cuda(),labels.cuda()

		pred_labels = cnn(images)
		loss = criterion(pred_labels,labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 5 == 0:
			print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
            	%(epoch+1, num_epochs, i+1, len(train_data)//50, loss.data))

	if True or epoch % 10 ==0 or epoch == num_epochs-1:
		learning_rate = learning_rate * 0.9

		train_acc = check_acc(cnn,train_loader)
		train_acc_history.append(train_acc)
		print('Train accuracy for epoch {}: {} '.format(epoch + 1,train_acc))

		val_acc = check_acc(cnn,test_loader)
		val_acc_history.append(val_acc)
		print('Validation accuracy for epoch {} : {} '.format(epoch + 1,val_acc))
		epoch_history.append(epoch+1)
		plot_performance_curves(train_acc_history,val_acc_history,epoch_history)

		is_best = val_acc > best_val_acc
		best_val_acc = max(val_acc,best_val_acc)
		save_checkpoint(
			{'epoch':epoch+1,
			'state_dict':cnn.state_dict(),
			'best_val_acc':best_val_acc,
			'optimizer':optimizer.state_dict()},is_best)
