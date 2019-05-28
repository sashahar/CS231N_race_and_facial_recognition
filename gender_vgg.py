import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import sampler
from torch.optim import lr_scheduler
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import numpy as np
from custom_dataset_loader import gender_race_dataset

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("USING GPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def check_acc(model,data_loader):
    num_correct,num_sample = 0, 0
    for gender_labels, race_labels, img_names, images in data_loader:
        gender_labels = torch.from_numpy(np.asarray(gender_labels))
        images = Variable(images).cuda()
        labels = gender_labels.cuda()
        outputs = model(images)

        _,pred = torch.max(outputs.data,1)
        num_sample += labels.size(0)
        num_correct += (pred == labels).sum()
        torch.cuda.empty_cache()
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

def save_checkpoint(state,is_best,file_name = 'vgg_checkpoint_v3.pth.tar'):
	torch.save(state,file_name)
	if is_best:
		shutil.copyfile(file_name,'vgg_model_best_v3.pth.tar')

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



# model = resnet18(pretrained=True)
# #for param in model.parameters():
# #   param.requires_grad = False

# # Parameters of newly constructed modules have requires_grad=True by default
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, NUM_CLASS)
# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained = True)


# Freeze training for all layers
for param in vgg16.features.parameters():
    param.require_grad = False

num_features1 = vgg16.classifier[3].in_features
num_features2 = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-4] # Remove last layer
features.extend([nn.Linear(num_features1, num_features2), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(num_features2, NUM_CLASSES)]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

# vgg16.classifier = nn.Sequential(
#     nn.Linear(3 * 32 * 32, hidden_layer_size),
#     nn.ReLU(),
#     nn.Linear(hidden_layer_size, 10),)

if use_gpu:
    vgg16.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(vgg16.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


loss_history = []
train_acc_history = []
val_acc_history = []
epoch_history = []

def train_model(vgg16, criterion, optimizer, scheduler, num_epochs=10):
	best_val_acc = 0.0

	print("Accuracy before training: ", check_acc(vgg16, test_loader))

	for epoch in range(num_epochs):
        #optimizer = torch.optim.SGD(vgg16.parameters(),lr=learning_rate,momentum=0.9)
		print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        #print('Learning Rate for this epoch: {}'.format(learning_rate))

		vgg16.train(True)

        #for gender_labels, race_labels, img_names, images in train_loader:
		for i,(labels, race_labels, img_names, images) in enumerate(train_loader):
			images = Variable(images)
			labels = Variable(labels)
			if use_gpu:
				images,labels = Variable(images.cuda()),Variable(labels.cuda())

			optimizer.zero_grad()
			pred_labels = vgg16(images)
			loss = criterion(pred_labels,labels)

			loss.backward()
			optimizer.step()
			torch.cuda.empty_cache()

			if (i+1) % 5 == 0:
				print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
					%(epoch+1, num_epochs, i+1, len(train_data)//50, loss.data))

		vgg16.train(False)
		vgg16.eval()
		if epoch % 5 ==0 or epoch == num_epochs-1:

			train_acc = check_acc(vgg16,train_loader)
			train_acc_history.append(train_acc)
            train_msg = 'Train accuracy for epoch {}: {} '.format(epoch + 1,train_acc)
			print(train_msg)
            epoch_history.append(train_msg)

			val_acc = check_acc(vgg16,test_loader)
			val_acc_history.append(val_acc)
			val_msg = 'Validation accuracy for epoch {}: {} '.format(epoch + 1,val_acc)
            print(val_msg)
            epoch_history.append(val_msg)
			#plot_performance_curves(train_acc_history,val_acc_history,epoch_history)
			best_val_acc = max(val_acc,best_val_acc)

			is_best = val_acc > best_val_acc
			save_checkpoint(
				{'epoch':epoch+1,
				'state_dict':vgg16.state_dict(),
				'best_val_acc':best_val_acc,
				'optimizer':optimizer.state_dict()},is_best)
	np.savetxt("training_log_vgg_v3.out", epoch_history, fmt='%s')

train_model(vgg16, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
