import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import sampler
from torch.optim import lr_scheduler
from torch.autograd import Variable
import shutil
import numpy as np
from custom_dataset_loader import gender_race_dataset
from CNN_architecture import NN, MyVgg

#Code adapted from: https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch

use_gpu = torch.cuda.is_available()
if use_gpu:
    print(torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 	for images,labels in data_loader:
def check_acc(model,data_loader):
    num_correct,num_sample = 0, 0
    for gender_labels, race_labels, img_names, images in data_loader:
        gender_labels = torch.from_numpy(np.asarray(gender_labels))
        if use_gpu:
            images = Variable(images.cuda())
            labels = Variable(gender_labels.cuda())
        outputs, pen_weights = model(images)
        _,pred = torch.max(outputs.data,1)
        num_sample += labels.size(0)
        num_correct += (pred == labels).sum()
        torch.cuda.empty_cache()
    return float(num_correct)/num_sample

def save_checkpoint(state,is_best,file_name = 'vgg_checkpoint.pth.tar'):
    torch.save(state,file_name)
    print(is_best)
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


print('Loading images...')
train_data = gender_race_dataset("train_labels_all.csv", "UTKFace/train", train_transform)
val_data = gender_race_dataset("val_labels_all.csv", "UTKFace/val", test_transform)
test_data = gender_race_dataset("test_labels_all.csv", "UTKFace/test", test_transform)


batch_size = 50
validation_split = .2
shuffle_dataset = True
random_seed= 42


train_loader = torch.utils.data.DataLoader(train_data,
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
	batch_size=batch_size,shuffle=False)
val_loader = torch.utils.data.DataLoader(val_data,
	batch_size=batch_size,shuffle=False)

NUM_CLASSES = 2

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

if use_gpu:
    print("Using GPU for adversary")
    adversary.cuda()

loss_history = []
num_epochs = 15
train_acc_history = []
val_acc_history = []
epoch_history = []
learning_rate = 0.001
best_val_acc = 0.0

#def train_model(vgg16, criterion, optimizer, adversary, nn_criterion, nn_optimizer, num_epochs=10):
def train_model(criterion, nn_criterion, num_epochs=10):
    best_val_acc = 0.0
    alpha = 1.0
    myVGG = MyVgg()
    if use_gpu:
        myVGG.cuda()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(myVGG.parameters(), lr=learning_rate, momentum=0.9)

    print("Accuracy before training: ", check_acc(myVGG, val_loader))

    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))

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
            #train_msg = 'Train accuracy for epoch {}: {} '.format(epoch + 1,train_acc)
			#print(train_msg)
            #epoch_history.append(train_msg)

			val_acc = check_acc(vgg16,test_loader)
			val_acc_history.append(val_acc)
			val_msg = 'Validation accuracy for epoch {}: {} '.format(epoch + 1,val_acc)
            print(val_msg)
            epoch_history.append(val_msg)
			best_val_acc = max(val_acc,best_val_acc)

			is_best = val_acc > best_val_acc
			save_checkpoint(
				{'epoch':epoch+1,
				'state_dict':vgg16.state_dict(),
				'best_val_acc':best_val_acc,
				'optimizer':optimizer.state_dict()},is_best)
	np.savetxt("training_log_vgg.out", epoch_history, fmt='%s')

train_model(vgg16, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
