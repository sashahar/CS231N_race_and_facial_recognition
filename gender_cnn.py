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
from custom_dataset_loader import gender_race_dataset
from CNN_architecture import CNN, NN

use_gpu = torch.cuda.is_available()

def check_acc(cnn,data_loader):
    num_correct,num_sample = 0, 0
    for gender_labels, race_labels, img_names, images in data_loader:
        gender_labels = torch.from_numpy(np.asarray(gender_labels))
        images = Variable(images).cuda()
        labels = gender_labels.cuda()
        outputs, penultimate_weights = cnn(images)

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
    
def save_checkpoint(state,is_best,file_name = 'cnn_checkpoint.pth.tar'):
	torch.save(state,file_name)
	if is_best:
		shutil.copyfile(file_name,'cnn_model_best.pth.tar')

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
train_data = gender_race_dataset("train_labels_all.csv", "UTKFace/train", train_transform)
test_data = gender_race_dataset("val_labels_all.csv", "UTKFace/val", test_transform)


batch_size = 50
validation_split = .2
shuffle_dataset = True
random_seed= 42


train_loader = torch.utils.data.DataLoader(train_data,
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
	batch_size=batch_size,shuffle=False)

cnn = CNN()
if use_gpu:
    print('Using GPU for cnn')
    cnn.cuda()
        
adversary = NN()
if use_gpu:
    print('Using GPU for nn')
    adversary.cuda()
    

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)

nn_criterion = nn.CrossEntropyLoss()
nn_optimizer = torch.optim.SGD(adversary.parameters(), lr=0.001, momentum=0.9)

def train_model(cnn, adversary, criterion, nn_criterion, optimizer, nn_optimizer, num_epochs = 100):
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    epoch_history = []
    learning_rate = 0.001
    best_val_acc = 0.0
    alpha = 1.0

    for epoch in range(num_epochs):
        optimizer = torch.optim.SGD(cnn.parameters(),lr=learning_rate,momentum=0.9)
        nn_optimizer = torch.optim.SGD(adversary.parameters(),lr=learning_rate,momentum=0.9)
        
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        i = 0
        for gender_labels, race_labels, img_names, images in train_loader:
            gender_labels = torch.from_numpy(np.asarray(gender_labels))
            race_labels = torch.from_numpy(np.asarray(race_labels))
            images = Variable(images)
            labels = Variable(gender_labels)
            race_labels = Variable(race_labels)
            if use_gpu:
                images,labels = images.cuda(),labels.cuda()
                race_labels = race_labels.cuda()
        
            pred_labels, penultimate_weights = cnn(images)
            
            nn_pred_labels = adversary(penultimate_weights)

            nn_loss = nn_criterion(nn_pred_labels, race_labels)   
       
            cnn_loss = criterion(pred_labels,labels)
            
            #loss for gender prediction model
            loss = cnn_loss - alpha*nn_loss                           
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
            
            #loss for adversary model
            nn_optimizer.zero_grad()
            nn_loss.backward()
            nn_optimizer.step()

            if (i+1) % 5 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] CNN Loss: %.4f Adversary Loss: %.4f'
                    %(epoch+1, num_epochs, i+1, len(train_data)//50, loss.data, nn_loss.data))
            i = i + 1
                
        if epoch % 10 == 0:
            learning_rate = learning_rate * 0.9

        if epoch % 5 ==0 or epoch == num_epochs-1:
            train_acc = check_acc(cnn,train_loader)
            train_acc_history.append(train_acc)
            train_msg = 'Train accuracy for epoch {}: {} '.format(epoch + 1,train_acc)
            print(train_msg)
            epoch_history.append(train_msg)

            val_acc = check_acc(cnn,test_loader)
            val_acc_history.append(val_acc)
            val_msg = 'Validation accuracy for epoch {}: {} '.format(epoch + 1,val_acc)
            print(val_msg)
            epoch_history.append(val_msg)
            #plot_performance_curves(train_acc_history,val_acc_history,epoch_history)

            is_best = val_acc > best_val_acc
            best_val_acc = max(val_acc,best_val_acc)
            save_checkpoint(
                {'epoch':epoch+1,
                'state_dict':cnn.state_dict(),
                'best_val_acc':best_val_acc,
                'optimizer':optimizer.state_dict()},is_best)

    np.savetxt("training_log.out", epoch_history, fmt='%s')


train_model(cnn, adversary, criterion, nn_criterion, optimizer, nn_optimizer)       

        

