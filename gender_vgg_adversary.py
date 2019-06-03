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
from CNN_architecture import NN, MyVgg

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("USING GPU")
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

def plot_performance_curves(train_acc_history,val_acc_history,epoch_history):
    plt.figure()
    plt.plot(np.array(epoch_history),np.array(train_acc_history),label = 'Training accuracy')
    plt.plot(np.array(epoch_history),np.array(val_acc_history),label = 'Validation accuracy')
    plt.title('Accuracy on training and validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of epochs')
    plt.legend()
    plt.savefig('acc_recode.png')

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
test_data = gender_race_dataset("val_labels_all.csv", "UTKFace/val", test_transform)


batch_size = 50
validation_split = .2
shuffle_dataset = True
random_seed= 42


train_loader = torch.utils.data.DataLoader(train_data,
	batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
	batch_size=batch_size,shuffle=False)

NUM_CLASSES = 2

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)

adversary = NN(4096)
if use_gpu:
    print("Using GPU for adversary")
    adversary.cuda()
    
nn_criterion = nn.CrossEntropyLoss()
nn_optimizer = torch.optim.SGD(adversary.parameters(), lr=0.001, momentum=0.9)

loss_history = []
num_epochs = 15
train_acc_history = []
val_acc_history = []
epoch_history = []
learning_rate = 0.001
best_val_acc = 0.0

#def train_model(vgg16, criterion, optimizer, adversary, nn_criterion, nn_optimizer, num_epochs=10):
def train_model(criterion, adversary, nn_criterion, nn_optimizer, num_epochs=10):
    best_val_acc = 0.0
    alpha = 1.0
    myVGG = MyVgg()
    if use_gpu:
        print("Using GPU for VGG16")
        myVGG.cuda()
    learning_rate = 0.001
    optimizer = torch.optim.SGD(myVGG.parameters(), lr=learning_rate, momentum=0.9)
    
    print("Accuracy before training: ", check_acc(myVGG, test_loader))
#     print("Accuracy before training: ", check_acc(vgg16, test_loader))
    
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        #print('Learning Rate for this epoch: {}'.format(learning_rate))
        
        i = 0
        for gender_labels, race_labels, img_names, images in train_loader:
            gender_labels = torch.from_numpy(np.asarray(gender_labels))
            race_labels = torch.from_numpy(np.asarray(race_labels))
            images = Variable(images)
            labels = Variable(gender_labels)
            if use_gpu:
                images,labels, race_labels = Variable(images.cuda()),Variable(labels.cuda()), Variable(race_labels.cuda())

            optimizer.zero_grad()

            pred_labels, pen_weights = myVGG(images)
            #######################################################
            nn_pred_labels = adversary(pen_weights)
            nn_loss = nn_criterion(nn_pred_labels, race_labels)
            vgg_loss = criterion(pred_labels,labels)
            loss = vgg_loss - alpha*nn_loss
            loss.backward(retain_graph = True)
            optimizer.step()
            
            #loss for adversary model
            nn_optimizer.zero_grad()
            nn_loss.backward()
            nn_optimizer.step()
            
            del images, labels, pred_labels, nn_pred_labels, race_labels
            torch.cuda.empty_cache()
            #######################################################

            if (i+1) % 5 == 0:
# 				print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
# 					%(epoch+1, num_epochs, i+1, len(train_data)//50, loss.data))
                print ('Epoch [%d/%d], Iter [%d/%d] Vgg Loss: %.4f Adversary Loss: %.4f'
                    %(epoch+1, num_epochs, i+1, len(train_data)//50, vgg_loss, nn_loss))
            i = i + 1

        if True:
            if epoch % 5 ==0 or epoch == num_epochs-1:
                learning_rate = learning_rate * 0.2

            train_acc = check_acc(myVGG,train_loader)
            train_acc_history.append(train_acc)
            train_msg = 'Train accuracy for epoch {}: {} '.format(epoch + 1,train_acc)
            print(train_msg)
            epoch_history.append(train_msg)
#             print('Train accuracy for epoch {}: {} '.format(epoch + 1,train_acc))

            val_acc = check_acc(myVGG,test_loader)
            val_acc_history.append(val_acc)
            val_msg = 'Validation accuracy for epoch {} : {} '.format(epoch + 1,val_acc)
            print(val_msg)
#             print('Validation accuracy for epoch {} : {} '.format(epoch + 1,val_acc))
            epoch_history.append(val_msg)
            # plot_performance_curves(train_acc_history,val_acc_history,epoch_history)
            is_best = val_acc > best_val_acc
            best_val_acc = max(val_acc,best_val_acc)      
        
        is_best = val_acc > best_val_acc
        save_checkpoint(
                {'epoch':epoch+1,
                'state_dict':myVGG.state_dict(),
                'best_val_acc':best_val_acc,
                'optimizer':optimizer.state_dict()},is_best)
            
        np.savetxt("training_log_vgg.out", epoch_history, fmt='%s')


train_model(criterion, adversary, nn_criterion, nn_optimizer, num_epochs=30)
