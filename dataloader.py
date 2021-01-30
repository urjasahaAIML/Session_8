"""
This is the data loader module
"""
import torch
import torchvision
import torchvision.transforms as transforms

import hyperparameters
from hyperparameters import * 

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # normalize image
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         transforms.RandomHorizontalFlip(),  
                         transforms.CenterCrop(32),                        
                         transforms.ToTensor(), 
                         transforms.Normalize(*stats)])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

# loaded only when loaddata() invoked
trainset = None
trainloader = None
testset = None
testloader = None

def loaddata():     
    global trainset, trainloader, testset, testloader, train_transform, test_transform #globals
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2) 
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2) 
    

 
