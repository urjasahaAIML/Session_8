"""
This module contains functions to test the model
"""
import hyperparameters
from hyperparameters import * 
import model_resnet
from model_resnet import * 
import dataloader
from dataloader import *
import training
from training import *

# test the network with test data
def testmodel(trainedmodel):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in dataloader.testloader:
            images, labels = data
            images = images[:,0:1,:,:]
            images=images.to(device)
            labels=labels.to(device)
            outputs = trainedmodel(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))

# Displays what % of classes are classified/misclassified
def stats_classified(trainedmodel):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader.testloader:
            images, labels = data
            images = images[:,0:1,:,:]
            images=images.to(device)
            labels=labels.to(device)
            outputs = trainedmodel(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))