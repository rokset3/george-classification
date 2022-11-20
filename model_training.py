import cv2

import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import lr_scheduler

import glob
from tqdm import tqdm

import os
import numpy as np

import warnings
import random


#import utility functions
from my_utils import GeorgeDataset
from my_utils import generate_paths
from train import train_model

warnings.filterwarnings('ignore')

#config for training, you can change these values
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.0001 #for lower layers (wanted to optimize, but not much/could've considered non-trainable, but why not)
lr_finetune = 0.01 #for last-output layer (fc.shape = (512,2))
weight_decay = 0.0005 #for weight_decay
step_size = 10
gamma = 0.1
num_epochs = 1
#end of config

if __name__ =='__main__':

    PATH = os.getcwd()

    train_path = PATH + r'/Dataset/train'
    test_path = PATH + r'/Dataset/val'
    classes = ['george', 'no_george'] #we may want to change it for other classes

    generate_paths(classes, train_path, test_path) #to generate paths for data

    #Transformations configs
    trainset_configs = dict(
        path_file = PATH + r'/paths/trainpaths.txt',
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((250,250)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    )

    testset_configs = dict(
        path_file = PATH + r'/paths/testpaths.txt',
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),     
        ])
    )

    #Creating dataset
    train_dataset = GeorgeDataset(
        path_file = trainset_configs['path_file'],
        transform = trainset_configs['transform']
        )

    test_dataset = GeorgeDataset(
        path_file = testset_configs['path_file'],
        transform = testset_configs['transform']
    )

    #Creating dalaLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory = torch.cuda.is_available()
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory = torch.cuda.is_available()
    )

    #Creating model (Chosen resnet-18, and changed last layers to match num_classes)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, len(classes))
    model = model.to(device)

    #Configurations for fine-tuning
    loss_function = nn.CrossEntropyLoss()

    optimiser = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.layer1.parameters()},
        {'params': model.layer2.parameters()},
        {'params': model.layer3.parameters()},
        {'params': model.layer4.parameters()},
        {'params': model.fc.parameters(), 'lr':lr_finetune} #We use big lr for fc layer
        ],
    lr=lr, #Very small lr for other layers
    weight_decay=weight_decay
    )
    scheduler = lr_scheduler.StepLR(optimiser, step_size=step_size, gamma=gamma)

    dataloaders = dict(
        train = train_dataloader,
        test = test_dataloader
    )
    
    train_model(model,
                dataloaders,
                loss_function,
                optimiser,
                scheduler,
                num_epochs=num_epochs
                )

