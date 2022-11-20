import torch 
import torchvision
from torchvision import models, transforms
from PIL import Image
import pandas as pd


import os
import numpy as np

import glob
import random
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = os.getcwd()
PATH_MODEL = PATH + r'/runs/train/resnet_finetuned.pt'
PATH_DATA = PATH + r'/inference/images/'
PATH_OUTPUT = PATH + r'/inference/output.csv'

if __name__ =='__main__':

    
    model = torch.load(PATH_MODEL)
    model.eval()
    classes = ['george', 'no_george']
    classes_to_idx = {
        'george' : 0,
        'no_george' : 1
    }

    transformations=transforms.Compose([
                 transforms.Resize(size=256),
                 transforms.CenterCrop(size=224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
    ])
    
    img_paths = []
    images_paths = []
    
    for path in  os.listdir(PATH_DATA):
        img_paths.append(os.path.join(PATH_DATA, path)) #for loading images
        images_paths.append(path)                       #for result.csv

    
    result_labels = []                                  #for result.csv
    with torch.no_grad():
        for num, img_path in enumerate(img_paths):
            img = Image.open(img_path).convert('RGB')
            inputs = transformations(img).unsqueeze(0).to(device)
            outputs = model(inputs)
            print(torch.softmax(outputs, dim=1))
            _, preds = torch.max(outputs, dim=1)
            result_labels.append(classes_to_idx[classes[preds]])

    result = pd.DataFrame()
    result['img'] = images_paths
    result['target'] = result_labels
    result.to_csv(PATH_OUTPUT, index=False)
    print('Success!\nResults saved at {}'.format(PATH_OUTPUT))
    