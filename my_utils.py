import os
from torch.utils.data import Dataset
import cv2
import numpy as np


class GeorgeDataset(Dataset):
    def __init__(self, path_file, transform=None):
        self.transform = transform
        with open(path_file, 'r') as f:
            self.paths = f.readlines()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        label, img_path = self.paths[idx].strip().split(',')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        return img, int(label)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def generate_paths(classes, train_path, test_path):
   
    PATH = os.getcwd()
    classes = classes

    if os.path.exists(PATH + r'/paths/trainpaths.txt'):
        os.remove(PATH + r'/paths/trainpaths.txt')

    root_dir = train_path
    for folder in os.listdir(root_dir):
        for filename in os.listdir(os.path.join(root_dir, folder)):
            path = os.path.join(root_dir, folder, filename)
            label = classes.index(folder)
            with open(PATH + r'/paths/trainpaths.txt','a') as f:
                f.writelines(str(label) + ',' + path +'\n')

    
    if os.path.exists(PATH + r'/paths/testpaths.txt'):
        os.remove(PATH + r'/paths/testpaths.txt')
    
    root_dir = test_path
    for folder in os.listdir(root_dir):
        for filename in os.listdir(os.path.join(root_dir, folder)):
            path = os.path.join(root_dir, folder, filename)
            label = classes.index(folder)
            with open(PATH + r'/paths/testpaths.txt','a') as f:
                f.writelines(str(label) + ',' + path +'\n')
    return
