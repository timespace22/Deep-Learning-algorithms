import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
class dataset(torch.utils.data.Dataset):
    def __init__(self, path='./dataset/trainset.h5'):
        f = h5py.File(path, 'r')
        self.data = f['data']
        self.label = f['label']
        
        self.length = self.data.shape[0]
    
    def __getitem__(self, index):
        label = self.label[index]
        new_label = np.zeros(2)
        new_label[label] = 1
        return self.data[index], new_label
    
    def __len__(self):
        return self.length
    
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def init_parameters(params: dict):
    w, b = params['w'], params['b']

    # Function to initialize the weight matrix and the bias term
    nn.init.kaiming_uniform_(params['w'], a=math.sqrt(5))
    if params['b'] is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(params['w'])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(params['b'], -bound, bound)

def init_params(params):
    for key in list(params.keys()):
        init_parameters(params[key])
        
def delta_cross_entropy(X, y):
    """
    Arguments:
    
    X:  the output from fully connected layer (batch_size, num_classes)
    y:  labels (batch_size, 1)
    """
    m = y.shape[0]
    grad = F.softmax(X, dim=1)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad