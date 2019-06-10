'''---------------------Imports------------------------'''
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import datasets
import models
from utils import AverageMeter, Logger
from center_loss import CenterLoss
from torch.utils.data import DataLoader
import transforms
import torchvision
import tqdm 
import gc
import torch.nn as nn
from torch.nn import functional as F
import math
from PIL import Image
import argparse


def main():

    '''---------------------Read parameters-------------------'''
    #blank
    model_savepath = './trained_models/'
    model_name = 'test_model'
    epoch = 39
    load_params = torch.load(model_savepath+model_name+str(epoch)+'.th')
    embed_size = 32
    num_classes = 4599
    '''---------------------Define Model ------------------------'''
    MODIFY_VGG = True
    model = torchvision.models.vgg19(pretrained=True)

    mean = [0.5,0.5,0.5]
    std = [0.5,0.5,0.5]
    resize = torchvision.transforms.Resize(224)

    aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((224,224)),
        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5, hue=0.05),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomAffine((-5,5), translate=None, scale=(0.8,1.2), shear=None, resample=False, fillcolor=0),
        ])

    tensor_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                      torchvision.transforms.Normalize(mean=mean,
                                                                       std=std)])

    train_transform = torchvision.transforms.Compose([resize,
        aug,
        tensor_transform])

    test_transform = torchvision.transforms.Compose([resize,
        torchvision.transforms.CenterCrop((224,224)),
        tensor_transform])

    if MODIFY_VGG:
        in_ =list(model.classifier[3].parameters())[0].shape[-1]
        new_fc = torch.nn.Linear(in_,embed_size)
        model.classifier[3] = new_fc
        out_ = num_classes
        new_out = torch.nn.Linear(embed_size,num_classes)

        model.classifier[-1] = new_out

    def fwd_hook(self,input,output):
       self.feat = output
       pass
    _ = model.classifier[-3].register_forward_hook(fwd_hook)
    _ = model.classifier[-1].register_forward_hook(fwd_hook)

    '''----------------load weights------------------------'''
    model.load_state_dict(load_params['model_sd'])


    impaths = sys.argv[1:3]
    '''----------------Get 2 images------------------------'''
    #impaths = ['lfw/Dalai_Lama/Dalai_Lama_0001.jpg','lfw/Dalai_Lama/Dalai_Lama_0002.jpg']
    ims = []
    import skimage.io
    for ip in impaths:
      ims.append(skimage.io.imread(ip))

    '''---------------------verify if images are same------------------------'''

    tens0 = test_transform(Image.fromarray(ims[0])).unsqueeze(0)
    tens1 = test_transform(Image.fromarray(ims[1])).unsqueeze(0) 

    batch = torch.cat((tens0,tens1),0)
    _ = model(batch)
    features = model.classifier[-3].feat
    feature_normed = features.div(
                torch.norm(features, p=2, dim=1, keepdim=True).expand_as(features))
    distances = torch.sum(torch.pow(feature_normed[0,:] - feature_normed[1,:], 2))

    print(distances)


if __name__ == '__main__':
    main()
