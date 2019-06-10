TEST_PIPELINE = True
# -*- coding: utf-8 -*-
"""running kaiyang zhou center loss with VGG19.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/126G4jt2zo1Qqi58ANW3gQHBAkYjSEkYL

todo:
- [ ] change model
- [ ] change image size
- [x] train test split
"""

if not 'COLLAB':
    #!pip install -I --no-cache-dir pillow
    pass
if 'local':
#     %cd /home/aniketsingh/code/dump
    pass

#!git clone https://github.com/KaiyangZhou/pytorch-center-loss.git
# %cd pytorch-center-loss

#!wget -nc http://vis-www.cs.umass.edu/lfw/lfw.tgz
#!tar -xzf lfw.tgz

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
import pickle
from skimage import io
from PIL import Image

'''
Getting information about the lfw folder
'''
#-------------------------------------------------------------------------------
''' get the folder structure '''
import os,collections
train_folder = 'lfw'
folder_structure = collections.OrderedDict({})
classes = [d for d in sorted(os.listdir(train_folder)) if os.path.isdir(os.path.join(train_folder,d)) and d not in ['.','..']]
# if test_mode:
# #     classes = [classes[c] for c in [2,5]]
#     classes = classes[:10]
print(len(classes))
folder_structure = collections.OrderedDict({c:[f for f in os.listdir(os.path.join(train_folder,c)) if not os.path.isdir(f)] for c in classes})
if False:print(folder_structure)
#-------------------------------------------------------------------------------
''' which classes have which files '''
class_to_idx = {k:[] for k in folder_structure.keys()}
filelist = []
i = 0
for ( c ,fls) in folder_structure.items():   
    
    for fi in fls:
        filelist.append('/'.join([c,fi]))
        class_to_idx[c].append(i)
        i+=1

# if True:print(class_to_idx,filelist)
#-------------------------------------------------------------------------------

''' --------------- Make the LFW Dataset Class --------------- '''

from torch.utils.data import DataLoader
import transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

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

class ClassBalancedSubsetSampler(torch.utils.data.Sampler):
    def __init__(self,classes,class_idx,class_to_image_idx,filelist,batch_size,weighted=True):
        n_classes_in_set = len(class_idx)
        class_idx = np.sort(class_idx)
        class_sample_counts = np.zeros((n_classes_in_set,))
        for cloc,cix in enumerate(class_idx):
            class_name = classes[cix]
            class_sample_counts[cloc] = len(class_to_idx[class_name]) + 1e-4
        if weighted:
            self.class_weights = 1./(class_sample_counts)
            self.class_weights = self.class_weights/self.class_weights.sum()
        else:
            self.class_weights = 1./n_classes_in_set * np.ones_like(class_sample_counts)
        self.classes = classes
        self.class_idx = class_idx
        self.class_to_image_idx = class_to_image_idx
        self.filelist = filelist
        self.batch_size = batch_size
        self.n_batches_per_epoch = int((sum(class_sample_counts)+self.batch_size-1)//self.batch_size)
        #import pdb;pdb.set_trace()
        pass

    def __iter__(self):
        while True:
            sampled_c = np.random.choice(self.class_idx, size=self.batch_size,replace = True, p= self.class_weights)
            #import pdb;pdb.set_trace()
            image_idx = [np.random.choice(self.class_to_image_idx[self.classes[c]]) for c in sampled_c]
            np.random.shuffle(image_idx)
            filenames = [self.filelist[iix]  for iix in image_idx]
            yield iter(image_idx)
        pass
    def __len__(self):
        return self.n_batches_per_epoch
class LFWDataset(torch.utils.data.Dataset):
    def __init__(self,rootdir,filelist,setnames,transform):
        super(LFWDataset,self).__init__()
        self.filelist = filelist
        self.rootdir = rootdir
        self.transform = transform
        self.setnames = setnames
    def __getitem__(self,idx):
        fname = os.path.join(self.rootdir,
                        self.filelist[idx])
        label = self.filelist[idx].split('/')[0]
        image = io.imread(fname)
        pil_image = Image.fromarray(image)
        tensor_image = self.transform(pil_image)
        return tensor_image,torch.tensor(self.setnames.index(label))
        pass
    def __len__(self):
        return len(self.filelist)
        pass

class LFWDataloaders(object):
    def __init__(self, batch_size, use_gpu, num_workers,train_transform,test_transform,splits = None):

        pin_memory = True if use_gpu else False

        

        #trainset = torchvision.datasets.ImageFolder(root='./lfw', transform=train_transform)
        #testset = torchvision.datasets.ImageFolder(root='./lfw', transform=test_transform)

    
        if splits is None:

            new_class_order = np.random.permutation(range(len(classes)))
            train_ratio = 0.8
            n_train_classes = int(len(classes)*train_ratio)
            train_idx = new_class_order[:n_train_classes]
            test_idx = new_class_order[n_train_classes:]
        else:
            train_idx = splits['train']
            test_idx = splits['test']
            n_train_classes = len(train_idx)
        if not 'try':
            print(classes[:10],
                        np.array(classes)[new_class_order[:10]])
        train_names = [classes[i] for i in train_idx]
        test_names = [classes[i] for i in test_idx]

        trainset = LFWDataset('./lfw',filelist,train_names,train_transform)# need custom Dataset to do balanced sampling
        testset = LFWDataset('./lfw',filelist,test_names,test_transform)


        # based on https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        #train_sampler = SubsetRandomSampler(train_idx)
        train_sampler = ClassBalancedSubsetSampler(classes,train_idx,class_to_idx,filelist,batch_size)
        test_sampler = ClassBalancedSubsetSampler(classes,test_idx,class_to_idx,filelist,batch_size,weighted=False)

        trainloader = torch.utils.data.DataLoader(
            trainset,  batch_sampler=train_sampler, 
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_sampler=test_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )


        self.trainloader = trainloader
        self.testloader = testloader
#         self.num_classes = 5749
        self.n_train_classes = n_train_classes
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.num_classes = len(classes)

'''---------------  Make the arguments for the run --------------- '''

parser = argparse.ArgumentParser("Center Loss Example")
# dataset
# parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'])
parser.add_argument('-d', '--dataset', type=str, default='mnist')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr-model', type=float, default=0.0001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='cnn')
# misc
parser.add_argument('--eval-freq', type=int, default=10)#10
parser.add_argument('--print-freq', type=int, default=25)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--seed', type=int, default=1)
# parser.add_argument('--use-cpu', action='store_false')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--plot', action='store_false', help="whether to plot features for every epoch")

#args = parser.parse_args(['--dataset','lfw','--gpu','0'])
args = parser.parse_args()
args.use_cpu = False
args.plot = True
args.embed_size = 32
args.use_normed = True
# args = parser.parse_args(['--plot','false'])

'''for saving'''
model_savepath = './trained_models/'
model_name = 'test_model'
if not os.path.isdir(model_savepath):
    os.makedirs(model_savepath)
save_every_n_epochs = 1#args.eval_freq

'''---------------  some prerequisites: reproducibility, gpu etc. --------------- '''

torch.manual_seed(args.seed)
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))

if use_gpu:
    print("Currently using GPU: {}".format(args.gpu))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
else:
    print("Currently using CPU")

'''---------------  Get the DataLoaders --------------- '''

print("Creating dataset: {}".format(args.dataset))

if args.dataset.lower() == 'mnist':
    dataset = datasets.create(
        name=args.dataset, batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers,)

#     dataset = MNIST(batch_size=args.batch_size, use_gpu=use_gpu,
#     num_workers=args.workers,)
    trainloader, testloader = dataset.trainloader, dataset.testloader
    num_classes = dataset.num_classes
elif args.dataset.lower() == 'lfw':
    #lfw_data = LFWDataloaders('lfw',filelist,transform = transform)
    #trainloader = torch.utils.data.DataLoader(dataset = lfw_data,batch_size=args.batch_size, 
    #num_workers=args.workers,)
    #testloader = trainloader
    #num_classes = 1680
    with open('lfw_splits','rb') as f:
      splits = pickle.load(f)
    dataset = LFWDataloaders(batch_size=args.batch_size, use_gpu=use_gpu,
    num_workers=args.workers,train_transform=train_transform,test_transform=test_transform,splits=splits)
    trainloader, testloader = dataset.trainloader, dataset.testloader
    num_classes = dataset.n_train_classes #dataset.num_classes

'''--------------- Make the model --------------- '''

#    

print("Creating model: {}".format(args.model))
if args.dataset.lower() == 'mnist':
    model = models.create(name=args.model, num_classes=dataset.num_classes)#ConvNet(num_classes=num_classes) 
else:
    
    model = torchvision.models.vgg19(pretrained=True)

    in_ =list(model.classifier[3].parameters())[0].shape[-1]
    new_fc = torch.nn.Linear(in_,args.embed_size)
    model.classifier[3] = new_fc

    out_ = num_classes
    new_out = torch.nn.Linear(args.embed_size,num_classes)
    
    model.classifier[-1] = new_out
    #--------------------------------------------
    '''
    def make_fwd_hook(store):
        def fwd_hook(self,input,output):
            gpu_ix = output.get_device()
            store[gpu_ix] = output
            pass
        return fwd_hook
    embeddings = [None]*torch.cuda.device_count()
    last_fcs = [None]*torch.cuda.device_count()
    _ = model.classifier[-3].register_forward_hook(make_fwd_hook(embeddings))
    _ = model.classifier[-1].register_forward_hook(make_fwd_hook(last_fcs))
    '''
    #----------------------------------------------
    
    def fwd_hook(self,input,output):
       self.feat = output
       pass
    _ = model.classifier[-3].register_forward_hook(fwd_hook)
    _ = model.classifier[-1].register_forward_hook(fwd_hook)
    
   #----------------------------------------------

if use_gpu:
    #model = nn.DataParallel(model)
    model = model.cuda()
    pass

'''--------------- plotting --------------- '''
def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

'''--------------- testing --------------- '''
def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            if args.dataset.lower() == 'mnist':
                features, outputs = model(data)
            else:
                _ = model(data)
                features = model.classifier[-3].feat 
                outputs = model.classifier[-1].feat

            feature_normed = features.div(
            torch.norm(features, p=2, dim=1, keepdim=True).expand_as(features))
 

            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
            
            if args.plot:
                if use_gpu:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features.data.numpy())
                    all_labels.append(labels.data.numpy())

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        #plot_features(all_features, all_labels, num_classes, epoch, prefix='test')

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

num_classes

'''--------------- Prepare the criterions --------------- '''

criterion_xent = nn.CrossEntropyLoss()
criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=args.embed_size, use_gpu=use_gpu)

'''--------------- Prepare the optimizers --------------- '''
optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
#optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

#optimizer_model = torch.optim.Adam(model.parameters())
optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

if args.stepsize > 0:
    scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

#outputs.shape,features.shape,labels.shape

'''--------------- training loop --------------- '''

start_time = time.time()

for epoch in tqdm.tqdm(range(args.max_epoch)):
    print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
    
    model.train()
    xent_losses = AverageMeter() 
    cent_losses = AverageMeter()
    losses = AverageMeter()
    
    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        #-----------------------------------------------------------------------
        # forward pass
        
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        if args.dataset.lower() == 'mnist':
            features, outputs = model(data)
        else:
            _ = model(data)
            features = model.classifier[-3].feat 
            outputs = model.classifier[-1].feat

        #-----------------------------------------------------------------------
        # losses
        
        #print(outputs.shape,features.shape,labels.shape)
        loss_xent = criterion_xent(outputs, labels)
        if args.use_normed:
            feature_normed = features.div(
            torch.norm(features, p=2, dim=1, keepdim=True).expand_as(features))

            loss_cent = criterion_cent(feature_normed, labels)
        else:
            loss_cent = criterion_cent(features, labels)

        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        #-----------------------------------------------------------------------
        # backward pass
        
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / args.weight_cent)
        optimizer_centloss.step()

        #-----------------------------------------------------------------------
        # bookkeeping
        
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

        #-----------------------------------------------------------------------
        # plotting
        
        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
                  .format(batch_idx+1,  losses.val, losses.avg, xent_losses.val, xent_losses.avg, cent_losses.val, cent_losses.avg))
        #-----------------------------------------------------------------------
        # garbage collection
        gc.collect()
        if batch_idx>len(trainloader):
            break
        if TEST_PIPELINE and batch_idx>2:
            break

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
#         plot_features(all_features, all_labels, num_classes, epoch, prefix='train')
   
    #-----------------------------------------------------------------------
    # scheduler step
    if args.stepsize > 0: scheduler.step()

    #-----------------------------------------------------------------------
    # testing phase        
    if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
        print("==> Test")
        acc, err = test(model, testloader, use_gpu, num_classes, epoch)
        print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
        #-----------------------------------------------------------------------
        # saving                
        save_params = { 
            'epoch': epoch,
            'model_sd': model.state_dict(),
            'optimizer_model_sd': optimizer_model.state_dict(),
            'scheduler_sd' : scheduler.state_dict(),
            'acc': acc,
            'err': err,
            }
        torch.save(save_params,model_savepath+model_name+str(epoch)+'.th')
        

#-----------------------------------------------------------------------
# elaspsed time        
elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

'''--------------- XXXXX  --------------- '''




