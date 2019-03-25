"""
training for CNN denoising

Copyright (C) 2018-2019, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
Copyright (C) 2018-2019, Pablo Arias <arias@cmla.ens-cachan.fr>
"""

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from PIL import Image

import torchvision.transforms as T

import numpy as np


### CODE FOR LISTING THE DIRECTORY CONTENT TOO BIG
IMG_EXTENSIONS = [ '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP' ]

def is_image_file(filename):
    return any(filename.upper().endswith(extension) for extension in IMG_EXTENSIONS)

'''
input : a folder path
return : a list containing the path of all the images in the folder
'''
def make_dataset(dir):
    import os
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


class ImageDenoisingDataset(Dataset):

    def __init__(self, root, sigma=0, Data_preprocessing=None, Randomnoiselevel=False, noise2noise=True):
        """
        Args:
            root (string): Root directory path.
            Data_preprocessing (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
            sigma: the level of noise we are going to add
            Randomnoiselevel: if True a random noise in [0,sigma]
            noise2noise: if True, the returned gt is also a noisy image 
        """
        imgs = make_dataset(root)
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.Randomnoiselevel=Randomnoiselevel
        self.sigma = sigma
        self.root = root
        self.imgs = imgs
        self.Data_preprocessing = Data_preprocessing
        self.noise2noise = noise2noise
        
    def add_noise(self, img, p):
        return img + torch.randn(img.size())*p
    
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, groundtruth) 
        """

        img_name = self.imgs[index]
        img = Image.open( open(img_name, 'rb') ).convert('L')
        #image = io.imread(img_name)
         
        if self.Data_preprocessing is not None:      
            img = self.Data_preprocessing(img)
        else:
            img = T.ToTensor()(img)

        if self.Randomnoiselevel is True :
            p = self.sigma/255*torch.rand(1)              
        else:
            p = self.sigma/255
        
        # add noise to groud truth if noise2noise
        gt = img.clone()
        if self.noise2noise:
            gt = self.add_noise(gt, p)

        # add noise to image
        img = self.add_noise(img, p)
                
        return img, gt


    def __len__(self):
        return len(self.imgs)


class ImageUniformNoiseDataset(Dataset):

    def __init__(self, root, sigma=0, Data_preprocessing=None, Randomnoiselevel=False, noise2noise=False):
        """
        Args:
            root (string): Root directory path.
            Data_preprocessing (callable, optional): A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
            sigma: the level of noise we are going to add
            Randomnoiselevel: if True a random noise in [0,sigma]
        """
        imgs = make_dataset(root)
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        if sigma > 1 or sigma < 0:
            raise(RuntimeError("Noise level %f ouside valid range [0,1]. " % (sigma)))
        
        self.Randomnoiselevel=Randomnoiselevel
        self.sigma = sigma
        self.root = root
        self.imgs = imgs
        self.Data_preprocessing = Data_preprocessing
        self.noise2noise = noise2noise

    def add_noise(self, img, p):
        mask = torch.bernoulli( 1 - p * torch.ones(img.size()))
        noise = torch.Tensor(img.size()).uniform_(0,1)
        return img * mask + noise * (1 - mask)

        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, groundtruth) 
        """

        img_name = self.imgs[index]
        img = Image.open( open(img_name, 'rb') ).convert('L')
        #image = io.imread(img_name)
         
        if self.Data_preprocessing is not None:
            img = self.Data_preprocessing(img)
        else:
            img = T.ToTensor()(img)

        if self.Randomnoiselevel is True:
            p = torch.rand(1)
        else:
            p = self.sigma

        # add noise to groud truth
        gt = img.clone()
        if self.noise2noise:
            gt = self.add_noise(gt, p)

        # add noise to image
        img = self.add_noise(img, p)

        return img, gt


    def __len__(self):
        return len(self.imgs)





        
def train_val_denoising_dataloaders(imagepath, noise_sigma=30, crop_size=40,
                                    train_batch_size=128, val_batch_size=32,
                                    validation_split_fraction=0.1,
                                    noise_type = 'gaussian',
                                    noise2noise = False):
    '''creates dataloaders '''
    # The torchvision.transforms package provides tools for preprocessing data
    # and for performing data augmentation; here we set up a transform to
    # preprocess the data by rancomly cropping the image and converting to tensor
    Data_preprocessing=T.Compose([ T.RandomCrop(crop_size), T.RandomHorizontalFlip(), 
                                   T.RandomVerticalFlip(), T.ToTensor() ])  

    # We set up a Dataset object; 
    # Datasets load training examples one at a time, so we wrap it in a DataLoader
    # which iterates through the Dataset and forms minibatches.
    if noise_type == 'gaussian':
        mydataset = ImageDenoisingDataset(imagepath, noise_sigma, Data_preprocessing, 
                                          Randomnoiselevel=False,
                                          noise2noise = noise2noise)
    elif noise_type == 'uniform-sp':
        mydataset = ImageUniformNoiseDataset(imagepath, noise_sigma, Data_preprocessing,
                                             Randomnoiselevel=False,
                                             noise2noise = noise2noise)
    else:
        raise(RuntimeError("Noise type %s unknown. " % (noise_type)))

    #trainloader = torch.utils.data.DataLoader(mydataset, shuffle=True, batch_size=128, num_workers=0)

    ## define our indices -- our dataset has 9 elements and we want a 8:4 split
    indices = list(range(len(mydataset)))
    split = int(validation_split_fraction*len(mydataset))

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    #print('training set = %d, val set = %d'%(len(train_idx), len(validation_idx)))

    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]

    ## define our samplers -- we use a SubsetRandomSampler because it will return
    ## a random subset of the split defined by the given indices without replaf
    train_sampler      = sampler.SubsetRandomSampler(train_idx)
    validation_sampler = sampler.SubsetRandomSampler(validation_idx)

    train_loader = DataLoader(mydataset, batch_size=train_batch_size, sampler=train_sampler)
    val_loader   = DataLoader(mydataset, batch_size=val_batch_size,   sampler=validation_sampler)
    return train_loader, val_loader
    
