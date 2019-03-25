"""
Parametrable DnCNN model (https://github.com/cszn/DnCNN.git)

Copyright (C) 2018-2019, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
Copyright (C) 2018-2019, Pablo Arias <arias@cmla.ens-cachan.fr>
Inspired on:
    https://github.com/SaoYan/DnCNN-PyTorch/
    https://github.com/Ourshanabi/Burst-denoising
"""


import torch
import torch.nn as nn


class CONV_BN_RELU(nn.Module):
    '''
    PyTorch Module grouping together a 2D CONV, BatchNorm and ReLU layers.
    This will simplify the definition of the DnCNN network.
    '''

    def __init__(self, in_channels=128, out_channels=128, kernel_size=7, 
                 stride=1, padding=3):
        '''
        Constructor

        Args:
            - in_channels: number of input channels from precedding layer
            - out_channels: number of output channels
            - kernel_size: size of conv. kernel
            - stride: stride of convolutions
            - padding: number of zero padding

        Return: initialized module
        '''
        super(__class__, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        '''
        Applies the layer forward to input x
        '''
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        return(out)



class DnCNN(nn.Module):
    '''
    PyTorch module for the DnCNN network.
    '''

    def __init__(self, in_channels=1, out_channels=1, num_layers=17, 
                 features=64, kernel_size=3, residual=True):
        '''
        Constructor for a DnCNN network.

        Args:
            - in_channels: input image channels (default 1)
            - out_channels: output image channels (default 1)
            - num_layers: number of layers (default 17)
            - num_features: number of hidden features (default 64)
            - kernel_size: size of conv. kernel (default 3)
            - residual: use residual learning (default True)

        Return: network with randomly initialized weights
        '''
        super(__class__, self).__init__()
        
        self.residual = residual
        
        # a list for the layers
        self.layers = []  
        
        # first layer 
        self.layers.append(CONV_BN_RELU(in_channels=in_channels,
                                        out_channels=features,
                                        kernel_size=kernel_size,
                                        stride=1, padding=kernel_size//2))
        # intermediate layers
        for _ in range(num_layers-2):
            self.layers.append(CONV_BN_RELU(in_channels=features,
                                            out_channels=features,
                                            kernel_size=kernel_size,
                                            stride=1, padding=kernel_size//2))
        # last layer 
        self.layers.append(nn.Conv2d(in_channels=features,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=1, padding=kernel_size//2))
        # chain the layers
        self.dncnn = nn.Sequential(*self.layers)

        
    def forward(self, x):
        ''' Forward operation of the network on input x.'''
        out = self.dncnn(x)
        
        if self.residual: # residual learning
            out = x - out 
        
        return(out)




def DnCNN_pretrained_grayscale(sigma=30, savefile=None, verbose=False):
    '''
    Loads the pretrained weights of DnCNN for grayscale images from 
    https://github.com/cszn/DnCNN.git

    Args:
        - sigma   : is the level of noise in range(10,76,5)
        - savefile: is the .pt file to save the model weights 
        - verbose : verbose output

    Returns:
        - DnCNN(1,1) model with 17 layers with the pretrained weights     
    '''
    
    # sigmas for which there is a pre-trained model
    pretrained_sigmas = list(range(10, 76, 5))
 
    # download the pretained weights
    import os
    import subprocess

    here = os.path.dirname(__file__)
    try:
        os.stat(here+'/DnCNN')
    except OSError:
        print('downloading pretrained models')
        subprocess.run(['git', 'clone',  'https://github.com/cszn/DnCNN.git'],cwd=here)


        
    # read the weights
    import numpy as np
    import hdf5storage
    import torch

    dtype = torch.FloatTensor

    # find closest pre-trained sigma
    pretrained_sigmas = np.array(pretrained_sigmas)
    closest_pt_sigma = pretrained_sigmas[np.argmin(np.abs(pretrained_sigmas - sigma))]
    if closest_pt_sigma != sigma:
        print("Warning: no pretrained DnCNN for sigma = %d. Using instead sigma = %d" 
              % (sigma, closest_pt_sigma))
        sigma = closest_pt_sigma

    m = DnCNN(1,1)
   

    ### CACHING SYSTEM
    cached_model_fname = here+'/DnCNN/cached_model_gray_sigma%02d.mat'%sigma
    try: 
        os.stat(cached_model_fname)
        if torch.cuda.is_available():
            loadmap = {'cuda:0': 'gpu'}
        else:
            loadmap = {'cuda:0': 'cpu'}
        m = torch.load(cached_model_fname, map_location=loadmap)
        return m
    except OSError:
        pass

        
    mat = hdf5storage.loadmat(here+'/DnCNN/model/specifics/sigma=%d.mat'%sigma)

    TRANSPOSE_PATTERN = [3, 2, 0, 1]

    # copy first 16 layers
    t=0
    for r in range(16):
        x = mat['net'][0][0][0][t]
        if verbose:
            print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)
            print(r, m.layers[r].conv.weight.shape, m.layers[r].conv.bias.shape)

        w = np.array(x[0][1][0,0])
        b = np.array(x[0][1][0,1]).squeeze()

        m.layers[r].conv.weight = torch.nn.Parameter( dtype( np.reshape(w.transpose(TRANSPOSE_PATTERN) , m.layers[r].conv.weight.shape  )  ) ) 
        m.layers[r].conv.bias   = torch.nn.Parameter( dtype( b ) )
        m.layers[r].bn.bias     = torch.nn.Parameter( m.layers[r].bn.bias    *0 )
        m.layers[r].bn.weight   = torch.nn.Parameter( m.layers[r].bn.weight  *0 +1)
        t+=2

    # copy last layer 
    r=16
    x = mat['net'][0][0][0][t]
    if verbose:
        print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)
        print(r, m.layers[r].weight.shape, m.layers[r].bias.shape)

    w = np.array(x[0][1][0,0])[:,:,:,np.newaxis]
    b = np.array(x[0][1][0,1])[:,0]
    
    m.layers[r].weight = torch.nn.Parameter( dtype( w.transpose(TRANSPOSE_PATTERN)  )  )  
    m.layers[r].bias   = torch.nn.Parameter( dtype( b ) )

    
    ### FILL CACHE 
    try: 
        os.stat(cached_model_fname)
    except OSError:
        torch.save(m, cached_model_fname)

    
    if savefile:
        torch.save(m, savefile)

    return m

    
