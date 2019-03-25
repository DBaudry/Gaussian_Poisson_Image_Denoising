"""
DCT-like CNN models

Copyright (C) 2018-2019, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

import torch
import torch.nn as nn

      

class DCTlike(nn.Module):
    """
    CNN with an architecture inspired by DCT denosing. It has
    two convolutional layers: the first one with s^2, s x s x 1
    filters, followed by an activation function and the output
    layer with 1 s x s x s^2 filters.
    """

    def __init__(self, ksize=7, sigma=30, initializeDCT=True, shrinkage='hard'):
        super(__class__, self).__init__()
        """
        Args:
            - ksize: patch size for the DCT
            - sigma: noise level (multiplies the threshold)
            - initializeDCT: if True, initializes the convolutional
                layers as the DCT and iDCT transforms; if false it
                uses a random initialization.
            - shrinkage: type of shrinkage used (hard thresholding, 
                soft shrinkage or tanh shrinkage)
        Returns:
            - model: initialized model
        """
        from scipy.fftpack import dct, idct
        import numpy as np

        dtype = torch.FloatTensor
        if torch.cuda.is_available():  dtype = torch.cuda.FloatTensor

        self.sigma = sigma
        self.dct = initializeDCT

        ch = ksize**2

        # pad by reflection: to have the output with the same size
        # as the input we pad the image boundaries. Usually, zero
        # padding is used for CNNs. However, since we want to
        # reproduce the DCT denoising, we use reflection padding.
        # Reflection padding is a differentiable layer.
        self.padding = nn.ReflectionPad2d(2*ksize//2-1)

        # first convolutional layer (e.g. DCT transform)
        self.conv_in = nn.Conv2d(in_channels=1, out_channels=ch,
                                 kernel_size=ksize, stride=1,
                                 padding=0, bias=not initializeDCT)

        # threshold parameter (one variable per frequency)
        self.thr = nn.Parameter(dtype(np.ones((1,ch,1,1))),
                                requires_grad=True)

        # shrinkage function
        if   shrinkage == 'hard': self.shrinkage = nn.Hardshrink(1.)
        elif shrinkage == 'soft': self.shrinkage = nn.Softshrink(1.)
        elif shrinkage == 'tanh': self.shrinkage = nn.Tanhshrink()
        else: print('DCTlike: unknown shrinkage option %s' % (shrinkage))

        # output conv layer (e.g. inverse DCT transform)
        self.conv_out = nn.Conv2d(in_channels=ch, out_channels=1,
                                  kernel_size=ksize, stride=1,
                                  padding=0, bias=not initializeDCT)

        # initialize the isometric DCT transforms
        if initializeDCT:

            # thresholding parameters (one per feature)
            factor = 3.0 if shrinkage == 'hard' else 1.5
            thr = np.ones((1,ch,1,1)) * sigma / 255. * factor
            thr[0,0] = 1e-3 # don't threshold DC component
            self.thr.data = nn.Parameter(dtype(thr), requires_grad=True)

            for i in range(ch):
                # compute dct coefficients using scipy.fftpack
                a=np.zeros((ksize,ksize)); a.flat[i] = 1

                # first layer with direct dct transform
                a1 = dct(dct(a.T,norm='ortho', type=3).T,norm='ortho', type=3)

                self.conv_in.weight.data[i,0,:,:] = nn.Parameter(dtype(a1));

                # second layer, inverse transform rotated pi degrees
                a2 = idct(idct(a.T,norm='ortho', type=2).T,norm='ortho', type=2)
                a2 = np.flip(np.flip(a2, axis=0), axis=1) # pi-rotation

                self.conv_out.weight.data[0,i,:,:] = 1/(ch)*nn.Parameter(dtype(a2.copy()))
        
        # random initialization
        else:
            # this comes from:
            # 1) that the image data follows N(1/2, 1/4) (so 0.5 +- 2*sigma = [0,1])
            # 2) imposing the output variance to be 0.5 (the default threshold for
            #    hardshrink)
            std = 2./np.sqrt(5.)/ksize
            for i in range(ch):
                self.conv_in .weight.data[i,0,:,:] = dtype(std*np.random.randn(ksize, ksize))
                self.conv_out.weight.data[0,i,:,:] = dtype(std*np.random.randn(ksize, ksize))

    def forward(self, x):

        # first convolutional layer
        out = self.conv_in(self.padding(x))
        
        # shrinkage non-linearity
        if self.dct:
            # we use the threshold weights only when using the DCT
            out = self.shrinkage(out / self.thr) * self.thr
        else:
            out = self.shrinkage(out)

        # final convolutional layer
        out = self.conv_out(out)

        return(out)
    
    
    