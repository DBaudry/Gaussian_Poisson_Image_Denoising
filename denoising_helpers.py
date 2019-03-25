"""
testing helpers for CNN denoising

Copyright (C) 2018-2019, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

def PSNR(img1, img2, peak=255):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    import numpy as np

    x = ((np.array(img1).squeeze() - np.array(img2).squeeze()).flatten() )
    return (10*np.log10(peak**2 / np.mean(x**2)))


def test_denoiser(denoiser, img_in, sigma=30, show=False, has_noise=False, sigma_param=False, img_clean=None):
    '''
    Helper function to test a denoising network.

    Args:
        denoiser: denoising network
        img_in: input image (either clean or noisy)
        sigma: noise standard deviation
        has_noise: set it to True if img_in is a noisy image.
                   set it to False if img_in is the clean image. In this
                   case noise will be added to test the denoising.
        show: if True shows a gallery with the denoising result
        sigma_param: True if the denoiser requires sigma as input
        img_clean: if set then has_noise<-False and PSNR is computed

    Returns:
        img_denoised: denoised image
        img_noisy: noisy image
        psnr_out: psnr after denoising (only if has_noise == False)
        psnr_in: psnr before denoising (only if has_noise == False)
    '''
    import vistools
    import numpy as np
    import torch 

    # put the image in the range [0,1] and add noise
    if has_noise == False:
        img_clean = img_in.astype('float32') / 255.
        img_test = img_clean + np.random.normal(0, sigma/255.0, img_clean.shape)
    else:
        if img_clean is not None:
            has_noise = False
            img_clean = img_clean.astype('float32') / 255.
        img_test = img_in.astype('float32') / 255.

    # call the denoiser

    # torch data type
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        # run on GPU
        denoiser = denoiser.cuda()
        dtype = torch.cuda.FloatTensor

    # set denoising network in evaluation (inference) mode
    denoiser.eval()

    # apply denoising network
    with torch.no_grad(): # tell pytorch that we don't need gradients
        img = dtype(img_test[np.newaxis,np.newaxis,:,:]) # convert to tensor
        if sigma_param == True:
            # apply network; equivalent to out = denoiser(img, sigma)
            out = denoiser.forward(img, sigma/255.)
        else:
            # apply network; equivalent to out = denoiser(img)
            out = denoiser.forward(img)
        out = out.cpu() # move to CPU memory

    # compute psnr
    if has_noise == False:
        psnrIN, psnrOUT = PSNR(img_clean, img, 1), PSNR(img_clean, out, 1)
    else:
        psnrIN, psnrOUT = -1, -1

    # scale outputs to [0,255]
    out *= 255.
    img *= 255.

    # visualize as gallery
    if show:
        if has_noise == False:
            vistools.display_gallery([np.array(img_clean).clip(0,1)*255,
                                      np.array(img).clip(0,255),
                                      np.array(out).clip(0,255)],
                                     ['clean', 'noisy (%.2f dB)'%psnrIN,
                                      'denoised (%.2f dB)'%psnrOUT])
        else:
            vistools.display_gallery([np.array(img).clip(0,255),
                                      np.array(out).clip(0,255)],                                      
                                     ['noisy', 'denoised'])

    return out, img, psnrOUT, psnrIN
