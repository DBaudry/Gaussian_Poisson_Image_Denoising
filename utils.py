from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import vistools  # image visualization toolbox
from skimage import io  # read and write images
if torch.cuda.is_available():
    loadmap = {'cuda:0': 'gpu'}
else:
    loadmap = {'cuda:0': 'cpu'}
from denoising_helpers import test_denoiser, PSNR
from models import DCTlike, DnCNN_pretrained_grayscale
from models import FFDNet, FFDNet_pretrained_grayscale
from vistools import unzip
from copy import copy
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
from scipy.optimize import minimize
import pickle
import seaborn
from scipy.stats import poisson
from scipy.stats import norm


# Noise Generation


def minmax(x):
    return max(0, min(255, x))


minmax = np.vectorize(minmax)


def mixture_noise(im, alpha, c, sig):
    """
    :param im: Input image
    :param alpha: Scaling parameter for Poisson noise
    :param c: Mean of the Gaussian noise
    :param sig: Standard deviation of the Gaussian noise
    :return: Image transformed with a Mixture of Gaussian and Poisson noise
    """
    poisson = alpha*np.random.poisson(lam=im)
    gaussian = np.random.normal(loc=c, scale=sig, size=im.shape)
    noisy_im = poisson+gaussian
    return noisy_im


def gaussian_noise(im, alpha, c, sig):
    """
    :param im: Input image
    :param sig: Standard Deviation of the noise
    :return:
    """
    n = np.random.normal(size=im.shape)
    return alpha * im + c + np.sqrt(alpha**2 * im + sig**2) * n


def plot_dist_noise(u, n_test, alpha, c, sig):
    """
    :param u: Real number in (1, 255) (luminence)
    :param n_test: number of samples
    :param sig: Standard deviation
    :return: Plot Comparison between Gaussian and Gaussian/Poisson noise distribution for a given luminence
    """
    plt.figure()
    test_noise = alpha * np.random.poisson(u, size=n_test) + c + sig * np.random.normal(size=n_test)
    test_noise_g = alpha * u + c + np.sqrt(alpha**2*u+sig**2) * np.random.normal(size=n_test)
    seaborn.distplot(test_noise)
    seaborn.distplot(test_noise_g)
    print('Expectation/Variance of Mixed Noise %f, %f' % (test_noise.mean(), test_noise.std()**2))
    print('Expectation/Variance of Gaussian Noise %f, %f' % (test_noise_g.mean(), test_noise_g.std()**2))


def load_generate_noise(name, alpha, c, sigma, display=False):
    """
    :param name: Name of the file
    :param alpha: alpha parameter of the mixture noise
    :param c: c parameter of the mixture noise
    :param sigma: sigma parameter of the mixture noise
    :param display: Bool, display the generated images along with the input or not
    :return: 3 images: input, GP mixture noise and gaussian noise
    """
    im_clean = io.imread(name, dtype='float32')
    if len(im_clean.shape) == 3:
        im_clean = im_clean[:, :, 0]
    im_noisy = mixture_noise(im_clean, alpha, c, sigma)
    im_noisy_gaussian = gaussian_noise(im_clean, alpha, c, sigma)
    if display:
        psnr_m, psnr_g = PSNR(im_noisy, im_clean), PSNR(im_noisy_gaussian, im_clean)
        captions_0 = ['Clean', 'Mixture - PSNR {}'.format(psnr_m), 'Gaussian - PSNR {}'.format(psnr_g)]
        vistools.display_gallery([im_clean, im_noisy, im_noisy_gaussian],captions_0)
    return im_clean, im_noisy, im_noisy_gaussian


# Tests with the DnCNN: raw input and transformed input


def test_raw_dncnn(alpha, c, sigma, im_list, display=False):
    """
    :param sigma: Standard deviation of the GP Mixture noise
    :param im_list: list of 3 images: input, GP mixture noise, Gaussian Noise
    :param display: Bool, Display or not the results
    :return: Output of DnCNN for the GP mixture and the Gaussian Noise
    """
    im = (copy(im_list[1]) - c)/alpha
    img = (copy(im_list[2]) - c)/alpha
    sig = np.sqrt(sigma**2/alpha**2 + im.mean())
    sig_g = np.sqrt(sigma**2/alpha**2 + img.mean())
    # sig = np.sqrt(sigma**2 + alpha**2 * im_list[1].mean())
    dncnn = DnCNN_pretrained_grayscale(sig)
    dncnn_g = DnCNN_pretrained_grayscale(sig_g)
    out = test_denoiser(dncnn, im, None, has_noise=True)[0]
    out_gaussian = test_denoiser(dncnn_g, img, None, has_noise=True)[0]
    if display:
        outputs = list()
        outputs.append((im_list[1], 'noisy input mixture with sigma = %f' % sigma))
        outputs.append((im_list[2], 'noisy input Gaussian with sigma = %f' % sigma))
        outputs.append((im_list[0], 'clean image'))
        outputs.append((out, 'dncnn output mixture- PSNR = %f' % PSNR(out, im_list[0])))
        outputs.append((out_gaussian, 'dncnn output gaussian - PSNR = %f' % PSNR(out_gaussian, im_list[0])))
        vistools.display_gallery(unzip(outputs, 0), unzip(outputs, 1))
    return out, out_gaussian


def denoise_transform(im, denoiser, transform, inv_transform, param_transform):
    """
    :param im: input image
    :param denoiser: A Denoiser that can be introduced in 'test_denoiser'
    :param transform: A transform function
    :param inv_transform: Inverse of the transform function
    :param param_transform: Parameters of the Transform
    :return: Denoised output
    """
    out = transform(im, **param_transform)
    param_inv = copy(param_transform)
    param_inv['param'] = (out[1], out[2])
    noise_lvl = 255/(out[2] - out[1])/2
    out = test_denoiser(denoiser(noise_lvl), out[0], None, has_noise=True)[0]
    return inv_transform(out, **param_inv)


def VST_Gaussian(im, alpha, c, sigma):
    """
    :param im: Input image
    :param sigma: Standard deviation of the Gaussian Noise
    :return: Anscombe transform of the input image, rescaled in (0, 255),
     and the corresponding standard deviation
    """
    im2 = (copy(im) - c)/alpha
    fu = np.sqrt(im2 + (sigma/alpha)**2)
    a = np.sqrt(1 + (sigma/alpha)**2)
    b = np.sqrt(255 + (sigma/alpha)**2)
    return (fu - a)/(b-a)*255., a, b


def inv_VST_Gaussian(im, alpha, c, sigma, param):
    """
    :param im: Output image of a VST
    :param sigma: Standard deviation of the Gaussian Noise
    :param std: Standard deviation of the noise of the transformed image
    :return: Inverse VST Transform
    """
    new_im = copy(im) * (param[1] - param[0])/255. + param[0]
    return new_im**2 - (sigma/alpha) ** 2


def test_VST(im, alpha, c, sigma):
    """
    :param im: Set of input images (clean, GP mixture Noise, Gaussian Noise)
    :param sigma: Standard deviation of the Noise
    :return: Display transformed images
    """
    clean = VST_Gaussian(im[0], alpha, c, sigma)[0]
    noisy_mixture, a, b = VST_Gaussian(im[1], alpha, c, sigma)
    noisy_gaussian = VST_Gaussian(im[2], alpha, c, sigma)[0]
    noise_m = np.abs(noisy_mixture-clean)*5
    noise_g = np.abs(noisy_gaussian-clean)*5
    vistools.display_gallery([im[0], noisy_mixture, noisy_gaussian])
    captions_1 = ['Transformed Mixed Noise', 'Transformed Gaussian Noise']
    vistools.display_gallery([noise_m, noise_g], captions_1)


def test_VST_bijective(im, alpha, c, sigma):
    """
    :param im: Input image
    :param sigma: Noise of the transform
    :return: Max absolute difference between the image and its Transformed/Inv_transformed counterpart
    to verify that the VST is bijective (this function is just a test function)
    """
    im2 = (copy(im)-c)/alpha
    test, a, b = VST_Gaussian(im, alpha, c, sigma)
    test2 = inv_VST_Gaussian(test, alpha, c, sigma, param=(a, b))
    return np.abs(test2-im2).max()


def test_dncnn_algos(name, alpha, c, sigma, display=True):
    im = load_generate_noise(name, alpha, c, sigma, display=False)
    dncnn_raw = test_raw_dncnn(alpha, c, sigma, im, display=False)
    dncnn = DnCNN_pretrained_grayscale
    param = {'alpha': alpha, 'c': c, 'sigma': sigma}
    out = denoise_transform(im[1], dncnn, VST_Gaussian, inv_VST_Gaussian, param)
    out_g = denoise_transform(im[2], dncnn, VST_Gaussian, inv_VST_Gaussian, param)
    outputs = im + dncnn_raw + (out, out_g)
    PSNR_im = []
    for image in outputs[1:]:
        PSNR_im.append(PSNR(image, im[0]))
    if display:
        captions = ['Clean Image', 'GP Mixture Noise - PSNR = %f' % PSNR_im[0],
                    'Gaussian Noise - PSNR = %f' % PSNR_im[1], 'GP Mixture Raw DnCNN - PSNR = %f' % PSNR_im[2],
                    'Gaussian Raw DnCNN - PSNR = %f' % PSNR_im[3], 'GP Mixture VST DnCNN - PSNR = %f' % PSNR_im[4],
                    'Gaussian VST DnCNN - PSNR = %f' % PSNR_im[5]]
        vistools.display_gallery(outputs, captions)
    return outputs


# ADMM


def logp_scipy(x, y, sigma, tol=1e-4):
    p = poisson(x)
    n = norm(y, sigma)
    max_range = np.sqrt(-2*sigma**2*np.log(tol))
    k_range = np.arange(min(299, max(int(y-max_range), 0)), min(int(y+max_range), 300))
    prob_list = np.zeros(k_range.shape[0])
    for i, k in enumerate(k_range):
        prob_list[i] = p.logpmf(k) + n.logpdf(k)
    a = prob_list.max()
    return a + np.log(np.exp(prob_list-a).sum())


def write_proba_grid(sigma):
    res = np.zeros((255, 255))
    for x in tqdm(range(1, 256)):
        for y in range(1, 256):
            res[x-1, y-1] = logp_scipy(x, y, sigma, tol=1e-4)
    name = 'pickle/GP_mixture_probability_sigma_'+str(sigma)+'.pk'
    pickle.dump(res, open(name, 'wb'))
    return res


def read_proba_grid(sigma, best_match=True):
    try:
        res = pickle.load(open('pickle/GP_mixture_probability_sigma_'+str(sigma)+'.pk', 'rb'))
        return res
    except:
        if not best_match:
            print('There was no pre-computed probability grid for this value of %f' %sigma)
            return write_proba_grid(sigma)
        else:
            available = np.array([15, 30, 45])
            sig_ind = np.where((available - sigma) > 0)[0]
            if sig_ind.shape[0] == 0:
                sig = available[-1]
            else:
                sig = available[sig_ind[0]]
                print('Downloaded model for sigma= %f instead of sigma=%f' % (sig, sigma))
            return pickle.load(open('pickle/GP_mixture_probability_sigma_'+str(sig)+'.pk', 'rb'))


def norm_L2(x):
    return np.sqrt((x**2).flatten().sum())


def compute_delta(x, v, u, xp, vp, up):
    return 1/np.sqrt(x.shape[0]*x.shape[1])*(norm_L2(x-xp) + norm_L2(v-vp) + norm_L2(u-up))


def update_x(im, prob, rho, u, v):
    """
    :param im: Noisy Image
    :param prob: grid of observation conditional likelihood
    :param rho: strength of regularization
    :param u: Lagrange multiplier
    :param v: Slack variable for x
    :return: New value of x maximizing regularized log-likelihood
    """
    new_x = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            y = im[i, j]
            index = max(1, min(255, int(round(y, 0))))
            obj = -prob[:, index-1] + rho/2 * (np.arange(1, 256) - v[i, j] + u[i, j])**2
            new_x[i, j] = obj.argmin()
    return new_x


def ADMM(im, alpha, c, sigma, denoiser, l, param, tol, max_iter, im_clean):
    """
    :param im: Input Image
    :param alpha: Underlying model scaling of poisson noise
    :param c: Underlying model mean of the gaussian noise
    :param sigma: Underlying model noise
    :param denoiser: Denoising algorithm class used in test_denoiser, is instanced with a certain sigma
    at each iteration
    :param l: Regularization parameter
    :param param: Parameters of the continuation scheme (rho, eta, gamma): Starting value, update criterion
    and update rate
    :param tol: tolerance for convergence
    :param max_iter: maximum number of loops before stopping the algorithm
    :return:
    """
    delta = tol + 0.01
    rho, eta, gamma = param
    n_iter = 0
    # x = np.random.uniform(0, 255, size=im.shape)
    im = (copy(im) - c)/alpha
    x = copy(im)
    prob_array = read_proba_grid(int(sigma/alpha), best_match=True)
    v, u = copy(x), np.zeros(im.shape)
    pbar = tqdm(total=max_iter)
    out_x, out_v = [], []
    PSNR_x, PSNR_v = [], []
    while delta > tol and n_iter < max_iter:
        n_iter += 1
        xp, vp, up, delta_p = copy(x), copy(v), copy(u), copy(delta)
        x = update_x(im, prob_array, rho, u, v)
        sig_denoiser = np.sqrt(l/rho)
        v = test_denoiser(denoiser(sig_denoiser), x+u, None, has_noise=True)[0]
        v = v[0, 0, :, :].numpy()
        u += x - v
        delta = compute_delta(x, v, u, xp, vp, up)
        if delta >= eta * delta_p:
            rho *= gamma
        pbar.update(1)
        out_x.append(x)
        out_v.append(v)
        if im_clean is not None:
            PSNR_x.append(PSNR(x, im_clean))
            PSNR_v.append(PSNR(v, im_clean))
    return (x+v)/2, out_x, out_v, PSNR_x, PSNR_v


if __name__ == '__main__':
    alpha0, c0, sigma0 = 0.8, 0.2*125, 20
    param = [5, 10, 20, 25, 40, 60]
    for p in param:
        write_proba_grid(p)