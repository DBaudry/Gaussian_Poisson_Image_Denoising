# Project for the Remote Sensing MVA course: Using Gaussian Denoisers for mixed Gaussian/Poisson Noise

For this project I implemented different methods to use Gaussian denoisers for luminence-dependant Gaussian/Poisson mixed noise.
The Gaussian denoiser I used is an implementation of the DnCNN available on the [IPOL platform](https://www.ipol.im/). The files denoising_dataloaders, denoising_helpers and training were implemented by CMLA researchers to ease the use of their denoisers, whereas the file vistools contains a lot of useful visualization functions.

My contribution is in the utils file, where one can find:

* An implementation of a Denoising algorithm using a Gaussian approximation of the mixed noise and a Variance Stabilizing (Anscombe) Transform to make the noise invariant with the intensity level
* An implementation of the iterative ADMM scheme, which solves the denoising problem as a Maximum A Posteriori problem and approximate a solution with the succession of: a maximum-likelihood step, a denoising step involving a Gaussian Denoiser, and a parameters' update step.

I made experiments in the notebook, and you can also find there further descriptions of the methods I used. A brief synthesis of the project is also available in the beamer.
