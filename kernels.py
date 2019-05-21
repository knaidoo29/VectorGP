import numpy as np


def gaussian_kernel(r2, sigma_rbf, l):
    """Gaussian kernel function.

    Parameters
    ----------
    r2 : array_like
        Distance.
    sigma_rbf : float
        Gaussian kernel amplitude.
    l : float
        Gaussian kernel scale length.

    Returns
    -------
    Returns the Gaussian kernel function.
    """
    return np.exp(-r2/(2*l**2.))*(sigma_rbf**2.)
