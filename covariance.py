import numpy as np
import kernels


def get_kappa(r2, kernel_dict, mode='Gaussian'):
    """Kernal function evaluated at for the squared distance r2.

    Parameters
    ----------
    r2 : array_like
        Distance squared.
    kernel_dict : dictionary
        Dictionary of kernel parameters, which depending on the mode you must
        define:
            - Gaussian : `sigma_rbf' and `l'.
    mode : str, optional
        Defines which kernel to use, `Gaussian' is the default.

    Returns
    -------
    kappa : array_like
        Kernel values.
    """
    if mode == 'Gaussian':
        kappa = kernels.gaussian_kernel(r2, kernel_dict['sigma_rbf'], kernel_dict['l'])
    else:
        print('mode = ' + str(mode) + ' is not supported.')
    return kappa


def get_k_train(r2_train, kernel_dict, sigma_n=None, mode='Gaussian'):
    """Gaussian process covariance for training data.

    Parameters
    ----------
    r2_train : array_like
        Training distance squared.
    kernel_dict : dictionary
        Dictionary of kernel parameters, which depending on the mode you must
        define:
            - Gaussian : `sigma_rbf' and `l'.
    mode : str, optional
        Defines which kernel to use, `Gaussian' is the default.
    sigma_n : float, optional
        The error of the training data.

    Returns
    -------
    k_train : array_like
        Gaussian process covariance for training data.
    """
    k_train = get_kappa(r2_train, kernel_dict, mode=mode)
    if sigma_n is not None:
        k_train += np.identity(len(r2_train))*sigma_n**2.
    return k_train


def get_k_star(r2_star, kernel_dict, mode='Gaussian'):
    """Gaussian process kernel vector.

    Parameters
    ----------
    r2_star : array_like
        Distance squared for kernels weighting between training data and new
        positions.
    kernel_dict : dictionary
        Dictionary of kernel parameters, which depending on the mode you must
        define:
            - Gaussian : `sigma_rbf' and `l'.
    mode : str, optional
        Defines which kernel to use, `Gaussian' is the default.

    Returns
    -------
    k_star : array_like
        Gaussian process vector for training data weighting to new positions.
    """
    k_star = get_kappa(r2_star, kernel_dict, mode=mode)
    return k_star


def get_k_starstar(kernel_dict, mode='Gaussian'):
    """Gaussian process kernel scalar for new positions.

    Parameters
    ----------
    kernel_dict : dictionary
        Dictionary of kernel parameters, which depending on the mode you must
        define:
            - Gaussian : `sigma_rbf' and `l'.
    mode : str, optional
        Defines which kernel to use, `Gaussian' is the default.

    Returns
    -------
    k_starstar : float
        Gaussian process kernel scalar for new positions.
    """
    k_starstar = get_kappa(0., kernel_dict, mode=mode)
    return k_starstar
