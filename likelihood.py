import numpy as np

import covariance
import utility


def get_likelihood(r2_train, f_train, kernel_dict, sigma_n, mode='Gaussian'):
    """Likelihood for training data.

    Parameters
    ----------
    r2_train : array_like
        Squared distance matrix for training data.
    f_train : array_like
        Training data values.
    kernel_dict : dictionary
        Dictionary of kernel parameters, which depending on the mode you must
        define:
            - Gaussian : `sigma_rbf' and `l'.
    sigma_n : float, optional
        The error of the training data.
    mode : str, optional
        Defines which kernel to use, `Gaussian' is the default.

    Returns
    -------
    log_likelihood : float
        log likelihood for training data.
    """
    k_train = covariance.get_k_train(r2_train, kernel_dict, sigma_n=sigma_n, mode=mode)
    if mode == 'Gaussian':
        if kernel_dict['sigma_rbf'] > 0. and kernel_dict['l'] > 0.:
            L = np.linalg.cholesky(k_train)
            if utility.check_matrix_invertible(L) == True:
                invL = np.linalg.inv(L)
                invk = invL.T.dot(invL)
                sign, logdet = np.linalg.slogdet(k_train)
                log_likelihood = -0.5*(f_train.dot(invk.dot(f_train.T))) - 0.5*logdet
                if np.isfinite(log_likelihood) == True and sign == 1.:
                    return log_likelihood
                else:
                    return -np.inf
            else:
                return -np.inf
        else:
            return -np.inf


def get_likelihood_LOO_CV(r2_train, f_train, kernel_dict, sigma_n, mode='Gaussian'):
    """Likelihood for the leave-one-out cross-validation for training data.

    Parameters
    ----------
    r2_train : array_like
        Squared distance matrix for training data.
    f_train : array_like
        Training data values.
    kernel_dict : dictionary
        Dictionary of kernel parameters, which depending on the mode you must
        define:
            - Gaussian : `sigma_rbf' and `l'.
    sigma_n : float, optional
        The error of the training data.
    mode : str, optional
        Defines which kernel to use, `Gaussian' is the default.

    Returns
    -------
    log_likelihood : float
        log likelihood for training data.
    """
    k_train = covariance.get_k_train(r2_train, kernel_dict, sigma_n=sigma_n, mode=mode)
    if mode == 'Gaussian':
        if kernel_dict['sigma_rbf'] > 0. and kernel_dict['l'] > 0.:
            L = np.linalg.cholesky(k_train)
            sign, logdet = np.linalg.slogdet(k_train)
            if utility.check_matrix_invertible(L) == True and sign == 1.:
                invL = np.linalg.inv(L)
                invk = invL.T.dot(invL)
                invk_f = invk.dot(f_train)
                for i in range(0, len(f_train)):
                    mu_i = f_train[i] - invk_f[i]/invk[i,i]
                    sigma_i = np.sqrt(1./invk[i,i])
                    sigma_i += sigma_n
                    if i == 0:
                        log_likelihood = -0.5*np.log(sigma_i**2.) - ((f_train[i] - mu_i)**2.)/(sigma_i**2.)
                    else:
                        log_likelihood += -0.5*np.log(sigma_i**2.) - ((f_train[i] - mu_i)**2.)/(sigma_i**2.)
                if np.isfinite(log_likelihood) == True:
                    return log_likelihood
                else:
                    return -np.inf
            else:
                return -np.inf
        else:
            return -np.inf
