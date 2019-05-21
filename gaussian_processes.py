import numpy as np
from scipy.optimize import minimize

import distance
import kernels
import likelihood
import covariance
import utility


class GaussianProcesses:

    def __init__(self, mode='Gaussian'):
        # defines the modes
        self.mode = mode
        self.is_1d = None
        # training data
        self.pos_train = None
        self.f_train = None
        self.sigma_n = None
        # distances
        self.r2_train = None
        # kernel function
        self.kernel_dict = None
        self.k_train = None
        self.k_starstar = None
        self.invk = None
        # optimization values
        self.use_loo_cv = False
        self.num_params = 2
        self.kernel_opt_dict = None

    # Hidden Functions ------------------------------------------------------- #

    def _get_negative_likelihood(self, param):
        if self.mode == 'Gaussian':
            kernel_dict = {
                'sigma_rbf': param[0],
                'l': param[1]
            }
        if self.use_loo_cv == False:
            return -1.*likelihood.get_likelihood(self.r2_train, self.f_train,
                                                 kernel_dict, self.sigma_n,
                                                 mode=self.mode)
        else:
            return -1.*likelihood.get_likelihood_LOO_CV(self.r2_train, self.f_train,
                                                        kernel_dict, self.sigma_n,
                                                        mode=self.mode)

    def _predict(self, pos_new):
        r2_star = distance.get_distance_squared_star(self.pos_train, pos_new, is_1d=self.is_1d)
        k_star = covariance.get_k_star(r2_star, self.kernel_opt_dict, mode=self.mode)
        mu = k_star.T.dot(self.invk.dot(self.f_train))
        std = np.sqrt(self.k_starstar - k_star.T.dot(self.invk.dot(k_star)))
        return mu, std

    # Functions -------------------------------------------------------------- #

    def set_training(self, pos_train, f_train, is_1d=True):
        self.pos_train = pos_train
        self.f_train = f_train
        self.is_1d = is_1d
        self.r2_train = distance.get_distance_squared_train(self.pos_train, is_1d=self.is_1d)

    def optimize(self, sigma_n, use_LOO_CV=False):
        if self.mode == 'Gaussian':
            self.num_params = 2
        self.sigma_n = sigma_n
        if use_LOO_CV == True:
            self.use_loo_cv = True
        res = minimize(self._get_negative_likelihood, np.ones(self.num_params),
                       method='L-BFGS-B', options={'gtol': 1e-6, 'disp': True})
        if self.mode == 'Gaussian':
            self.kernel_opt_dict = {
                'sigma_rbf': res.x[0],
                'l': res.x[1]
            }
            print('Gaussian Parameters')
            print('-------------------')
            print('sigma_rbf = '+str(self.kernel_opt_dict['sigma_rbf']))
            print('l = ' + str(self.kernel_opt_dict['l']))
        self.k_train = covariance.get_k_train(self.r2_train, self.kernel_opt_dict,
                                              sigma_n=self.sigma_n, mode=self.mode)
        L = np.linalg.cholesky(self.k_train)
        invL = np.linalg.inv(L)
        self.invk = invL.T.dot(invL)
        self.k_starstar = covariance.get_k_starstar(self.kernel_opt_dict, mode=self.mode)

    def predict(self, pos_new):
        if np.isscalar(pos_new) == True:
            mu, std = self._predict(pos_new)
        else:
            mu, std = [], []
            for i in range(0, len(pos_new)):
                _mu, _std = self._predict(pos_new[i])
                mu.append(_mu)
                std.append(_std)
            mu, std = np.array(mu), np.array(std)
        return mu, std

    def clean(self):
        self.__init__()
