
# Functions
## Covariance Functions
from .covariance import get_kappa
from .covariance import get_k_train
from .covariance import get_k_star
from .covariance import get_k_starstar

## Distance Functions
from .distance import get_distance_squared
from .distance import get_distance_squared_train
from .distance import get_distance_squared_star

# Kernel Functions
from .kernels import gaussian_kernel

# Likelihood Functions
from .likelihood import get_likelihood
from .likelihood import get_likelihood_LOO_CV

# Utility Functions
from .utility import check_matrix_invertible

# Classes
## Gaussian Processes class 1D
from .gaussian_processes import GaussianProcesses

## Gaussian Processes class 2D
from .vector_gaussian_processes import VectorGaussianProcesses
