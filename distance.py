import numpy as np


def get_distance_squared(pos1, pos2):
    """Returns distance squared.

    Parameters
    ----------
    pos1, pos2 : array_like
        Positions in 1 axis.

    Returns
    -------
    r2 : array_like
        Distance squared.
    """
    r2 = (pos1 - pos2)**2.
    return r2


def get_distance_squared_train(pos_train, is_1d=True):
    """Returns squared matrix of distance squared for training data.

    Parameters
    ----------
    pos_train : array_like
        Positions of training data.
    is_1d : bool, optional
        Are position in 1 axis or many.

    Returns
    -------
    r2 : array_like
        Distance squared matrix for training data.
    """
    if is_1d == True:
        pos1, pos2 = np.meshgrid(pos_train, pos_train, indexing='ij')
        r2 = get_distance_squared(pos1, pos2)
    else:
        for i in range(0, len(pos_train[0])):
            pos1, pos2 = np.meshgrid(pos_train[:, i], pos_train[:, i], indexing='ij')
            _r2 = get_distance_squared(pos1, pos2)
            if i == 0:
                r2 = _r2
            else:
                r2 += _r2
    return r2


def get_distance_squared_star(pos_train, pos_new, is_1d=True):
    """Returns vector of distance squared for new position.

    Parameters
    ----------
    pos_train : array_like
        Positions of training data.
    pos_new : float/array_like
        Position of new point.
    is_1d : bool, optional
        Are position in 1 axis or many.

    Returns
    -------
    r2 : array_like
        Distance squared matrix for training data.
    """
    if is_1d == True:
        r2 = get_distance_squared(pos_train, pos_new)
    else:
        for i in range(0, len(pos_new)):
            _r2 = get_distance_squared(pos_train[:, i], pos_new[i])
            if i == 0:
                r2 = _r2
            else:
                r2 += _r2
    return r2
