import numpy as np
from importlib import import_module


class ROCKET:
    r"""
    NAME: ROCKET

    This is a class that represents ROCKET by Angus Dempster, Francois Petitjean, Geoff Webb.
    https://arxiv.org/abs/1910.13051 (preprint)

    Parameters
    ----------
    num_kernels :   int (default 10,000)
                    The number of random convolution kernels to be used.
    random_seed:    int (default 1992)
                    The initial seed to be used by random function.

    Returns
    -------
    object:         self
                    ROCKET class with the parameters supplied.
    """

    def __init__(self, num_kernels=10_000, random_seed=1992):
        self.num_kernels = num_kernels
        self.random_seed = random_seed
        self.f = import_module("mlots.transformation._rocket_algos._rocket")

    def fit(self, X=None):
        r"""
        Parameters
        ----------
        X :             ndarray (default None)
                        The time-series data to be used to derive the kernels.
        Returns
        -------
        object:         self
                        ROCKET class with the the fitted kernels.
        """
        X = X.astype(np.float64)
        self.kernels = self.f._fit(X.shape[-1], num_kernels=self.num_kernels, random_seed=self.random_seed)
        return self

    def transform(self, X=None):
        r"""
        Parameters
        ----------
        X :             ndarray (default None)
                        The time-series data to be transformed.
        Returns
        -------
        X :             ndarray
                        The time-series data after transformation.
        """
        X = X.astype(np.float64)
        return self.f._transform(X, self.kernels)
