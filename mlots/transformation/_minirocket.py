import numpy as np
from importlib import import_module


class MINIROCKET:
    r"""
    NAME: MINIROCKET

    This is a class that represents MINIROCKET by Angus Dempster, Francois Petitjean, Geoff Webb.
    https://arxiv.org/abs/2012.08791 (preprint)

    Parameters
    ----------
    num_kernels :               int (default 10,000)
                                The number of random convolution kernels to be used.
    max_dilations_per_kernel:   int (default 32)
                                The maximum dilation allowed per kernel.
    ts_type:                       str (default "univariate")
                                The type of time-series data; either univariate or multivariate.
    random_seed:                int (default 1992)
                                The initial seed to be used by random function.

    Returns
    -------
    object:         self
                    MINIROCKET class with the parameters supplied.
    """
    def __init__(self, num_kernels=10_000, max_dilations_per_kernel=32, ts_type="univariate", random_seed=1992):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_seed = random_seed

        if ts_type == "univariate":
            self.f = import_module("mlots.transformation._rocket_algos._minirocket")
        elif ts_type == "multivariate":
            self.f = import_module("mlots.transformation._rocket_algos._minirocket_multivariate")
        else:
            raise ValueError("Incompatible type supplied!")

    def fit(self, X=None):
        r"""
        Parameters
        ----------
        X :             ndarray (default None)
                        The time-series data to be used to derive the kernels.
        Returns
        -------
        object:         self
                        MINIROCKET class with the the fitted kernels.
        """
        X = X.astype(np.float64)
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
            X = X[:, 0, :].astype(np.float64)

        self.parameters = self.f._fit(X, num_features=self.num_kernels,
                               max_dilations_per_kernel=self.max_dilations_per_kernel, random_seed=self.random_seed)
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
        if X.ndim == 2:
            X = X.reshape((X.shape[0], 1, X.shape[1]))
            X = X[:, 0, :].astype(np.float64)
        return self.f._transform(X, self.parameters)
