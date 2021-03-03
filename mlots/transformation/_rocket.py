import numpy as np
from numba import njit, prange


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
        X.astype(np.float32)
        self.kernels = _fit(X.shape[-1], num_kernels=self.num_kernels, random_seed=self.random_seed)
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
        X.astype(np.float32)
        return _transform(X, self.kernels)


@njit("Tuple((float64[:],int32[:],float64[:],int32[:],int32[:]))(int64,int64,int32)")
def _fit(input_length, num_kernels, random_seed): # pragma: no cover
    np.random.seed(random_seed)
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)

    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0

    for i in range(num_kernels):
        _length = lengths[i]

        _weights = np.random.normal(0, 1, _length)

        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1

    return weights, lengths, biases, dilations, paddings


@njit(fastmath=True)
def _apply_kernel(X, weights, length, bias, dilation, padding): # pragma: no cover
    input_length = len(X)

    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):

        _sum = bias

        index = i

        for j in range(length):

            if -1 < index < input_length:
                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[:],float64[:],int32[:],int32[:])))",
      parallel=True, fastmath=True)
def _transform(X, kernels): # pragma: no cover
    weights, lengths, biases, dilations, paddings = kernels

    num_examples, _ = X.shape
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float64)  # 2 features per kernel

    for i in prange(num_examples):

        a1 = 0  # for weights
        a2 = 0  # for features

        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + 2

            _X[i, a2:b2] = \
                _apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])

            a1 = b1
            a2 = b2

    return _X
