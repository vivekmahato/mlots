import numpy as np
from sklearn.model_selection import train_test_split
import sys


def from_pandas_dataframe(d_frame=None, target=None,
                          test_size=None, shuffle=False, random_seed=1992,):
    r"""
    NAME: from_pandas_dataframe

    This is a utility provided in mlots package. The function helps representing a pandas dataframe into numpy arrays.

    Parameters
    ----------
    d_frame                             :       pandas.DataFrame (default None)
                                                The pandas DataFrame that needs to be transformed.
    target                              :       str or int (default None)
                                                The column name of the target (y) variable.
    test_size                           :       float or int (default None)
                                                The value is the percentage of the data to be split for the test set.
    shuffle                             :       bool (default False)
                                                If True, the data is shuffled randomly.
    random_seed                         :       int (default 1992)
                                                The initial seed to be used by random function.

    Returns
    -------
    X_train, X_test, y_train, y_test    :       ndarray, ndarray, ndarray, ndarray
                                                If target is None, only X is returned.
                                                If test_size is None, no train and test split is performed.

    """
    if target is not None:
        y = d_frame[target].values
        d_frame.drop([target], axis=1, inplace=True)
    else:
        y = None

    X = d_frame.to_numpy(dtype=np.float32)

    if test_size is None:
        if y is None:
            return X
        return X, y

    if isinstance(test_size, int):
        test_size = test_size/100
    if y is None:
        return train_test_split(X, test_size=test_size, random_state=random_seed, shuffle=shuffle)
    return train_test_split(X, y, test_size=test_size, random_state=random_seed, shuffle=shuffle)
