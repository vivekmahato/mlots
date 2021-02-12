import numpy as np
from sklearn.model_selection import train_test_split
import sys


def from_pandas_dataframe(d_frame=None, target=None,
                          test_size=None, random_seed=1992, shuffle=False):

    if target is not None:
        try:
            y = d_frame[target].values
            d_frame.drop([target], axis=1, inplace=True)
        except KeyError as e:
            sys.exit(str(e))
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
