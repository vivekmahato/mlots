import annoy
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tslearn.metrics import dtw
import os


class AnnoyClassifier(BaseEstimator, ClassifierMixin):
    r"""
    NAME: AnnoyClassifier

    This is a class that represents Annoy model with MAC/FAC strategy.

    Parameters
    ----------
    n_neighbors :   int (default 5)
                    The n (or k) neighbors to consider for classification.
    mac_neighbors : int (default None)
                    Number of neighbors to consider for MAC stage.
                    If None, n_neighbors are used for classification directly.
                    If int; the classification is in two stages:
                            MAC stage: mac_neighbors are returned using 'metric'.
                            FAC stage: n_neighbors are used for classification using DTW.
    metric:         str (default "euclidean")
                    The distance metric to be employed for Annoy.
                    Check annoy library for allowed metrics.
    metric_params:  dict() (default None)
                    The parameters of the metric being employed.
                    Example: For metric = "dtw", the metric_params can be:
                            { "global_restraint" : "sakoe_chiba",
                              "sakoe_chiba_radius": 1  }
                    See tslearn.metrics for more details.
    n_trees:        int (default -1)
                    The number of RPTrees to create for Annoy.
                    If n_trees=-1, it creates as many RPTs as possible.
    n_jobs:         int (default -1)
                    The number of CPU threads to use to build Annoy. -1 to use all the available threads.
    random_seed:    int (default 1992)
                    The initial seed to be used by random function.

    Returns
    -------
    object:         self
                    AnnoyClassifier class with the parameters supplied.

    """

    def __init__(self, n_neighbors=5, mac_neighbors=None, metric='euclidean',
                 metric_params=None, n_trees=-1, n_jobs=-1, random_seed=1992):

        if metric_params is None:
            metric_params = dict()
        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.metric_params = metric_params
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        self.n_jobs = n_jobs
        self.random_seed = random_seed

    def fit(self, X_train, y_train):
        r"""
        This is the fit function for NSW model.

        Parameters
        ----------
        X_train :   ndarray
                    The train data to be fitted.
        y_train :   array
                    The true labels of X_train data.

        Returns
        -------
        object:     self
                    AnnoyClassifier class with train data fitted.

        """
        self.X_train = X_train.astype("float32")

        self.N_feat = X_train.shape[1]
        self.N_train = X_train.shape[0]
        self.y_train = np.asarray(y_train)
        self.t = annoy.AnnoyIndex(self.N_feat, metric=self.metric)
        self.t.set_seed(self.random_seed)
        for i, v in enumerate(X_train):
            self.t.add_item(i, v)
        self.t.build(self.n_trees, n_jobs=self.n_jobs)
        return self

    def predict(self, X_test):
        r"""
        This is the predict function for AnnoyClassifier model.

        Parameters
        ----------
        X_test :    ndarray
                    The test data for the prediction.

        Returns
        -------
        y_hat :     array
                    The predicted labels of the test samples.

        """
        self.X_test = X_test.astype("float32")

        if self.mac_neighbors is None:
            return self._predict_mac()
        return self._predict_macfac()

    def _predict_mac(self):
        y_hat = np.empty(self.X_test.shape[0])
        for i, tv in enumerate(self.X_test):
            self.curr_nn_inds = self.t.get_nns_by_vector(tv, self.n_neighbors)
            nn_classes = [self.y_train[nn] for nn in self.curr_nn_inds]

            y_hat[i] = max(set(nn_classes), key=nn_classes.count)
        return y_hat

    def _predict_macfac(self):
        y_hat = np.empty(self.X_test.shape[0])
        for i, tv in enumerate(self.X_test):
            self.curr_nn_inds = self.t.get_nns_by_vector(tv, self.mac_neighbors)
            self._nn_dtw(tv)
            self.curr_nn_inds = self.curr_nn_inds[:self.n_neighbors]
            nn_classes = [self.y_train[nn] for nn in self.curr_nn_inds]
            y_hat[i] = max(set(nn_classes), key=nn_classes.count)
        return y_hat

    def _nn_dtw(self, tv):
        costs = np.empty(self.mac_neighbors)
        for i, nn in enumerate(self.curr_nn_inds):
            tr_v = self.X_train[nn]
            cost = dtw(tr_v, tv, **self.metric_params)
            costs[i] = cost
        sorted_cost_inds = np.argsort(costs)
        self.curr_nn_inds = np.asarray(self.curr_nn_inds)[sorted_cost_inds]
        self.curr_nn_inds = self.curr_nn_inds[:self.n_neighbors]
