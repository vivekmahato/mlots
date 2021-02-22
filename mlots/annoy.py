import annoy
import numpy as np
from multiprocessing.pool import ThreadPool
from sklearn.base import BaseEstimator, ClassifierMixin
from tslearn.metrics import dtw


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
                    The number of CPU threads to use. -1 to use all the available threads.
    random_seed:    int (default 1992)
                    The initial seed to be used by random function.

    Returns
    -------
    object:         self
                    AnnoyClassifier class with the parameters supplied.

    """

    def __init__(self, n_neighbors=5, mac_neighbors=None, metric='euclidean',
                 metric_params=None, n_trees=-1, n_jobs=-1, random_seed=1992, ):

        if metric_params is None:
            metric_params = dict()
        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.metric_params = metric_params
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
        self.t.build(self.n_trees, self.n_jobs)
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
            return self.predict_mac()
        return self.predict_macfac()

    def predict_mac(self):
        try:
            pool = ThreadPool(processes=self.n_jobs)
        except ValueError:
            pool = ThreadPool(processes=None)

        def query_f(tv):
            nn = self.t.get_nns_by_vector(tv, self.n_neighbors)
            nn_classes = self.y_train[nn]
            return np.bincount(nn_classes).argmax()

        y_hat = pool.map(query_f, self.X_test)
        pool.close()
        return np.asarray(y_hat)

    def predict_macfac(self):
        try:
            pool = ThreadPool(processes=self.n_jobs)
        except ValueError:
            pool = ThreadPool(processes=None)

        def query_f(tv):
            nns = self.t.get_nns_by_vector(tv, self.mac_neighbors)
            try:
                pool2 = ThreadPool(processes=self.n_jobs)
            except ValueError:
                pool2 = ThreadPool(processes=None)

            def dtw_dist(nn):
                tr_v = self.X_train[nn]
                return dtw(tr_v, tv, **self.metric_params)

            costs = pool2.map(dtw_dist, nns)
            pool2.close()
            sorted_cost_inds = np.argsort(costs)
            nns = np.asarray(nns)[sorted_cost_inds]
            nns = nns[:self.n_neighbors]
            nn_classes = [self.y_train[nn] for nn in nns]
            return max(set(nn_classes), key=nn_classes.count)

        y_hat = pool.map(query_f, self.X_test)
        pool.close()
        return np.asarray(y_hat)
