import annoy
import numpy as np
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
    random_seed:    int (default 1992)
                    The initial seed to be used by random function.

    Returns
    -------
    object:         self
                    AnnoyClassifier class with the parameters supplied.

    """

    def __init__(self, n_neighbors=5, mac_neighbors=None, metric='euclidean',
                 metric_params=None, n_trees=-1, random_seed=1992):

        if metric_params is None:
            metric_params = dict()
        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.metric_params = metric_params
        self.random_seed = random_seed

    def fit(self, X_train, y_train):
        r"""
        This is the fit function for NSW model.

        Parameters
        ----------
        X_train :   np.array
                    The train data to be fitted.
        y_train :   np.array
                    The true labels of X_train data.

        Returns
        -------
        object:     self
                    AnnoyClassifier class with train data fitted.

        """
        try:
            self.X_train = X_train.astype("float32")
        except:
            self.X_train = np.asarray(X_train, dtype="float32")

        self.N_feat = X_train.shape[1]
        self.N_train = X_train.shape[0]
        self.y_train = y_train
        self.t = annoy.AnnoyIndex(self.N_feat, metric=self.metric)
        self.t.set_seed(self.random_seed)
        for i, v in enumerate(X_train):
            self.t.add_item(i, v)
        self.t.build(self.n_trees)
        return self

    def predict(self, X_test):
        r"""
        This is the predict function for AnnoyClassifier model.

        Parameters
        ----------
        X_test :    np.array
                    The test data for the prediction.

        Returns
        -------
        y_hat :     np.array
                    The predicted labels of the test samples.

        """
        try:
            self.X_test = X_test.astype("float32")
        except:
            self.X_test = np.asarray(X_test, dtype="float32")

        if self.mac_neighbors is None:
            return self.predict_mac()
        return self.predict_macfac()

    def predict_mac(self):
        y_hat = np.empty(self.X_test.shape[0])
        for i, tv in enumerate(self.X_test):
            self.curr_nn_inds = self.t.get_nns_by_vector(tv, self.n_neighbors)
            nn_classes = [self.y_train[nn] for nn in self.curr_nn_inds]

            y_hat[i] = max(set(nn_classes), key=nn_classes.count)
        return y_hat

    def predict_macfac(self):
        y_hat = np.empty(self.X_test.shape[0])
        for i, tv in enumerate(self.X_test):
            self.curr_nn_inds = self.t.get_nns_by_vector(tv, self.mac_neighbors)
            self.nn_dtw(tv)
            self.curr_nn_inds = self.curr_nn_inds[:self.n_neighbors]
            nn_classes = [self.y_train[nn] for nn in self.curr_nn_inds]
            y_hat[i] = max(set(nn_classes), key=nn_classes.count)
        return y_hat

    def nn_dtw(self, tv):
        costs = np.empty(self.mac_neighbors)
        for i, nn in enumerate(self.curr_nn_inds):
            tr_v = self.X_train[nn]
            cost = dtw(tr_v, tv, **self.metric_params)
            costs[i] = cost
        sorted_cost_inds = np.argsort(costs)
        self.curr_nn_inds = np.asarray(self.curr_nn_inds)[sorted_cost_inds]
        self.curr_nn_inds = self.curr_nn_inds[:self.n_neighbors]
