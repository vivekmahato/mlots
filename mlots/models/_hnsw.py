import hnswlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tslearn.metrics import dtw


class HNSWClassifier(BaseEstimator, ClassifierMixin):
    r"""
    NAME: HNSWClassifier

    This is a class that represents HNSW model from hnswlib combined with MAC/FAC strategy.

    Parameters
    ----------
    n_neighbors     :   int (default 1)
                        The n (or k) neighbors to consider for classification.
    mac_neighbors   :   int (default None)
                        Number of neighbors to consider for MAC stage.
                        If None, n_neighbors are used for classification directly.
                        If int; the classification is in two stages:
                            MAC stage: mac_neighbors are returned using HNSW with supplied 'space'.
                            FAC stage: n_neighbors are used for classification using DTW.
    space           :   str (default "l2")
                        The distance metric to be employed for HNSW.
                        Check hnswlib library for allowed metrics.
    max_elements    :   int (default 10)
                        The maximum number of elements that can be stored in the structure.
    M               :   int (default 5)
                        The maximum number of outgoing connections in the graph.
    ef_construction :   int (default 100)
                        Controls the tradeoff between construction time and accuracy.
                        Bigger ef_construction leads to longer construction, but better index quality.
    ef_Search       :   int (default 50)
                        The size of the dynamic list for the nearest neighbors in HNSW.
                        Higher ef leads to more accurate but slower search.
                        The value ef of can be anything between k and the size of the dataset.
                        if mac_neighbors = None; k = n_neighbors
                        if mac_neighbors = int;  k = mac_neighbors
    metric_params   :   dict() (default None)
                        The parameters of the metric being employed.
                        Example: For metric = "dtw", the metric_params can be:
                            { "global_restraint" : "sakoe_chiba",
                              "sakoe_chiba_radius": 1  }
                        See tslearn.metrics for more details.
    n_jobs          :   int (default -1)
                        The number of CPU threads to use. -1 to use all the available threads.
    random_seed     :   int (default 1992)
                        The initial seed to be used by random function.

    Returns
    -------
    object          :   self
                        HNSWClassifier class with the parameters supplied.

    """

    def __init__(self,
                 n_neighbors=1,
                 mac_neighbors=None,
                 space="l2",
                 max_elements=10,
                 M=5,
                 ef_construction=100,
                 ef_Search=50,
                 metric_params=None,
                 random_seed=1992,
                 n_jobs=-1):

        if metric_params is None:
            metric_params = {}
        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.space = space
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef_Search = ef_Search
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.random_seed = random_seed

    def fit(self, X_train=None, y_train=None):
        r"""
        This is the fit function for HNSWClassifier model.

        Parameters
        ----------
        X_train :   ndarray
                    The train data to be fitted.
        y_train :   array
                    The true labels of X_train data.

        Returns
        -------
        object  :   self
                    HNSWClassifier class with train data fitted.

        """
        self.X_train = X_train.astype("float32")
        self.dimension = self.X_train.shape[1]
        self.num_elements = self.X_train.shape[0]
        self.y_train = np.asarray(y_train)

        self.model = hnswlib.Index(space=self.space, dim=self.dimension)
        self.model.init_index(max_elements=self.num_elements,
                              ef_construction=self.ef_construction, M=self.M)
        self.model.set_ef(self.ef_Search)
        self.model.set_num_threads(self.n_jobs)
        self.model.add_items(self.X_train, np.arange(self.num_elements))
        return self

    def predict(self, X_test):
        r"""
        This is the predict function for HNSWClassifier model.

        Parameters
        ----------
        X_test :    ndarray
                    The test data for the prediction.

        Returns
        -------
        y_hat :     array
                    The predicted labels of the test samples.

        """
        np.random.seed(self.random_seed)
        self.X_test = X_test.astype("float32")

        if self.mac_neighbors is None:
            return self._predict_mac()
        return self._predict_macfac()

    def _predict_mac(self):
        self.nbrs_all_query, _ = self.model.knn_query(self.X_test,
                                                      k=self.n_neighbors)
        y_hat = np.empty(self.X_test.shape[0])
        for i, nbrs in enumerate(self.nbrs_all_query):
            labels = list(self.y_train[nbrs])
            y_hat[i] = max(set(labels), key=labels.count)
        return y_hat

    def _predict_macfac(self):
        self.nbrs_all_query, _ = self.model.knn_query(self.X_test,
                                                      k=self.mac_neighbors)
        self._nn_dtw()
        y_hat = np.empty(self.X_test.shape[0])
        for i, nbrs in enumerate(self.nbrs_all_query):
            nn_classes = [self.y_train[nn] for nn in nbrs]
            y_hat[i] = max(set(nn_classes), key=nn_classes.count)
        return y_hat

    def _nn_dtw(self):
        dtw_nbrs_all_query = []
        for te_idx, nbrs in enumerate(self.nbrs_all_query):
            costs = np.empty(len(nbrs))
            for i, nn in enumerate(nbrs):
                tr_v = self.X_train[nn]
                te_v = self.X_test[te_idx]
                cost = dtw(tr_v,
                           te_v,
                           **self.metric_params)
                costs[i] = cost
            sorted_cost_inds = np.argsort(costs)
            nbrs = np.asarray(nbrs)[sorted_cost_inds]
            nbrs = nbrs[:self.n_neighbors]
            dtw_nbrs_all_query.append(nbrs)
        self.nbrs_all_query = dtw_nbrs_all_query
