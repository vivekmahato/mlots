import annoy
import hnswlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import dtw, lb_keogh
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


class AnnoyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, mac_neighbors=None, metric='euclidean',
                 metric_params=None, n_trees=-1, random_seed=1992):

        """
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
        metric_params:  dict (default None)
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
        AnnoyClassifier class with the parameters supplied.
        """

        if metric_params is None:
            metric_params = dict()
        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.metric_params = metric_params
        self.random_seed = random_seed

    def fit(self, X_train, y_train):
        """
        This is the fit function for NSW model.

        Parameters
        ----------
        X_train :   np.array
                    The train data to be fitted.
        y_train :   np.array
                    The true labels of X_train data.

        Returns
        -------
        AnnoyClassifier class with fitted train data.
        """
        self.X_train = X_train
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
        """
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
        self.X_test = X_test
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


class HNSWClassifier(BaseEstimator, ClassifierMixin):
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

        """
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
                                MAC stage: mac_neighbors are returned using 'metric'.
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
        metric_params   :   dict (default None)
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
        HNSWClassifier class with the parameters supplied.
        """

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
        """
        This is the fit function for HNSWClassifier model.

        Parameters
        ----------
        X_train :   np.array
                    The train data to be fitted.
        y_train :   np.array
                    The true labels of X_train data.

        Returns
        -------
        HNSWClassifier class with fitted train data.
        """
        self.X_train = X_train.astype('float32')
        self.dimension = self.X_train.shape[1]
        self.num_elements = self.X_train.shape[0]
        self.y_train = y_train

        self.model = hnswlib.Index(space=self.space, dim=self.dimension)
        self.model.init_index(max_elements=self.num_elements,
                              ef_construction=self.ef_construction, M=self.M)
        self.model.set_ef(self.ef_Search)
        self.model.set_num_threads(self.n_jobs)
        self.model.add_items(self.X_train, np.arange(self.num_elements))
        return self

    def predict(self, X_test):
        """
        This is the predict function for HNSWClassifier model.

        Parameters
        ----------
        X_test :    np.array
                    The test data for the prediction.

        Returns
        -------
        y_hat :     np.array
                    The predicted labels of the test samples.

        """
        np.random.seed(self.random_seed)
        self.X_test = X_test.astype("float32")
        if self.mac_neighbors is None:
            return self.predict_mac()
        return self.predict_macfac()

    def predict_mac(self):
        self.nbrs_all_query, _ = self.model.knn_query(self.X_test,
                                                      k=self.n_neighbors)
        y_hat = np.empty(self.X_test.shape[0])
        for i, nbrs in enumerate(self.nbrs_all_query):
            labels = list(self.y_train[nbrs])
            y_hat[i] = max(set(labels), key=labels.count)
        return y_hat

    def predict_macfac(self):
        self.nbrs_all_query, _ = self.model.knn_query(self.X_test,
                                                      k=self.mac_neighbors)
        self.nn_dtw()
        y_hat = np.empty(self.X_test.shape[0])
        for i, nbrs in enumerate(self.nbrs_all_query):
            nn_classes = [self.y_train[nn] for nn in nbrs]
            y_hat[i] = max(set(nn_classes), key=nn_classes.count)
        return y_hat

    def nn_dtw(self):
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


class kNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_neighbors=5, mac_neighbors=None, weights="uniform", mac_metric="euclidean",
                 metric_params=None, n_jobs=-1):
        """
       NAME: kNNClassifier

       This is a class that represents Annoy model with MAC/FAC strategy.

        Parameters
        ----------
        n_neighbors     :   int (default 5)
                            The n (or k) neighbors to consider for classification.
        mac_neighbors   :   int (default None)
                            Number of neighbors to consider for MAC stage.
                            If None, n_neighbors are used for classification directly.
                            If int; the classification is in two stages:
                                MAC stage: mac_neighbors are returned using 'metric'.
                                FAC stage: n_neighbors are used for classification using DTW.
        weights         :   str (default "uniform")
                            The weighting scheme of the distances. Options: "uniform" or "distance"
        mac_metric      :   str (default "euclidean")
                            The distance metric to be employed for MAC stage.
                            Check tslearn's KNeighborsTimeSeriesClassifier model for allowed metrics.
        metric_params   :   dict (default None)
                            The parameters of the metric being employed for FAC stage.
                            Example: For metric = "dtw", the metric_params can be:
                                { "global_restraint" : "sakoe_chiba",
                                  "sakoe_chiba_radius": 1  }
                            Check tslearn's KNeighborsTimeSeriesClassifier model for allowed metrics.
        n_jobs          :   int (default -1)
                            The number of CPU threads to use. -1 to use all the available threads.

        Returns
        -------
        kNNClassifier class with the parameters supplied.
        """
        if metric_params is None:
            metric_params = {}
        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.mac_metric = mac_metric
        self.weights = weights
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X_train, y_train):
        """
        This is the fit function for kNNClassifier model.

        Parameters
        ----------
        X_train :   np.array
                    The train data to be fitted.
        y_train :   np.array
                    The true labels of X_train data.

        Returns
        -------
        kNNClassifier class with fitted train data.
        """
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train

        self.model = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors,
                                                    metric=self.mac_metric,
                                                    weights=self.weights,
                                                    n_jobs=self.n_jobs).fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test):
        """
        This is the predict function for kNNClassifier model.

        Parameters
        ----------
        X_test :    np.array
                    The test data for the prediction.

        Returns
        -------
        y_hat :     np.array
                    The predicted labels of the test samples.

        """
        if self.mac_neighbors is None:
            return self.model.predict(X_test)

        y_hat = np.empty(X_test.shape[0])
        k_neighbors = self.model.kneighbors(X_test,
                                            n_neighbors=self.mac_neighbors,
                                            return_distance=False)
        for idx, k in enumerate(k_neighbors):
            X_train = self.X_train[k]
            y_train = self.y_train[k]
            self.model = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors,
                                                        metric="dtw",
                                                        weights=self.weights,
                                                        n_jobs=self.n_jobs,
                                                        metric_params=self.metric_params).fit(X_train, y_train)
            pred = self.model.predict(X_test[idx])
            y_hat[idx] = pred
        return y_hat


class kNNClassifier_CustomDist(BaseEstimator, ClassifierMixin):

    def __init__(self, n_neighbors=5, mac_neighbors=None, weights="uniform", mac_metric="lb_keogh",
                 metric_params=None, n_jobs=-1):
        """
       NAME: kNNClassifier_CustomDist

       This is a class that represents Annoy model with MAC/FAC strategy.

        Parameters
        ----------
        n_neighbors     :   int (default 5)
                            The n (or k) neighbors to consider for classification.
        mac_neighbors   :   int (default None)
                            Number of neighbors to consider for MAC stage.
                            If None, n_neighbors are used for classification directly.
                            If int; the classification is in two stages:
                                MAC stage: mac_neighbors are returned using 'metric'.
                                FAC stage: n_neighbors are used for classification using DTW.
        weights         :   str (default "uniform")
                            The weighting scheme of the distances. Options: "uniform" or "distance"
        mac_metric      :   str (default "lb_keogh")
                            The distance metric to be employed for MAC stage.
                            Options:    "lb_keogh",
                                        any allowed distance measures for scikit-learn's KNeighborsClassifier,
                                        or, a callable distance function.
                            If mac_metric = "lb_keogh", provide "radius" parameter for it in metric_params.
        metric_params   :   dict (default None)
                            The parameters of the metric being employed for FAC stage.
                            Example: For metric = "dtw", the metric_params can be:
                                { "global_restraint" : "sakoe_chiba",
                                  "sakoe_chiba_radius": 1  }
                            Check tslearn's KNeighborsTimeSeriesClassifier model for allowed metrics.
        n_jobs          :   int (default -1)
                            The number of CPU threads to use. -1 to use all the available threads.

        Returns
        -------
        kNNClassifier_CustomDist class with the parameters supplied.
        """

        if metric_params is None:
            metric_params = {}
        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.mac_metric = mac_metric
        self.weights = weights
        self.metric_params = metric_params
        self.n_jobs = n_jobs

        if self.mac_metric == "lb_keogh":
            def lbk(ts1, ts2, radius=self.metric_params["radius"]):
                return lb_keogh(ts1, ts2, radius)

            self.mac_metric = lbk
            del self.metric_params["radius"]

    def fit(self, X_train, y_train):
        """
        This is the fit function for kNNClassifier_CustomDist model.

        Parameters
        ----------
        X_train :   np.array
                    The train data to be fitted.
        y_train :   np.array
                    The true labels of X_train data.

        Returns
        -------
        kNNClassifier_CustomDist class with fitted train data.
        """
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train

        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                          metric=self.mac_metric,
                                          weights=self.weights,
                                          algorithm="brute",
                                          n_jobs=self.n_jobs).fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test):
        """
        This is the predict function for kNNClassifier_CustomDist model.

        Parameters
        ----------
        X_test :    np.array
                    The test data for the prediction.

        Returns
        -------
        y_hat :     np.array
                    The predicted labels of the test samples.

        """
        if self.mac_neighbors is None:
            return self.model.predict(X_test)

        y_hat = np.empty(X_test.shape[0])
        k_neighbors = self.model.kneighbors(X_test,
                                            n_neighbors=self.mac_neighbors,
                                            return_distance=False)
        for idx, k in enumerate(k_neighbors):
            X_train = self.X_train[k]
            y_train = self.y_train[k]
            self.model = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors,
                                                        metric="dtw",
                                                        weights=self.weights,
                                                        n_jobs=self.n_jobs,
                                                        metric_params=self.metric_params).fit(X_train, y_train)
            pred = self.model.predict(X_test[idx])
            y_hat[idx] = pred
        return y_hat
