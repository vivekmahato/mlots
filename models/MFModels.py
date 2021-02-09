import annoy
import hnswlib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tslearn.metrics import dtw
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


class AnnoyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, mac_neighbors=None, metric='euclidean',
                 metric_params={}, n_trees=-1, random_seed=1992):

        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.metric = metric
        self.n_trees = n_trees
        self.metric_params = metric_params
        self.random_seed = random_seed

    def fit(self, X_train, y_train):
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
                 metric_params={},
                 random_seed=1992,
                 num_threads=40):

        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.space = space
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef_Search = ef_Search
        self.metric_params = metric_params
        self.random_seed = random_seed
        self.num_threads = num_threads

    def fit(self, X_train=None, y_train=None):
        self.X_train = X_train.astype('float32')
        self.dimension = self.X_train.shape[1]
        self.num_elements = self.X_train.shape[0]
        self.y_train = y_train

        self.model = hnswlib.Index(space=self.space, dim=self.dimension)
        self.model.init_index(max_elements=self.num_elements,
                              ef_construction=self.ef_construction, M=self.M)
        self.model.set_ef(self.ef_Search)
        self.model.add_items(self.X_train, np.arange(self.num_elements))
        return self

    def predict(self, X_test):
        np.random.seed(self.random_seed)
        self.X_test = X_test.astype("float32")
        if self.mac_neighbors is None:
            return self.predict_mac()
        return self.predict_macfac()

    def predict_mac(self):
        self.nbrs_all_query, distances = self.model.knn_query(self.X_test,
                                                              k=self.n_neighbors)
        y_hat = np.empty(self.X_test.shape[0])
        for i, nbrs in enumerate(self.nbrs_all_query):
            labels = list(self.y_train[nbrs])
            y_hat[i] = max(set(labels), key=labels.count)
        return y_hat

    def predict_macfac(self):
        self.nbrs_all_query, distances = self.model.knn_query(self.X_test,
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
                 metric_params={}, n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.mac_metric = mac_metric
        self.weights = weights
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X_train, y_train):
        self.X_train = X_train.astype(np.float32)
        self.y_train = y_train

        self.model = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors,
                                                    metric=self.mac_metric,
                                                    weights=self.weights,
                                                    n_jobs=self.n_jobs).fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test):
        if self.mac_neighbors is None:
            return self.model.predict(X_test)
        else:
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
