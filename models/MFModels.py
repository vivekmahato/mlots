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


class HNSWClassifier:
    def __init__(self,
                 n_neighbors=1,
                 mac_neighbors=None,
                 space="l2",
                 max_elements=10,
                 M=5,
                 ef_construction=100,
                 ef_Search=50,
                 sakoe_chiba_radius=1,
                 random_seed=1992,
                 num_threads=40):

        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.space = space
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef_S = ef_Search
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.seed = random_seed
        self.num_threads = num_threads

    def get_params(self, deep=True):
        return {
            "n_neighbors": self.n_neighbors,
            "mac_neighbors": self.mac_neighbors,
            "space": self.space,
            "max_elements": self.max_elements,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_Search": self.ef_S,
            "sakoe_chiba_radius": self.sakoe_chiba_radius,
            "random_seed": self.seed,
            "num_threads": self.num_threads
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X_train=None, y_train=None):
        self.X_train = X_train.astype('float32')
        self.dimesion = self.X_train.shape[1]
        self.num_elements = self.X_train.shape[0]
        self.y_train = y_train

        self.model = hnswlib.Index(space=self.space, dim=self.dimesion)
        self.model.init_index(max_elements=self.num_elements,
                              ef_construction=self.ef_construction, M=self.M)
        self.model.set_ef(self.ef_S)
        self.model.add_items(self.X_train, np.arange(self.num_elements))
        return self

    def predict(self, X_test):
        np.random.seed(self.seed)
        self.X_test = X_test.astype("float32")
        if self.mac_neighbors is None:
            return self.predict_euc()
        return self.predict_dtw()

    def predict_dtw(self):
        self.nbrs_all_query, distances = self.model.knn_query(self.X_test,
                                                              k=self.mac_neighbors)
        self.nn_dtw()
        y_hat = []
        for nbrs in self.nbrs_all_query:
            nn_classes = [self.y_train[nn] for nn in nbrs]
            y_hat.append(most_frequent(nn_classes))
        return y_hat

    def nn_dtw(self):
        dtw_nbrs_all_query = []
        for te_idx, nbrs in enumerate(self.nbrs_all_query):
            costs = []
            for nn in nbrs:
                tr_v = self.X_train[nn]
                te_v = self.X_test[te_idx]
                cost = dtw(tr_v,
                           te_v,
                           global_constraint="sakoe_chiba",
                           sakoe_chiba_radius=self.sakoe_chiba_radius)
                costs.append(cost)
            sorted_cost_inds = np.argsort(np.array(costs))
            nbrs = np.asarray(nbrs)[sorted_cost_inds]
            nbrs = nbrs[:self.n_neighbors]
            dtw_nbrs_all_query.append(nbrs)
        self.nbrs_all_query = dtw_nbrs_all_query

    def predict_euc(self):
        self.nbrs_all_query, distances = self.model.knn_query(self.X_test,
                                                              k=self.n_neighbors)
        y_hat = []
        for nbrs in self.nbrs_all_query:
            labels = self.y_train[nbrs]
            pred = most_frequent(labels)
            y_hat.append(pred)
        return y_hat


class kNNClassifier:
    def __init__(self, n_neighbours=5, mac_neighbours=None, weights="uniform",
                 metric_params={}, n_jobs=-1):
        self.n_neighbours = n_neighbours
        self.mac_neighbours = mac_neighbours
        self.weights = weights
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return {"n_neighbours": self.n_neighbours,
                "mac_neighbours": self.mac_neighbours,
                "weights": self.weights,
                "metric_params": self.metric_params,
                "n_jobs": self.n_jobs
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.model = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbours,
                                                    metric="euclidean",
                                                    weights=self.weights,
                                                    n_jobs=self.n_jobs).fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test):
        if self.mac_neighbours is None:
            return self.model.predict(X_test)
        else:
            y_hat = []
            k_neighbors = self.model.kneighbors(X_test,
                                                n_neighbors=self.mac_neighbours,
                                                return_distance=False)
            for idx, k in enumerate(k_neighbors):
                X_train = self.X_train[k]
                y_train = self.y_train[k]
                self.model = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbours,
                                                            metric="dtw",
                                                            weights=self.weights,
                                                            n_jobs=self.n_jobs,
                                                            metric_params=self.metric_params).fit(X_train, y_train)
                pred = self.model.predict(X_test[idx])
                y_hat.append(pred)
        return y_hat
