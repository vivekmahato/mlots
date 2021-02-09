import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sortedcollections import ValueSortedDict
from tqdm import tqdm
from tslearn.metrics import dtw, lb_keogh


class Node:
    def __init__(self, index: int, values: list, label=None):
        self.index = index
        self.values = values
        self.label = label
        self.neighbors = ValueSortedDict()

    def __repr__(self):
        return {'index': self.index, 'label': self.label}

    def __str__(self):
        return 'Node(index=' + str(self.index) + ', Label=' + str(self.label) + ')'

    def connect(self, index, cost, f):
        """
        Calculate distance and store in a sorteddict
        """
        # The dict would be sorted by values
        self.neighbors[index] = cost
        while len(self.neighbors) > f:
            self.neighbors.popitem()
        return self


class NSW(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 f: int = 1,
                 m: int = 1,
                 k: int = 1,
                 metric: object = "euclidean",
                 metric_params: dict = {},
                 random_seed: int = 1992) -> object:

        self.seed = random_seed
        self.f = f
        self.m = m
        self.k = k
        self.metric = metric
        self.metric_params = metric_params
        self.corpus = {}

    def switch_metric(self, ts1=None, ts2=None):
        if self.metric == "euclidean":
            return np.linalg.norm(ts1 - ts2)
        elif self.metric == "dtw":
            return dtw(ts1, ts2, **self.metric_params)
        elif self.metric == "lb_keogh":
            return lb_keogh(ts1, ts2, **self.metric_params)
        return None

    def nn_insert(self, index=int, values=[], label=None):
        # create node with the given values
        node = Node(index, values, label)

        # check if the corpus is empty
        if len(self.corpus) < 1:
            self.corpus[node.index] = node
            return self

        neighbors, _ = self.knn_search(node, self.f)

        for key, cost in list(neighbors.items())[:self.f]:
            # have the store the updated node back in the corpus
            neighbor = self.corpus[key]
            assert neighbor.index == key
            neighbor = neighbor.connect(node.index, cost, self.f)
            self.corpus[neighbor.index] = neighbor

            # connect new node to its neighbor
            node = node.connect(neighbor.index, cost, self.f)

        # storing new node in the corpus
        self.corpus[node.index] = node
        return self

    def batch_insert(self, indices=[]):

        for i in tqdm(list(range(self.X_train.shape[0]))):
            self.nn_insert(indices[i], self.X_train[i], self.y_train[i])

        return self

    def get_closest(self):
        k = next(iter(self.candidates))
        return {k: self.candidates.pop(k)}

    def check_stop_condition(self, c, k):
        # if c is further than the kth element in the result

        k_dist = self.result[list(self.result.keys())[k - 1]]
        c_dist = list(c.values())[0]

        return bool(c_dist > k_dist)

    def knn_search(self, q=None, k=1):

        self.q = q
        self.visitedset = set()
        self.candidates = ValueSortedDict()
        self.result = ValueSortedDict()
        count = 0

        for i in range(self.m):
            v_ep = self.corpus[np.random.choice(list(self.corpus.keys()))]
            if self.dmat is None:
                cost = self.switch_metric(self.q.values, v_ep.values)
            else:
                cost = self.dmat[q.index][v_ep.index]
            count += 1
            self.candidates[v_ep.index] = cost
            self.visitedset.add(v_ep.index)
            tempres = ValueSortedDict()

            while True:

                # get element c closest from candidates to q, and remove c
                # from candidates
                if len(self.candidates) > 0:
                    c = self.get_closest()
                else:
                    break

                # check stop condition
                if len(self.result) >= k:
                    if self.check_stop_condition(c, k):
                        break
                tempres.update(c)

                # add neighbors of c if not in visitedset
                c = self.corpus[list(c.keys())[0]]

                for key in list(c.neighbors.keys()):
                    if key not in self.visitedset:
                        if self.dmat is None:
                            cost = self.switch_metric(self.q.values, v_ep.values)
                        else:
                            cost = self.dmat[q.index][v_ep.index]
                        count += 1
                        self.visitedset.add(key)
                        self.candidates[key] = cost
                        tempres[key] = cost

            # add tempres to result
            self.result.update(tempres)
        # return k neighbors/result
        return self.result, count

    def fit(self, X_train, y_train, dist_mat=None):
        np.random.seed(self.seed)
        try:
            self.X_train = X_train.astype("float32")
        except:
            self.X_train = np.asarray(X_train, dtype="float32")
        self.y_train = y_train
        self.dmat = dist_mat

        indices = np.arange(len(X_train))
        self.batch_insert(indices)

        print("Model is fitted with the provided data.")
        return self

    def predict(self, X_test):
        try:
            X_test = X_test.astype("float32")
        except:
            X_test = np.asarray(X_test, dtype="float32")

        y_hat = np.empty(X_test.shape[0])

        for i in tqdm(range(X_test.shape[0])):
            q_node = Node(0, X_test[i], None)

            neighbors, _ = self.knn_search(q_node, self.k)

            labels = [self.corpus[key].label for key in list(neighbors.keys())[:self.k]]

            label = max(set(labels), key=labels.count)

            y_hat[i] = label

        return y_hat

    def kneighbors(self, X_test=None, indices=[], dist_mat=None, return_prediction=False):
        X_test.astype("float32")
        self.dmat = dist_mat
        all_nns = []
        y_hat = np.empty(X_test.shape[0])
        counts = []

        for i in tqdm(range(X_test.shape[0])):
            q_node = Node(indices[i], X_test[i], None)
            neighbors, count = self.knn_search(q_node, self.k)
            counts.append(count)
            neighbors = list(neighbors.keys())[:self.k]
            if return_prediction:
                lst = self.y_train[neighbors]
                y_hat[i] = max(set(lst), key=lst.count)
            all_nns.append(neighbors)
        if return_prediction:
            return all_nns, y_hat, counts
        return all_nns
