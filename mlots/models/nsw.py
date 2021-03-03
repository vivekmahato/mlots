import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sortedcollections import ValueSortedDict
from tqdm import tqdm
from tslearn.metrics import dtw, lb_keogh


class Node:
    r"""
    NAME: Node

    This is a class that represents a Node in the NSW network.

    Parameters
    ----------
    index       :       int
                        The index of the Node.
    values      :       array
                        The time-series datum.
    label       :       int or str (default None)
                        The ground truth class/label of the time-series datum.

    Attributes
    ----------
    neighbors   :       ValueSortedDict()
                        It stores the nodes/neighbors in the network connected to this node. The indices of the nodes
                        are the keys, and the values are the distances between the two nodes.

    Returns
    -------
    object      :       self
                        Node class with the parametric values supplied.

    """
    def __init__(self, index: int, values: list, label=None):
        self.index = index
        self.values = values
        self.label = label
        self.neighbors = ValueSortedDict()

    def __repr__(self):
        return 'index: ' + str(self.index) + ', label:' + str(self.label)

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


class NSWClassifier(BaseEstimator, ClassifierMixin):
    r"""
    NAME: Navigable Small Worlds

    This is a class that represents NSW model.

    Parameters
    ----------
    f       :       int (default 1)
                       The maximum number of friends a node can have or connect to.
    m       :       int (default 1)
                       Number of iterations or search in the network.
    k       :       int (default 1)
                       The number of neighbors to consider for classification.
    metric  :       str (default "euclidean")
                       The distance metric/measure to be employed. Can be one from the list: euclidean, dtw, lb_keogh
    metric_params:  dict() (default None)
                       The parameters of the metric being employed.
                           Example: For metric = "dtw", the metric_params can be:
                                       {   "global_restraint" : "sakoe_chiba",
                                           "sakoe_chiba_radius": 1             }
                       See tslearn.metrics for more details.
    random_seed:    int (default 1992)
                       The initial seed to be used by random function.

    Attributes
    ----------
    corpus  :       dict()
                        It stores the all the nodes in the network. The keys are the indices of the nodes and the values
                        are the node objects of Node class.

    Returns
    -------
    object  :       self
                       NSW class with the parameters supplied.

    """

    def __init__(self,
                 f: int = 1,
                 m: int = 1,
                 k: int = 1,
                 metric: str = "euclidean",
                 metric_params=None,
                 random_seed: int = 1992) -> object:

        if metric_params is None:
            metric_params = dict()
        self.random_seed = random_seed
        self.f = f
        self.m = m
        self.k = k
        self.metric = metric
        self.metric_params = metric_params
        self.corpus = {}

    def _switch_metric(self, ts1=None, ts2=None):
        if self.metric == "euclidean":
            return np.linalg.norm(ts1 - ts2)
        if self.metric == "dtw":
            return dtw(ts1, ts2, **self.metric_params)
        if self.metric == "lb_keogh":
            return lb_keogh(ts1, ts2, **self.metric_params)

    def _nn_insert(self, index=int, values=[], label=None):
        # create node with the given values
        node = Node(index, values, label)

        # check if the corpus is empty
        if len(self.corpus) < 1:
            self.corpus[node.index] = node
            return self

        neighbors, _ = self._knn_search(node, self.f)

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

    def _batch_insert(self, indices=None):
        for i in tqdm(list(range(self.X_train.shape[0]))):
            self._nn_insert(indices[i], self.X_train[i], self.y_train[i])

        return self

    def _get_closest(self):
        k = next(iter(self.candidates))
        return {k: self.candidates.pop(k)}

    def _check_stop_condition(self, c, k):
        # if c is further than the kth element in the result

        k_dist = self.result[list(self.result.keys())[k - 1]]
        c_dist = list(c.values())[0]

        return bool(c_dist > k_dist)

    def _knn_search(self, q=None, k=1):

        self.q = q
        self.visitedset = set()
        self.candidates = ValueSortedDict()
        self.result = ValueSortedDict()
        count = 0

        for i in range(self.m):
            v_ep = self.corpus[np.random.choice(list(self.corpus.keys()))]
            if self.dmat is None:
                cost = self._switch_metric(self.q.values, v_ep.values)
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
                    c = self._get_closest()
                else:
                    break

                # check stop condition
                if len(self.result) >= k:
                    if self._check_stop_condition(c, k):
                        break
                tempres.update(c)

                # add neighbors of c if not in visitedset
                c = self.corpus[list(c.keys())[0]]

                for key in list(c.neighbors.keys()):
                    if key not in self.visitedset:
                        if self.dmat is None:
                            cost = self._switch_metric(self.q.values, v_ep.values)
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
        r"""
        This is the fit function for NSW model.

        Parameters
        ----------
        X_train :   ndarray
                    The train data to be fitted.
        y_train :   array
                    The true labels of X_train data.
        dist_mat :  ndarray (default None)
                    [Optional] Pre-computed distance matrix for X_train vs X_train

        Returns
        -------
        object  :   self
                    NSW class with train data fitted.

        """
        np.random.seed(self.random_seed)
        self.X_train = X_train.astype("float32")
        self.y_train = np.asarray(y_train)
        self.dmat = dist_mat

        indices = np.arange(len(X_train))
        self._batch_insert(indices)

        print("Model is fitted with the provided data.")
        return self

    def predict(self, X_test, dist_mat=None):

        r"""
        This is the predict function for NSW model.

        Parameters
        ----------
        X_test :    ndarray
                    The test data for the prediction.
        dist_mat :  ndarray (default None)
                    [Optional] Pre-computed distance matrix for X_test vs X_train

        Returns
        -------
        y_hat :     array
                    The predicted labels of the test samples.

        """
        X_test = X_test.astype("float32")
        y_hat = np.empty(X_test.shape[0])
        self.dmat = dist_mat

        for i in tqdm(range(X_test.shape[0])):
            q_node = Node(i, X_test[i], None)

            neighbors, _ = self._knn_search(q_node, self.k)

            labels = [self.corpus[key].label for key in list(neighbors.keys())[:self.k]]

            label = max(set(labels), key=labels.count)

            y_hat[i] = label

        return y_hat

    def kneighbors(self, X_test=None, dist_mat=None, return_prediction=False):
        """
        This is the kneighbors function for NSW model. The kneighbors are fetched for the test samples.

        Parameters
        ----------
        X_test :    ndarray
                    The test data for the prediction.
        dist_mat :  ndarray (default None)
                    [Optional] Pre-computed distance matrix for X_test vs X_train
        return_prediction: bool (default False)
                    If True, the function returns kneighbors and predictions (nns and y_hat)

        Returns
        -------
        nns     :   ndarray
                    The kneighbors of the test samples.
        y_hat   :   array
                    The predicted labels of the test samples.

        """
        X_test = X_test.astype("float32")
        self.dmat = dist_mat
        nns = np.empty((X_test.shape[0], self.k))
        y_hat = np.empty(X_test.shape[0])
        # counts = []

        for i in tqdm(range(X_test.shape[0])):
            q_node = Node(i, X_test[i], None)
            neighbors, _ = self._knn_search(q_node, self.k)
            # counts.append(count)
            neighbors = list(neighbors.keys())[:self.k]
            if return_prediction:
                lst = list(self.y_train[neighbors])
                y_hat[i] = max(set(lst), key=lst.count)
            nns[i] = np.asarray(neighbors)
        if return_prediction:
            return nns, y_hat
        return nns
