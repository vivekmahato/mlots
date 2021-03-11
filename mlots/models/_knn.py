import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import lb_keogh
from tslearn.neighbors import KNeighborsTimeSeriesClassifier


class kNNClassifier(BaseEstimator, ClassifierMixin):
    r"""
    NAME: kNNClassifier

    This is a class that represents kNNClassifier model with MAC/FAC strategy.

    Parameters
    ----------
    n_neighbors     :   int (default 5)
                        The n (or k) neighbors to consider for classification.
    mac_neighbors   :   int (default None)
                        Number of neighbors to consider for MAC stage.
                        If None, n_neighbors are used for classification directly.

                        If int; the classification is in two stages:
                            MAC stage: mac_neighbors are returned using 'mac_metric'.

                            FAC stage: n_neighbors are used for classification using DTW.
    weights         :   str (default "uniform")
                        The weighting scheme of the distances.

                        Options:
                            "uniform" or "distance"
    mac_metric      :   str (default "euclidean")
                        The distance metric to be employed for MAC stage.
                        Check tslearn.neighbors.KNeighborsTimeSeriesClassifier model for allowed metrics.
    metric_params:      dict() (default None)
                        The parameters of the metric being employed for FAC stage.
                        Example: For metric = "dtw", the metric_params can be:
                            { "global_restraint" : "sakoe_chiba", "sakoe_chiba_radius": 1  }
                        See tslearn.metrics for more details.
    n_jobs          :   int (default -1)
                        The number of CPU threads to use. -1 to use all the available threads.

    Returns
    -------
    object          :   self
                        kNNClassifier class with the parameters supplied.
    See Also
    --------
    tslearn.neighbors.KNeighborsTimeSeriesClassifier:   The underlying k-NN module for time-series data.
    tslearn.metrics.dtw:                                The underlying dtw function.

    Examples
    --------
    >>> from mlots.models import kNNClassifier
    >>> model = kNNClassifier(n_neighbors=5)
    >>> model.fit(X_train, y_train)
    >>> model.score(X_test, y_test)
    >>> 0.7814569536423841
    """

    def __init__(self, n_neighbors=5, mac_neighbors=None, weights="uniform", mac_metric="euclidean",
                 metric_params=None, n_jobs=-1):

        if metric_params is None:
            metric_params = {}
        self.n_neighbors = n_neighbors
        self.mac_neighbors = mac_neighbors
        self.mac_metric = mac_metric
        self.weights = weights
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X_train, y_train):
        r"""
        This is the fit function for kNNClassifier model.

        Parameters
        ----------
        X_train :   ndarray
                    The train data to be fitted.
        y_train :   array
                    The true labels of X_train data.

        Returns
        -------
        object  :   self
                    kNNClassifier class with train data fitted.

        """
        self.X_train = X_train.astype("float32")
        self.y_train = np.asarray(y_train)

        self.model = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors,
                                                    metric=self.mac_metric,
                                                    weights=self.weights,
                                                    n_jobs=self.n_jobs).fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test):
        r"""
        This is the predict function for kNNClassifier model.

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
    r"""
    NAME: kNNClassifier_CustomDist

    This is a class that represents kNNClassifier_CustomDist model with MAC/FAC strategy.

    Parameters
    ----------
    n_neighbors     :   int (default 5)
                        The n (or k) neighbors to consider for classification.
    mac_neighbors   :   int (default None)
                        Number of neighbors to consider for MAC stage.
                        If None, n_neighbors are used for classification directly.

                        If int; the classification is in two stages:
                            MAC stage: mac_neighbors are returned using 'mac_metric'.

                            FAC stage: n_neighbors are used for classification using DTW.
    weights         :   str (default "uniform")
                        The weighting scheme of the distances. Options: "uniform" or "distance"
    mac_metric      :   str (default "lb_keogh")
                        The distance metric to be employed for MAC stage.

                        Options:
                            "lb_keogh",

                            any allowed distance measures for sklearn.neighbors.KNeighborsClassifier,

                            or, a callable distance function.
                        If mac_metric = "lb_keogh", provide "radius" parameter for it in metric_params.
    metric_params:      dict() (default None)
                        The parameters of the metric being employed.
                        Example: For metric = "dtw", the metric_params can be:
                            { "global_restraint" : "sakoe_chiba", "sakoe_chiba_radius": 1  }
                        Check tslearn.neighbors.KNeighborsTimeSeriesClassifier module for allowed metrics.
    n_jobs          :   int (default -1)
                        The number of CPU threads to use. -1 to use all the available threads.
    See Also
    --------
    sklearn.neighbors.KNeighborsClassifier:             The underlying k-NN module for MAC stage with custom distance measure.
    tslearn.neighbors.KNeighborsTimeSeriesClassifier:   The underlying k-NN module for FAC stage with dtw.
    tslearn.metrics.dtw:                                The underlying dtw function.

    Returns
    -------
    object          :   self
                        kNNClassifier_CustomDist class with the parameters supplied.
    Examples
    --------
    >>> from mlots.models import kNNClassifier_CustomDist
    >>> model = kNNClassifier_CustomDist(mac_metric="lb_keogh", mac_neighbors=20, metric_params={"radius": 23})
    >>> model.fit(X_train, y_train)
    >>> model.score(X_test, y_test)
    >>> 0.7748344370860927
    """
    def __init__(self, n_neighbors=5, mac_neighbors=None, weights="uniform", mac_metric="lb_keogh",
                 metric_params=None, n_jobs=-1):

        if metric_params is None:
            metric_params = {"radius": 1}
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
        r"""
        This is the fit function for kNNClassifier_CustomDist model.

        Parameters
        ----------
        X_train :   ndarray
                    The train data to be fitted.
        y_train :   array
                    The true labels of X_train data.

        Returns
        -------
        object  :   self
                    kNNClassifier_CustomDist class with train data fitted.

        """
        self.X_train = X_train.astype("float32")
        self.y_train = np.asarray(y_train)

        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                          metric=self.mac_metric,
                                          weights=self.weights,
                                          algorithm="brute",
                                          n_jobs=self.n_jobs).fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test):
        r"""
        This is the predict function for kNNClassifier_CustomDist model.

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
