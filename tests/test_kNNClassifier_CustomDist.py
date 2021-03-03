import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import unittest

from mlots.models import kNNClassifier_CustomDist


class TestkNNClassifier_CustomDist(unittest.TestCase):

    def setUp(self) -> None:
        print("Starting a test in TestkNNClassifier_CustomDist..")
        data = np.load("input/AM_Datasets/plarge300.npy", allow_pickle=True).item()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(data['X'], data['y'], test_size=0.5,
                             random_state=1992)

    def tearDown(self):
        # will be executed after every test
        self.X_train, self.X_test, self.y_train, self.y_test = \
            (None, None, None, None)

    def test_gscv_works(self):
        param_grid = {
            "n_neighbors": np.arange(1, 4, 2),
            "weights": ["uniform", "distance"]
        }
        model = kNNClassifier_CustomDist()
        gscv = GridSearchCV(model, param_grid, cv=2,
                            scoring="accuracy", n_jobs=-1)
        gscv.fit(self.X_train, self.y_train)
        assert gscv.best_params_ is not None
        assert gscv.best_score_ is not None

    def test_kNNClassifier_CustomDist_LB_Keogh_wo_MACFAC(self):
        model = kNNClassifier_CustomDist(mac_metric="lb_keogh",
                                 metric_params={"radius": 23},
                                 n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_hat = model.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 0.7814569536423841,
                         "kNNClassifier_CustomDist_LB_Keogh_wo_MACFAC Failed!")

    def test_kNNClassifier_CustomDist_LB_Keogh_w_MACFAC(self):
        model = kNNClassifier_CustomDist(mac_metric="lb_keogh",
                                mac_neighbors=20,
                                 metric_params={"radius": 23},
                                 n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_hat = model.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 0.7748344370860927,
                         "kNNClassifier_CustomDist_LB_Keogh_w_MACFAC Failed!")

    def test_kNNClassifier_CustomDist_CustomMeasure(self):
        def dist_measure(ts1, ts2):
            return np.linalg.norm(ts1 - ts2)

        measure = dist_measure

        model = kNNClassifier_CustomDist(mac_metric=measure,
                                         n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        y_hat = model.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 0.7814569536423841,
                         "test_kNNClassifier_CustomDist_CustomMeasure Failed!")


# if __name__ == '__main__':
#     unittest.main()
