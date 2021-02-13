# %%
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import unittest

from mlots.nsw import NSWClassifier


class TestNSW(unittest.TestCase):

    def setUp(self) -> None:
        print("Starting a test in NSW..")
        data = np.load("../input/plarge300.npy", allow_pickle=True).item()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(data['X'], data['y'], test_size=0.5,
                             random_state=1992)

    def tearDown(self):
        # will be executed after every test
        self.X_train, self.X_test, self.y_train, self.y_test = \
            (None, None, None, None)

    def test_gscv_works(self):
        param_dict = {
            'f': np.arange(1, 3, 1),
            'm': np.arange(1, 3, 1),
            'k': np.arange(1, 3, 1)
        }
        model = NSWClassifier(random_seed=42)
        gscv = GridSearchCV(model, param_dict, cv=2,
                            scoring="accuracy", n_jobs=-1)
        gscv.fit(self.X_train, self.y_train)
        assert gscv.best_params_ is not None
        assert gscv.best_score_ is not None

    def test_NSW_euc(self):
        nsw = NSWClassifier(f=1, k=5, m=9, metric="euclidean",
                  random_seed=42)
        nsw.fit(self.X_train, self.y_train)
        y_hat = nsw.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 0.695364238410596, "NSW-EUC Failed!")

    def test_NSW_lb_keogh(self):
        nsw = NSWClassifier(f=1, k=5, m=9, metric="lb_keogh",
                  metric_params={"radius": 23})
        nsw.fit(self.X_train, self.y_train)
        y_hat = nsw.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 0.6887417218543046, "NSW-LB_Keogh Failed!")

    def test_NSW_dtw(self):
        nsw = NSWClassifier(f=1, k=5, m=9, metric="dtw",
                            metric_params={"global_constraint": "sakoe_chiba",
                                           "sakoe_chiba_radius": 23})
        nsw.fit(self.X_train, self.y_train)
        y_hat = nsw.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 0.6754966887417219, "NSW-DTW Failed!")

    def test_NSW_dtw(self):
        nsw = NSWClassifier(f=1, k=5, m=9, metric="dtw",
                            metric_params={"global_constraint": "sakoe_chiba",
                                           "sakoe_chiba_radius": 23})
        nsw.fit(self.X_train, self.y_train)
        y_hat = nsw.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 0.6754966887417219, "NSW-DTW Failed!")

    def test_NSW_kneighbors_v1(self):
        nsw = NSWClassifier(f=1, k=5, m=9, metric="euclidean",
                            random_seed=42)
        nsw.fit(self.X_train, self.y_train)
        nns, y_hat = nsw.kneighbors(self.X_test, return_prediction=True)
        acc = accuracy_score(y_hat, self.y_test)
        np.testing.assert_array_equal(nns[2], [41., 22., 18., 38., 103.], "test_NSW_kneighbors_v1 Failed!")
        self.assertEqual(acc, 0.695364238410596, "test_NSW_kneighbors_v1 Failed!")

    def test_NSW_kneighbors_v2(self):
        nsw = NSWClassifier(f=1, k=5, m=9, metric="euclidean",
                            random_seed=42)
        nsw.fit(self.X_train, self.y_train)
        nns = nsw.kneighbors(self.X_test, return_prediction=False)
        np.testing.assert_array_equal(nns[2], [41., 22., 18., 38., 103.], "test_NSW_kneighbors_v2 Failed!")

    def test_str(self):
        nsw = NSWClassifier(f=1, k=5, m=9, metric="euclidean",
                            random_seed=42)
        nsw.fit(self.X_train, self.y_train)
        assert str(nsw.corpus[0]) == "Node(index=0, Label=1)"

    def test_repr(self):
        nsw = NSWClassifier(f=1, k=5, m=9, metric="euclidean",
                            random_seed=42)
        nsw.fit(self.X_train, self.y_train)
        assert repr(nsw.corpus[0]) == "index: 0, label:1"


if __name__ == '__main__':
    unittest.main()
