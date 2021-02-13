import unittest
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from mlots.annoy import AnnoyClassifier


class TestAnnoyClassifier(unittest.TestCase):
    def setUp(self) -> None:
        print("Starting a test in TestAnnoyClassifier..")
        data = np.load("input/plarge300.npy", allow_pickle=True).item()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(data['X'], data['y'], test_size=0.5,
                             random_state=1992)

    def tearDown(self):
        # will be executed after every test
        self.X_train, self.X_test, self.y_train, self.y_test = \
            (None, None, None, None)

    def test_gscv_works(self):
        param_grid = {
            "n_neighbors": np.arange(1, 3, 1)
        }
        model = AnnoyClassifier()
        gscv = GridSearchCV(model, param_grid, cv=2,
                            scoring="accuracy", n_jobs=-1)
        gscv.fit(self.X_train, self.y_train)
        assert gscv.best_params_ is not None
        assert gscv.best_score_ is not None

    def test_AnnoyClassifier_wo_MACFAC(self):
        model = AnnoyClassifier(n_neighbors=9, random_seed=42)
        model.fit(self.X_train, self.y_train)
        y_hat = model.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 0.7880794701986755,
                         "AnnoyClassifier_wo_MACFAC Failed!")

    def test_AnnoyClassifier_w_MACFAC(self):
        model = AnnoyClassifier(n_neighbors=3, mac_neighbors=30,
                              metric_params={
                                  "global_constraint": "sakoe_chiba",
                                  "sakoe_chiba_radius": 23},
                                random_seed=42)
        model.fit(self.X_train, self.y_train)
        y_hat = model.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 0.8145695364238411,
                         "AnnoyClassifier_w_MACFAC Failed!")


if __name__ == '__main__':
    unittest.main()

