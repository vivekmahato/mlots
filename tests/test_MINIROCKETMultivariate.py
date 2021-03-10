import unittest
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from mlots.models import RidgeClassifier, RidgeClassifierCV
from mlots.transformation import MINIROCKET
# from mlots.transformation._rocket_algos._minirocket_multivariate import _fit, _transform


class TestMINIROCKETMultivariate(unittest.TestCase):
    def setUp(self) -> None:
        print("Starting a test in TestMINIROCKETMultivariate..")
        name = "BasicMotions"
        dataset = np.load(f'input/{name}/{name}_TRAIN.npz'.format(name=name))
        self.X_train = dataset["data"]
        self.y_train = dataset["labels"]

        dataset = np.load(f'input/{name}/{name}_TEST.npz'.format(name=name))
        self.X_test = dataset["data"]
        self.y_test = dataset["labels"]

        minirocket = MINIROCKET(ts_type="multivariate", random_seed=1992)
        minirocket.fit(self.X_train)
        self.X_train = minirocket.transform(self.X_train)
        self.X_test = minirocket.transform(self.X_test)

    def tearDown(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            (None, None, None, None)

    def test_MINIROCKETIncompatibleType(self):
        with self.assertRaises(ValueError) as raises:
            MINIROCKET(ts_type="blah")

    def test_gscv_works(self):
        param_grid = {
            "alpha": np.logspace(-3, 3, 10)
        }
        model = RidgeClassifier(normalize=True)
        gscv = GridSearchCV(model, param_grid, cv=5,
                            scoring="accuracy", n_jobs=-1)
        gscv.fit(self.X_train, self.y_train)
        assert gscv.best_params_ is not None
        assert gscv.best_score_ is not None

    def test_MINIROCKETClassification(self):
        model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        model.fit(self.X_train, self.y_train)
        y_hat = model.predict(self.X_test)
        acc = accuracy_score(y_hat, self.y_test)
        self.assertEqual(acc, 1.0,
                         "test_MINIROCKETClassification!")

    # def test_f(self) -> None:
    #     name = "BasicMotions"
    #     dataset = np.load(f'../input/{name}/{name}_TRAIN.npz'.format(name=name))
    #     X_train = dataset["data"].astype("float64")
    #
    #     dataset = np.load(f'../input/{name}/{name}_TEST.npz'.format(name=name))
    #     X_test = dataset["data"].astype("float64")
    #
    #     parameters = _fit(X_train, random_seed=1992)
    #     X_train = _transform(X_train, parameters)
    #     X_test = _transform(X_test, parameters)
    #
    #     np.testing.assert_array_equal(self.X_train, X_train)
    #     np.testing.assert_array_equal(self.X_test, X_test)
