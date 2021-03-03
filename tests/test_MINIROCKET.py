import unittest
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from mlots import RidgeClassifier, RidgeClassifierCV
from mlots.transformation.minirocket import MINIROCKET


class TestMINIROCKET(unittest.TestCase):
    def setUp(self) -> None:
        print("Starting a test in TestMINIROCKET..")
        data = np.load("input/AM_Datasets/plarge300.npy", allow_pickle=True).item()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(data['X'], data['y'], test_size=0.5,
                             random_state=1992)
        minirocket = MINIROCKET()
        minirocket.fit(self.X_train)
        self.X_train = minirocket.transform(self.X_train)
        self.X_test = minirocket.transform(self.X_test)

    def tearDown(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            (None, None, None, None)

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
        self.assertEqual(acc, 0.9271523178807947,
                         "test_MINIROCKETClassification!")
