import unittest
import numpy as np
import pandas as pd

from mlots import from_pandas_dataframe


class TestFPD(unittest.TestCase):
    def setUp(self) -> None:
        print("Starting a test in TestFPD..")
        data = {
            1: [1, 2, 3, 4, 5],
            2: [6, 7, 8, 9, 10],
            3: [11, 12, 13, 14, 15],
            "labels": ["a", "b", "c", "d", "e"]
        }
        self.d_frame = pd.DataFrame(data=data)

    def tearDown(self):
        # will be executed after every test
        self.d_frame = None

    def test_conversion_wo_split_wo_target(self):
        X = from_pandas_dataframe(self.d_frame.drop(["labels"], axis=1))
        self.assertTupleEqual(X.shape, (5, 3), "test_conversion_wo_split failed!")

    def test_conversion_w_split_wo_target(self):
        X_train, X_test = from_pandas_dataframe(self.d_frame.drop(["labels"], axis=1), test_size=33)
        self.assertTupleEqual(X_train.shape, (3, 3), "test_conversion_wo_split failed!")
        self.assertTupleEqual(X_test.shape, (2, 3), "test_conversion_wo_split failed!")

    def test_conversion_wo_split(self):
        X, y = from_pandas_dataframe(self.d_frame, target="labels")
        self.assertTupleEqual(X.shape, (5, 3), "test_conversion_wo_split failed!")
        self.assertTupleEqual(y.shape, (5,), "test_conversion_wo_split failed!")

    def test_conversion_w_split(self):
        X_train, X_test, y_train, y_test = from_pandas_dataframe(self.d_frame, target="labels", test_size=0.33)
        self.assertTupleEqual(X_train.shape, (3, 3), "test_conversion_w_split failed!")
        self.assertTupleEqual(X_test.shape, (2, 3), "test_conversion_w_split failed!")
        self.assertTupleEqual(y_train.shape, (3,), "test_conversion_w_split failed!")
        self.assertTupleEqual(y_test.shape, (2,), "test_conversion_w_split failed!")

    def test_conversion_w_split_shuffle(self):
        X_train, X_test, y_train, y_test = from_pandas_dataframe(self.d_frame, target="labels", test_size=0.33,
                                                                 shuffle=True, random_seed=42)
        np.testing.assert_array_equal(X_train, [[3., 8., 13.], [1., 6., 11.],
                                                [4., 9., 14.]], "test_conversion_w_split_shuffle failed!")
        np.testing.assert_array_equal(X_test, [[2., 7., 12.], [5., 10., 15.]],
                                      "test_conversion_w_split_shuffle failed!")
        np.testing.assert_array_equal(y_train, ['c', 'a', 'd'], "test_conversion_w_split_shuffle failed!")
        np.testing.assert_array_equal(y_test, ['b', 'e'], "test_conversion_w_split_shuffle failed!")

    def test_keyerror(self):
        with self.assertRaises(KeyError) as raises:
            from_pandas_dataframe(self.d_frame, target="check", test_size=0.33, shuffle=True,
                                  random_seed=42)
            self.assertEqual(raises.exception.message, "KeyError")


if __name__ == '__main__':
    unittest.main()
