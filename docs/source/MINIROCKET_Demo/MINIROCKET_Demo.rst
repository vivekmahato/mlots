MINIROCKET
==========

Dempster et al. https://arxiv.org/abs/2012.08791

Importing Packages
------------------

.. code:: ipython3

    from mlots.models import RidgeClassifierCV
    from mlots.transformation import MINIROCKET
    from scipy.io import arff
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.metrics import accuracy_score

Loading Data
------------

| Here we are loading the ``SmoothSubspace`` dataset.
| The datasets are in two ``.arff`` files with pre-defined train and
  test splits.
| The following code reads the two files stores the ``X`` (time-series
  data) and ``y`` (labels), into their specific train and test sets.
  \**\*

.. code:: ipython3

    name = "SmoothSubspace"
    
    dataset = arff.loadarff(f'input/{name}/{name}_TRAIN.arff'.format(name=name))[0]
    X_train = np.array(dataset.tolist(), dtype=np.float32)
    y_train = X_train[: , -1]
    X_train = X_train[:, :-1]
    
    dataset = arff.loadarff(f'input/{name}/{name}_TEST.arff'.format(name=name))[0]
    X_test = np.array(dataset.tolist(), dtype=np.float32)
    y_test = X_test[: , -1]
    X_test = X_test[:, :-1]
    
    #Converting target from bytes to integer
    y_train = [int.from_bytes(el, "little") for el in y_train]
    y_test = [int.from_bytes(el, "little") for el in y_test]
    X_train.shape, X_test.shape




.. parsed-literal::

    ((150, 15), (150, 15))



===== =========== =========
Set   Sample size TS length
===== =========== =========
Train 150         15
Test  150         15
===== =========== =========

Transforming Data using ``MINIROCKET``
--------------------------------------

.. code:: ipython3

    print("Shape of X_train and X_test before transformation: ",X_train.shape,", ",X_test.shape)


.. parsed-literal::

    Shape of X_train and X_test before transformation:  (150, 15) ,  (150, 15)


.. code:: ipython3

    #ts_type denotes if we are using univariate or multivariate version of the algorithm
    #we use "univariate" version as the dataset is a univariate time-series
    minirocket = MINIROCKET(ts_type="univariate") 
    
    minirocket.fit(X_train)
    X_train = minirocket.transform(X_train)
    X_test = minirocket.transform(X_test)

.. code:: ipython3

    print("Shape of X_train and X_test after transformation: ",X_train.shape,", ",X_test.shape)


.. parsed-literal::

    Shape of X_train and X_test after transformation:  (150, 9996) ,  (150, 9996)


Classification
--------------

| We can employ ``RidgeClassifierCV`` as our linear model for the
  classification task.
| \**\*

.. code:: ipython3

    model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    model = model.fit(X_train, y_train)

.. code:: ipython3

    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.2f}%")


.. parsed-literal::

    Model accuracy: 0.95%

