Machine Learning On Time-Series (MLOTS)
=======================================
.. role:: raw-html(raw)
    :format: html

.. image:: source/signal.gif

:raw-html:`<br />`

.. image:: https://travis-ci.com/vivekmahato/mlots.svg?branch=main
    :target: https://travis-ci.com/vivekmahato/mlots
.. image:: https://codecov.io/gh/vivekmahato/mlots/branch/main/graph/badge.svg?token=YRbBDwzetb
    :target: https://codecov.io/gh/vivekmahato/mlots
.. image:: https://img.shields.io/pypi/pyversions/mlots.svg
    :target: https://pypi.python.org/pypi/mlots/
.. image:: https://readthedocs.org/projects/mlots/badge/?version=latest
    :target: http://mlots.readthedocs.io/?badge=latest
.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
.. image:: https://img.shields.io/twitter/url/https/twitter.com/mistermahato.svg?style=social&label=Follow
    :target: https://twitter.com/mistermahato


``mlots``
=========

``mlots`` provides Machine Learning tools for time-series classification.
This package builds on (and hence depends on) `scikit-learn <https://scikit-learn.org//>`_, `numpy <https://numpy.org/>`_, `tslearn <https://tslearn.readthedocs.io/>`_, `annoy <https://github.com/spotify/annoy>`_, and `hnswlib <https://github.com/nmslib/hnswlib>`_ libraries.

It can be installed as a python package from the `PyPI <https://pypi.org/project/mlots/>`_ repository.

Installation
------------

Install ``mlots`` by running:
::

   pip install mlots

After installation, it can be imported to a ``python`` environment to be employed.
::

   import mlots

Getting Started
---------------

.. toctree::
   :titlesonly:
   :caption: Demo Notebooks:

   source/AnnoyClassifier_Demo/AnnoyClassifier_Demo.rst
   source/NSW_Demo/NSWClassifier_Demo.rst
   source/kNNClassifier_Demo/kNNClassifier_Demo.rst
   source/MACFAC_Demo/MACFAC_Demo.rst
   source/MINIROCKET_Demo/MINIROCKET_Demo.rst


Models
--------

.. toctree::
   :maxdepth: 2
   :caption: mlots Contents:

   source/mlots.rst
   source/mlots.models.rst
   source/mlots.transformation.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Contribute
----------

Source Code: https://github.com/vivekmahato/mlots

Support
-------

| If you are having issues, please let us know.
|
| Issue Tracker: https://github.com/vivekmahato/mlots/issues

License
-------

The project is licensed under the BSD 3-Clause license.
