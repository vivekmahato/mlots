import os

import setuptools

ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name='mlots',
    version='0.0.2',
    author="Vivek Mahato",
    author_email="vivek.mahato@ucdconnect.ie",
    description="Machine Learning Over Time-Series: A toolkit for time-series analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/vivekmahato/mlots',
    packages=["mlots"],
    install_requires=[
        'tslearn', 'numpy', 'scikit-learn', 'annoy', 'hnswlib', 'sortedcollections',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
