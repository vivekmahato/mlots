import os

import setuptools

ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name='mlots',
    version='0.0.4.3',
    author="Vivek Mahato",
    author_email="vivek.mahato@ucdconnect.ie",
    description="Machine Learning Over Time-Series: A toolkit for time-series analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://mlots.readthedocs.io/',
    packages=["mlots"],
    install_requires=[
        'tslearn', 'numpy', 'scikit-learn', 'annoy', 'hnswlib', 'sortedcollections',
        'tqdm', 'pandas'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        'Documentation': 'https://mlots.readthedocs.io/',
        'Source': 'https://github.com/vivekmahato/mlots',
        'Tracker': 'https://github.com/vivekmahato/mlots/issues',
    },
)
