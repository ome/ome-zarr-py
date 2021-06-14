#!/usr/bin/env python

import codecs
import os
from typing import List

from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()


install_requires: List[List[str]] = []
install_requires += (["dataclasses;python_version<'3.7'"],)
install_requires += (["tifffile<2020.09.22;python_version<'3.7'"],)
install_requires += (["numpy"],)
install_requires += (["dask"],)
install_requires += (["zarr>=2.8.1"],)
install_requires += (["fsspec>=0.9.0"],)
install_requires += (["s3fs"],)
install_requires += (["aiohttp"],)
install_requires += (["requests"],)
install_requires += (["scikit-image"],)
install_requires += (["toolz"],)
install_requires += (["vispy"],)
install_requires += (["opencv-contrib-python-headless"],)


setup(
    name="ome-zarr",
    version="0.0.20",
    author="The Open Microscopy Team",
    url="https://github.com/ome/ome-zarr-py",
    description="Implementation of images in Zarr files.",
    long_description=read("README.rst"),
    packages=["ome_zarr"],
    py_modules=["ome_zarr"],
    python_requires=">=3.6",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
    entry_points={
        "console_scripts": ["ome_zarr = ome_zarr.cli:main"],
    },
    tests_require=["pytest"],
)
