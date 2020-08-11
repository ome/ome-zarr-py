#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()


install_requires = []
install_requires += (["numpy"],)
install_requires += (["dask"],)
install_requires += (["zarr"],)
install_requires += (["s3fs"],)
install_requires += (["aiohttp"],)
install_requires += (["requests"],)
install_requires += (["toolz"],)
install_requires += (["vispy"],)


setup(
    name="ome-zarr",
    version="0.0.11",
    author="The Open Microscopy Team",
    url="https://github.com/ome/ome-zarr-py",
    description="Implementation of images in Zarr files.",
    long_description=read("README.rst"),
    py_modules=["ome_zarr", "ome_zarr_cli"],
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
        "console_scripts": ["ome_zarr = ome_zarr_cli:main"],
        "napari.plugin": ["ome_zarr = ome_zarr"],
    },
    extras_require={"napari": ["napari"]},
    tests_require=["pytest", "pytest-capturelog"],
)
