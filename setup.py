#!/usr/bin/env python

import codecs
import os

from setuptools import setup


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()


def get_requirements(filename="requirements.txt"):
    with open(filename) as f:
        rv = f.read().splitlines()
    return rv


install_requires = get_requirements()


setup(
    name="ome-zarr",
    version="0.2.1.dev0",
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
