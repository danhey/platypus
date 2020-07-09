#!/usr/bin/env python

import os
import sys
from setuptools import setup, Extension

# Get dependencies
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="platypus",
    version='0.0.1',
    license="MIT",
    package_dir={"platypus": "platypus",},
    packages=["platypus"],
    install_requires=install_requires,
    url="https://github.com/danhey/platypus",
)