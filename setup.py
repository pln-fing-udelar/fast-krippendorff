#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

import pypandoc

setup(
    name="krippendorff",
    version="0.3.0",
    description="Fast computation of the Krippendorff's alpha measure.",
    long_description=pypandoc.convert_file('README.md', 'rst'),
    url="https://github.com/pln-fing-udelar/fast-krippendorff",
    keywords=["Krippendorff", "alpha", "agreement", "reliability", "coding", "coders", "units", "values"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=find_packages(),
    author="Santiago Castro",
    author_email="sacastro@fing.edu.uy",
    maintainer="Santiago Castro",
    maintainer_email="sacastro@fing.edu.uy",
    license="GPL 3.0",
)
