#!/usr/bin/env python3

from distutils.core import setup

setup(
	name='pysdo',
	version='0.1',
	description='SDO outlier detection algorithm',
	author='Alexander Hartl',
	author_email='alexander.hartl@tuwien.ac.at',
	url='https://github.com/CN-TU/pysdo',
	packages=['pysdo'],
	requires=['numpy', 'scikit_learn', 'scipy']
)
