#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

setup(
    name='calib',
    version='0.8',
    description="Function library for jabozzo's thesis.",

    author='Juan Andrés Bozzo',
    author_email='jabozzo@uc.cl',
    #url='https://github.com/jerrytheo/psopy',
    packages=find_packages(),

    # Classifiers list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
        'License :: OSI Approved :: BSD License',

        #'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',

        # Supported Python versions.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=[],
    install_requires=[
        'scipy',
        'numpy',
        'numexpr',
	'parsec'
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=[
        'pytest'
    ],
    scripts=[
        'calib/script/batch.py',
        'calib/script/estimate.py',
        'calib/script/join_stages.py',
        'calib/script/make_adc_list.py',
        'calib/script/make_adc.py',
        'calib/script/make_config.py',
        'calib/script/make_meta.py',
        'calib/script/make_stage_list.py',
        'calib/script/make_stage_testbench.py',
        'calib/script/mkdir.py',
        'calib/script/mv.py',
        'calib/script/plot.py',
        'calib/script/plot_snr.py',
        'calib/script/simulate.py',
        'calib/script/stack_configs.py',
        'calib/script/sweep_testbench.py'
    ]
)
