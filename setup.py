import setuptools
from setuptools import setup

setup(
    name='nc',
    version='1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch==1.11.0',
        'sklearn',
        'transformers==4.19.0',
        'datasets',
        'accelerate'
    ]
    )