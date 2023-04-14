import setuptools
from setuptools import setup

setup(
    name='nc',
    version='1.0',
    packages=setuptools.find_packages(),
    install_requires=[
        'transformers==4.19.0',
        'datasets',
        'accelerate',
        'sklearn'
    ]
    )