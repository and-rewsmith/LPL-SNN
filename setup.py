from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='LPL-SNN',
    version='0.1.0',
    packages=find_packages(),
    install_requires=required,
)
