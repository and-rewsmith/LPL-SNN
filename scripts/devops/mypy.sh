#!/bin/bash
set -e

# Upgrade pip
python -m pip install --upgrade pip

# Install pycodestyle
pip install mypy
pip install types-toml

mypy --config-file ./mypi.ini datasets

echo "Mypy check passed successfully!"