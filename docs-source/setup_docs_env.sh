#! /bin/bash

set -e

echo "Creating Python 3 virtual environment for compiling project documentation..."
# Python 3 is needed to setup the python environment, but the module is not
# needed once it is setup
module load python/3.6.4 >/dev/null 2>&1 || \
    echo "Using system's Python 3 to setup virtual environment"

python3 -m venv pyenv-docs
source pyenv-docs/bin/activate
pip3 install --upgrade pip
pip3 install wheel
pip3 install -r docs-python-requirements.txt
