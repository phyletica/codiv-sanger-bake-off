#! /bin/bash

set -e

echo "Creating Python 2 virtual environment for running PyMsBayes..."
# Python 2 is needed to setup the python environment for running PyMsBayes, but
# the module is not needed once it is setup
module load python/2.7.15 >/dev/null 2>&1 || \
    echo "Using system's Python 2 to setup virtual environment"

pip2 install virtualenv
if [ -e "${HOME}/.local/bin/virtualenv" ]
then
    export PATH="${PATH}:${HOME}/.local/bin"
fi
virtualenv pyenv-abc
source pyenv-abc/bin/activate
pip2 install --upgrade pip
pip2 install wheel
pip2 install -r abc-python-requirements.txt
deactivate

module unload python/2.7.15 >/dev/null 2>&1
