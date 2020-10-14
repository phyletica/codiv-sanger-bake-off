#! /bin/bash

set -e

pymsbayes_commit="54f720df"

echo "Creating Python 2 virtual environment for running PyMsBayes..."
# Python 2 is needed to setup the python environment for running PyMsBayes, but
# the module is not needed once it is setup
module load python/2.7.15 >/dev/null 2>&1 || \
    echo "Using system's Python 2 to setup virtual environment"

venv_wrapper_path="$(which virtualenvwrapper.sh)"
if [ -z "$venv_wrapper_path" ]
then
    echo "ERROR: Cannot find virtualenvwrapper."
    echo "       Setting up the environment for PyMsBayes requires "
    echo "       virtualenvwrapper to be installed."
fi
source "$venv_wrapper_path"
venv_name="codiv-bakeoff"

mkvenv_return="$(mkvirtualenv -p "python2" "$venv_name")"

source "${WORKON_HOME}/${venv_name}/bin/activate"

echo "Installing PyMsBayes..."
git clone https://github.com/joaks1/PyMsBayes.git
(
    cd PyMsBayes
    git checkout -b testing "$pymsbayes_commit"
    python setup.py install
    echo "    Commit $pymsbayes_commit of PyMsBayes successfully installed"
    cd ..
)

deactivate

module unload python/2.7.15 >/dev/null 2>&1
