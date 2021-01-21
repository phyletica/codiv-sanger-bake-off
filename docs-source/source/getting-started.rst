.. _getting-started:

###############
Getting started
###############

If you haven't already, you should familiarize yourself with the
:ref:`prerequisites` and :ref:`background` sections of documentation.
If you are a |phyleticalab|_ member, make sure you have done everything in the
:ref:`setting-up` section. 


.. _clone-project:

Cloning the project repo
========================

Open up a terminal and navigate to where you want to work on this project.
For |phyleticalab|_, this means logging in to Hopper::

    ssh YOUR-AU-USERNAME@hopper.auburn.edu

or::

    ssh hopper

and navigating to your home directory (or any directory where you want to keep
this project)::

    cd ~

Now, clone the |git|_ repository for this project::

    git clone git@github.com:phyletica/codiv-sanger-bake-off.git

If this command is successful, when you::

    ls
    
you should see a directory called ``codiv-sanger-bake-off`` listed.
Go ahead and ``cd`` into this directory::

    cd codiv-sanger-bake-off

To get oriented to the contents of the project repository, please checkout the
:ref:`tour` section of the documentation.


.. _setup-project:

Setting up the project
======================

.. note:: Run the following command to make sure you don't have an active
    ``conda`` environment::

        conda deactivate

The first thing we need to do is run the ``setup_project_env.sh`` Bash script
which is located at the base of the project directory::

    bash setup_project_env.sh

.. note:: You may get an error message during the setup process that looks
    something like::
    
        ERROR: pycoevolity 0.2.6 requires munkres<=1.0.12, but you'll have munkres 1.1.4 which is incompatible.

    This is nothing to worry about. Pycoevolity requires different versions of
    the munkres package depending on the version of Python, which causes this
    error message about not having the version of munkres that's needed by a
    different version of Python.

This script will:

1.  Create a Python virtual environment for the project.
2.  Download and build a specific version of |eco|_ and will install all of the
    |eco|_ tools in the ``bin`` directory within our project directory.
        
Let's check to make sure the setup script did what it was supposed to do.
If the Python virtual environment was setup correctly, we should be
able to activate it::

    source pyenv/bin/activate

and then verify the |pyco|_ tools were installed in the environment by typing::

    pyco-sumtimes -h

which should display the help menu of the ``pyco-sumtimes`` tool, the beginning of which should look something like::

    ========================================================================
                                  Pycoevolity                               
                       Summarizing evolutionary coevality                   
    
            A Python package for analyzing the output of Ecoevolity         
    
               Version 0.2.6 (main 738ab1e: 2020-10-12 16:00:23)            
    
                 License: GNU General Public License Version 3              
    ========================================================================
    usage: pyco-sumtimes [-h] [-b BURNIN] [-p PREFIX] [-f]
                         [-l COMPARISON-LABEL REPLACEMENT-LABEL]
                         [-i COMPARISON-LABEL] [--violin] [--include-map-model]
                         [-z] [--x-limits LOWER-LIMIT UPPER-LIMIT] [-x X_LABEL]
                         [-y Y_LABEL] [-w WIDTH] [--base-font-size BASE_FONT_SIZE]
                         [--colors [COLORS [COLORS ...]]] [--no-plot]
                         ECOEVOLITY-STATE-LOG-PATH [ECOEVOLITY-STATE-LOG-PATH ...]

If |eco|_ was successfully installed by the setup script, you should be able to
call up the help menu of ``ecoevolity`` by entering (from the base directory of
the project)::

    bin/ecoevolity -h

This should display the help menu that begins with something like::

    ======================================================================
                                  Ecoevolity
                      Estimating evolutionary coevality
          Version 0.3.2 (testing c128046: 2020-09-15T22:35:02-05:00)
    ======================================================================
    
    Usage: ecoevolity [OPTIONS] YAML-CONFIG-FILE
    
    Ecoevolity: Estimating evolutionary coevality
    
    Options:
      --version             show program's version number and exit
      -h, --help            show this help message and exit
    
Once everything is setup, we have no need for the ``ecoevolity`` directory that
was cloned by the ``setup_project_env.sh`` script, so you can go ahead and
remove it::

    rm -rf ../codiv-sanger-bake-off/ecoevolity

.. note:: The extra long path in the above command is to help ensure you are
    where you think you are on the file system and don't blow away anything you
    didn't intend to.

Congrats! You are all set and ready to begin working on the project.

Why all the trouble?
--------------------

If the setup process seemed onerous, you might be wondering, "why all the
trouble?" Well, the goal is to maximize transparency and reproducibility.
Everyone reaching this point should have the *exact* same version of |eco|_
installed to simulate datasets and analyze them, and a very similar Python
environment for running the ancillary scripts to parse, summarize, and plot the
results of these analyses.
This helps ensure that all of the details of the project are open, clear, and
can be repeated.

Next, let's go to the :ref:`sim-analyses` section to get started
with analyses!
