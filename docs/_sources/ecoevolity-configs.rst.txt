.. _eco-configs:

######################
The ecoevolity configs
######################

.. note:: For a detailed treatment of |eco|_ config files, please see
    `the config-file section of the ecoevolity documentation <https://phyletica.org/ecoevolity/ecoevolity/yaml-config.html>`_.

The |eco|_ configuration files in the ``configs`` directory specify:

1.  The model we will be using to simulate and analyze sequence datasets.
2.  The size of the datsets we will simulate and analyze (i.e., the number of
    pairs of populations, the number of individuals sampled from each
    population, and the number of sites sequenced from each individual).

For ALL of the datasets we simulate, there will be:

1.  5 pairs of populations.
2.  5 diploid individuals (i.e., 10 gene copies) sampled from each population.
3.  Each locus will comprise 500 linked sites.

We will vary the number of 500-site loci, and this can be deduced from the name
of each config file:

====================    ===============
End of file name        # of 500bp loci
====================    ===============
``-sites-00500.yml``    1
``-sites-01000.yml``    2
``-sites-02500.yml``    5
``-sites-10000.yml``    20
====================    ===============

Also, we will simulate data where:

- All 5 pairs diverge simultaneously (1 shared divergence time)

  -   All config files that begin with ``fixed-simultaneous-``
  -   Used to simulate data only

- All 5 pairs diverge independently (5 divergence times)

  -   All config files that begin with ``fixed-independent-``
  -   Used to simulate data only

- The number of divergence times, and the assignment of our 5 pairs is free
  to vary according to a Dirichlet-process model

  -   All config files that beging with ``pairs-05-``
  -   Used to simulate and analyze data
