Installation
============

:code:`iconspy` can be downloaded from github. To obtain the source code run the command:

.. code-block:: bash

   git clone https://github.com/fraserwg/iconspy.git

conda/mamba (recommended)
-------------------------
To install the dependencies into an existing environment run:

.. code-block:: bash

    cd iconspy
    mamba install -f environment.yml

or to create a new environment:

.. code-block:: bash

   cd iconspy
   mamba env create -f environment.yml
   mamba activate ispy

you can then install the package using:

.. code-block:: bash

   pip install .

pip
---

.. code-block:: bash

   cd iconspy
   pip install .