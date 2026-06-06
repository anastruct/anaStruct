Installation
############

You will need Python and a few packages as pre-requisites of the anaStruct on your system.


Install the Python
******************

Linux
=====

Python is normally delivered on any Linux distribution. So you basically just need to call the python keyword which is stored on your operating system's path. To call version 3 of python on Linux you can use `python3` in the terminal. You can check installation status and version of the python on your system.

::

    python3 --version

In case you are missing the python on your system, you can install it from the repositories of your system. For instance, on Ubuntu, you can easily install python 3.9 with the following commands:

::

    sudo apt-get update
    sudo apt-get install python3.9

Windows
=======

On windows (and for other OS's too) you can download the installation source of the version you prefer from the `Python's website <https://www.python.org>`_. You can choose between the various versions and cpu architectures.

Mac
=====

For Mac OS install Python 3 using homebrew

::

    brew install python

Install the prerequisites
*************************

You will need the `NumPy <https://numpy.org/>`_ and `SciPy <https://scipy.org/>`_. packages to be able to use the anaStruct package. However, if you are using the pip to install the package, it will take care of all dependencies and their versions.

Install the anaStruct
*********************

You can install anaStruct with pip! If you like to use the computational backend of the package without having the plotting features, simply run the code below in the terminal. Pip will install a headless version of anaStruct (with no plotting abilities).

::

    python -m pip install anastruct

Otherwise you can have a full installation using the following code in your terminal.

::

    python -m pip install anastruct[plot]

In case you need a specific version of the package, that's possible too. Simple declare the version condition over the code in terminal.

::

    python -m pip install anastruct==1.4.1

Alternatively, you can build the package from the source by cloning the source from the git repository. Updates are made regularly released on `PyPi <https://pypi.org/>`_, and if you'd like the bleeding edge newest features and fixes, or if you'd like to contribute to the development of `anaStruct <https://pypi.org/project/anastruct/>`_, then install from github.

::

    pip install git+https://github.com/ritchie46/anaStruct.git