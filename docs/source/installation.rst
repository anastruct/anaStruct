Installation
############

You will need Python to install anaStruct on your system. 


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
    sudo apt-get install python3

Windows
=======

On Windows (and for other OS's) you can download the installer for the version you prefer from `Python's website <https://www.python.org>`_.

Mac
=====

For Mac OS install Python 3 using homebrew

::

    brew install python

Install anaStruct
*****************

anaStruct is a lightweight package with only one mandatory prerequisite: `numpy`. If you want to use the plotting features of the package, then you will also need `matplotlib`. The installation process will automatically install these dependencies for you.

If you like to use the computational backend of the package without having the plotting features, simply run the code below in the terminal. Pip will install a headless version of anaStruct (with no plotting abilities).

::

    pip install anastruct

Otherwise you can have a full installation using the following code in your terminal.

::

    pip install anastruct[plot]

Alternatively, you can build the package from source by cloning the source from the git repository. Updates are regularly released on `PyPi <https://pypi.org/>`_, but if you'd like the bleeding edge newest features and fixes, or if you'd like to contribute to the development of `anaStruct <https://pypi.org/project/anastruct/>`_, then install from github.

::

    pip install git+https://github.com/anastruct/anaStruct.git