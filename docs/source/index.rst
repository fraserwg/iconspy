.. iconspy documentation master file, created by
   sphinx-quickstart on Mon Mar 31 16:46:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

iconspy
=======

ICON Section in PYthon
----------------------
:code:`iconspy` is a python package for constructing sections on the ICON model's native grid.
It offers the ability to create sections that approximate great circles, follow lines of constant latitude or longitude,
and follow contours of an arbitrary field.

The functionality builds upon that provided  by the :code:`pyicon` package;
however, :code:`iconspy` offers more flexibility ———
in particular the ability to join together multiple sections and construct sections which follow contours.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api