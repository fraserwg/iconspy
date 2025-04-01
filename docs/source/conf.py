# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'iconspy'
copyright = '2025, Fraser William Goldsworth'
author = 'Fraser William Goldsworth'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Extracts documentation from docstrings
    'sphinx.ext.napoleon',      # Supports Google- and NumPy-style docstrings
    'sphinx.ext.autosummary',   # Generates summary tables for modules/classes
    'sphinx_autodoc_typehints',  # Shows type hints in documentation
    'sphinx_copybutton',
    'nbsphinx',            # For Jupyter Notebook support
]

# Set the default theme
html_theme = "sphinx_rtd_theme"

# Specify paths for autodoc
import os
import sys
sys.path.insert(0, os.path.abspath(".."))  # Adjust path to include