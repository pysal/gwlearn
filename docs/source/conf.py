# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

import sphinx_autosummary_accessors

sys.path.insert(0, os.path.abspath("../gwlearn/"))

import gwlearn  # noqa

project = "gwlearn"
copyright = "2025-, gwlearn Developers"
author = "Martin Fleischmann"

version = gwlearn.__version__
release = gwlearn.__version__

language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_autosummary_accessors",
    "sphinx_copybutton",
]

bibtex_bibfiles = ["_static/references.bib"]

master_doc = "index"

templates_path = [
    "_templates",
    sphinx_autosummary_accessors.templates_path,
]
exclude_patterns = []

intersphinx_mapping = {
    "geopandas": ("https://geopandas.org/en/latest", None),
    "libpysal": (
        "https://pysal.org/libpysal/",
        "https://pysal.org/libpysal//objects.inv",
    ),
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_use_plots = True
class_members_toctree = True
numpydoc_show_inherited_class_members = True
numpydoc_xref_param_type = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
}
plot_include_source = True

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# html_logo = "_static/logo.svg"
# html_favicon = "_static/icon.png"
html_theme_options = {
    "use_sidenotes": True,
}
nb_execution_mode = "off"
autodoc_typehints = "none"
