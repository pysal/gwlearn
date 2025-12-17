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
copyright = "2025-, Martin Fleischmann & PySAL Developers"
author = "Martin Fleischmann"

version = gwlearn.__version__.split("+", 1)[0]
release = version

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
    "sphinx_immaterial",
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

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/pysal_logo.svg"
html_favicon = "_static/pysal_favicon.ico"
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "site_url": "https://pysal.org/gwlearn",
    "repo_url": "https://github.com/pysal/gwlearn/",
    "repo_name": "pysal/gwlearn",
    "features": [
        # "navigation.expand",
        # "navigation.tabs",
        # "navigation.tabs.sticky",
        # "toc.integrate",
        # "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        "navigation.footer",
        # "navigation.tracking",
        # "search.highlight",
        # "search.share",
        # "search.suggest",
        # "toc.follow",
        # "toc.sticky",
        # "content.tabs.link",
        "content.code.copy",
        # "content.action.edit",
        # "content.action.view",
        # "content.tooltips",
        # "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "Switch to light mode",
            },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "black",
            "accent": "red",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "red",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to system preference",
            },
        },
    ],
}
nb_execution_mode = "off"
autodoc_typehints = "none"
