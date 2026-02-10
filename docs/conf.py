# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'quadrupole'
copyright = '2026, Brock Dyer'
author = 'Brock Dyer'
release = '0.2.1'

import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
    "pydata_sphinx_theme",
    "numpydoc",
]
numpydoc_show_class_members = False
imgmath_image_format = "svg"
source_suffix = {
    ".rst": "restructuredtext",
    ".md" : "markdown",
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_theme_options = {
    "logo": {
        "image_dark": "_static/quadrupole_dark_with_text.svg",
        "image_light": "_static/quadrupole_light_with_text.svg",
    },
    "github_url": "https://github.com/brockdyer03/quadrupole"
}
