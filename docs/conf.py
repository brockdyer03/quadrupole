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
    #"sphinx.ext.autodoc",
    #"sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "pydata_sphinx_theme",
    "sphinx_design",
    "numpydoc",
]

numpydoc_class_members_toctree = False
numpydoc_show_class_members = True

source_suffix = {
    ".rst": "restructuredtext"
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = ["quadrupole.css"]
html_js_files = [
    ("custom-icons.js", {"defer": "defer"}),
]
html_theme_options = {
    "logo": {
        "alt_text": "Quadrupole Package - Home",
        "image_dark": "_static/quadrupole_dark_with_text.svg",
        "image_light": "_static/quadrupole_light_with_text.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/brockdyer03/quadrupole",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/quadrupole/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "navbar-icon-links"
    ],
    "show_toc_level": 3,
    "secondary_sidebar_items": ["page-toc"],
}
html_favicon = "_static/quadrupole_dark_favicon.svg"
html_sidebars = {
    "user_guide": [],
}

html_context = {"default_mode": "auto"}



