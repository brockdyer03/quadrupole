# Quadrupole Analysis

Python functions for taking molecular quadrupole tensors and converting to forms for comparison to literature.

Includes rotating quadrupole into the molecular inertial frame, detracing the quadrupole tensor, and providing tools to empirically compare quadrupoles to literature values.

## Project Goals

This project is, in some ways, supposed to implement my vision of a "perfect" Python package.
Some of the most important tenets of which are described below.

### Support Policy

With modern Python package managers (e.g. `uv`), it is generally trivial to stay consistently up-to-date with Python versions, and this package will do just that.
This means it will take full advantage of the absolute bleeding edge features in Python, even if that leads to only supporting the latest Python version(s).

### Documentation

This project aims to provide not just comprehensive API docs, but also a thorough User Guide and in-depth theoretical background for some of the theory involved in this package.
It is absolutely required to provide documentation for any non-obvious user-facing functions (non-obvious meaning any function whose name and type hints do not make its behavior obvious).
Additionally, we strive to provide citations for every piece of data used, and for any scientific literature referenced in the docs.
Our docstrings follow the [Numpy docstring](https://numpydoc.readthedocs.io/en/latest/) convention, our documentation is written in [reStructuredText](https://docutils.sourceforge.io/rst.html), and is built using [Sphinx](https://www.sphinx-doc.org/en/master/).

### Test Coverage

Another important tenet of this package is maintaining 100% test coverage.
This helps ensure that changes to the code do not unknowingly break features that already exist, and similarly ensures graceful error handling.
All of our coverage information can be found on [CodeCov](https://app.codecov.io/gh/brockdyer03/quadrupole).
