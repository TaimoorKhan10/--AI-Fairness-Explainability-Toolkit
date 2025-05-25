"""
Initialize Sphinx documentation for the project.

This script sets up the basic Sphinx documentation structure.
"""

import os
import sys
import subprocess
from pathlib import Path

def init_docs():
    """Initialize Sphinx documentation."""
    # Create necessary directories
    docs_dir = Path("docs")
    (docs_dir / "_static").mkdir(exist_ok=True)
    (docs_dir / "_templates").mkdir(exist_ok=True)
    
    # Create basic files
    with open(docs_dir / "conf.py", "w", encoding="utf-8") as f:
        f.write('''# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AFET'
copyright = '2025, AFET Contributors'
author = 'AFET Contributors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'nbsphinx',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}
''')

    # Create index.rst
    with open(docs_dir / "index.rst", "w", encoding="utf-8") as f:
        f.write('''.. AFET documentation master file, created by
   sphinx-quickstart on Sat May 25 17:52:00 2025.

Welcome to AFET's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   api
   examples
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
''')

    # Create basic documentation pages
    for page in ["installation", "user_guide", "api", "examples", "contributing", "changelog"]:
        with open(docs_dir / f"{page}.rst", "w", encoding="utf-8") as f:
            f.write(f"{page.title()}\n" + "=" * len(page) + "\n\n" + "Coming soon...\n")

    # Create .gitkeep files in empty directories
    for dir_path in ["_static", "_templates"]:
        (docs_dir / dir_path / ".gitkeep").touch()

    # Create Makefile and make.bat for building docs
    with open(docs_dir / "Makefile", "w", encoding="utf-8") as f:
        f.write('''# Minimal makefile for Sphinx documentation
#
# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
''')

    with open(docs_dir / "make.bat", "w", encoding="utf-8") as f:
        f.write('''@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%

:end
popd
''')

    print("Documentation structure created successfully!")
    print("\nTo build the documentation, run:")
    print("  cd docs")
    print("  make html")
    print("\nThe documentation will be available in docs/_build/html/index.html")

if __name__ == "__main__":
    init_docs()
