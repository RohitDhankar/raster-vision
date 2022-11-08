# flake8: noqa

from pallets_sphinx_themes import ProjectLink, get_version

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'Raster Vision'
copyright = '2018, Azavea'
author = 'Azavea'

# The short X.Y version
version = '0.13'
# The full version, including alpha/beta/rc tags
release = '0.13.1'

# -- Extension configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'pallets_sphinx_themes',
    # https://www.sphinx-doc.org/en/master/tutorial/automatic-doc-generation.html
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    # support Google-style docstrings
    # https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    'sphinx.ext.napoleon',
    # mardown support
    'myst_parser',
    # allow linking to python docs; see intersphinx_mapping below
    'sphinx.ext.intersphinx',
    # better rendering of pydantic Configs
    'sphinxcontrib.autodoc_pydantic',
    # for linking to source files from docs
    'sphinx.ext.viewcode',
    # for rendering examples in docstrings
    'sphinx.ext.doctest',
    # jupyter notebooks
    'nbsphinx',
    # jupyter notebooks in a gallery
    'sphinx_gallery.load_style',
]

#########################
# autodoc, autosummary
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
#########################
# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

autosummary_generate = True
autosummary_ignore_module_all = False

autodoc_typehints = 'both'
autodoc_class_signature = 'separated'
autodoc_member_order = 'groupwise'
autodoc_mock_imports = ['torch', 'torchvision', 'pycocotools', 'geopandas']
#########################

#########################
# nbsphinx options
#########################
nbsphinx_execute = 'never'
sphinx_gallery_conf = {
    'line_numbers': True,
}
# external thumnails
nbsphinx_thumbnails = {
    # The _images dir is under build/html. This looks brittle but using the
    # more natural img/tensorboard.png path does not work.
    'tutorials/train': '_images/tensorboard.png',
}
nbsphinx_prolog = r"""
{% set docpath = env.doc2path(env.docname, base=False) %}
{% set docname = docpath.split('/')|last %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note:: This page was generated from `{{ docname }} <https://github.com/azavea/raster-vision/blob/master/docs/{{ docpath }}>`__.
"""
#########################

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# https://read-the-docs.readthedocs.io/en/latest/faq.html#i-get-import-errors-on-libraries-that-depend-on-c-modules
import sys
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = ['pyproj', 'h5py', 'osgeo']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)

# connect docs in other projects
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3",
        "https://docs.python.org/3/objects.inv",
    ),
    "rasterio": (
        "https://rasterio.readthedocs.io/en/stable/",
        "https://rasterio.readthedocs.io/en/stable/objects.inv",
    ),
    "shapely": (
        "https://shapely.readthedocs.io/en/stable/",
        "https://shapely.readthedocs.io/en/stable/objects.inv",
    ),
    "matplotlib": (
        "https://matplotlib.org/stable/",
        "https://matplotlib.org/stable/objects.inv",
    ),
    "geopandas": (
        "https://geopandas.org/en/stable/",
        "https://geopandas.org/en/stable/objects.inv",
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        "https://numpy.org/doc/stable/objects.inv",
    ),
    "pytorch": (
        "https://pytorch.org/docs/stable/",
        "https://pytorch.org/docs/stable/objects.inv",
    ),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', 'README.md', '**.ipynb_checkpoints'
]

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'sphinx'

# HTML -----------------------------------------------------------------

html_theme = 'furo'
# https://pradyunsg.me/furo/customisation/
html_theme_options = {
    'sidebar_hide_name': True,
    'top_of_page_button': None,
    'navigation_with_keys': True,
}
html_context = {
    'project_links': [
        ProjectLink('Quickstart', 'quickstart.html'),
        ProjectLink('Documentation TOC', 'index.html#documentation'),
        ProjectLink('Examples', 'examples.html'),
        ProjectLink('AWS Batch Setup', 'cloudformation.html'),
        ProjectLink('Project Website', 'https://rastervision.io/'),
        ProjectLink('PyPI releases', 'https://pypi.org/project/rastervision/'),
        ProjectLink('GitHub Repo', 'https://github.com/azavea/raster-vision'),
        ProjectLink('Discussion Forum',
                    'https://github.com/azavea/raster-vision/discussions'),
        ProjectLink('Issue Tracker',
                    'https://github.com/azavea/raster-vision/issues/'),
        ProjectLink('CHANGELOG', 'changelog.html'),
        ProjectLink('Azavea', 'https://www.azavea.com/'),
    ]
}
html_static_path = ['_static']
html_css_files = ['custom.css']
html_favicon = 'img/raster-vision-icon.png'
html_logo = 'img/raster-vision-logo.png'
html_title = f'Raster Vision Documentation ({version})'
html_show_sourcelink = False
html_domain_indices = False
html_experimental_html5_writer = True

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'RasterVisiondoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'RasterVision.tex', 'Raster Vision Documentation', 'Azavea',
     'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, 'RasterVisoin-{}.tex', html_title, [author],
              'manual')]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'RasterVision', 'Raster Vision Documentation', author,
     'RasterVision', 'One line description of project.', 'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------

programoutput_prompt_template = '> {command}\n{output}'

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
