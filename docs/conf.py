import os
import sys

sys.path.append(os.path.abspath(".."))

import sphinx_rtd_theme

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
]


# Always includes todos
todo_include_todos = True

# Generates auto-summary automatically
autosummary_generate = True

# Create numbers on figures with captions
numfig = True

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = u"leaf_cc"

exclude_patterns = ["README.rst"]
autodoc_mock_imports = ["pytest"]

pygments_style = "sphinx"

project_variable = project.replace(".", "_")
short_description = u"Plant Species Classification based on Leaves Shape Features"

# -- Options for HTML output ---------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
htmlhelp_basename = project_variable + u"_doc"

# -- Post configuration --------------------------------------------------------

# Default processing flags for sphinx
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_default_flags = [
    "members",
    "undoc-members",
    "show-inheritance",
]
