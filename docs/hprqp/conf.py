
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "HPR-QP"
html_title = "HPR-QP"
html_short_title = "HPR-QP"


copyright = '2025, HPR Methods Developer Team'
author = 'HPR  Methods Developer Team' 

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc", "sphinx.ext.autosummary",
    "sphinx.ext.napoleon", "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]
autosummary_generate = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_permalinks = False

extensions = ["myst_parser"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# -- Options for HTML output -------------------------------------------------

html_favicon = "../assets/favicon.ico"


# html_theme = "furo"
# html_theme = "sphinx_book_theme"
# html_theme = "pydata_sphinx_theme"
html_theme = "renku" #  like OSQP
#html_theme = "press"




html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

html_theme_options = { 
    "collapse_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "sticky_navigation": True,
    # "titles_only": False,
}

