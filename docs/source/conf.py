from anastruct._version import __version__

# -- Project information
project = "anaStruct"
copyright = "2018, Ritchie Vink"
author = "Ritchie Vink"
maintainer = "Brooks Smith"

release = __version__
version = __version__

# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]
autoapi_dirs = ["../../anastruct"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

master_doc = "index"

pygments_style = "sphinx"

# -- Options for HTML output
htmlhelp_basename = "anaStructdoc"


html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
