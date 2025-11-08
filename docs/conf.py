# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "py123d"
copyright = "2025, 123D Contributors"
author = "123D Contributors"
release = "v0.0.7"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoclasstoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "myst_parser",
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {}

autodoc_typehints = "both"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "special-members": False,
    "private-members": True,
    "inherited-members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "exclude-members": "__post_init__, __new__, __weakref__, __iter__,  __hash__, annotations, _array",
    "imported-members": True,
}

autosummary_generate = True

autoclasstoc_sections = [
    "public-attrs",
    "public-methods-without-dunders",
    "private-methods",
]
html_css_files = ["css/theme_overrides.css", "css/version_switch.css"]
html_js_files = ["js/version_switch.js"]


# Custom CSS for color theming
html_css_files = [
    "custom.css",
]

# Additional theme options for color customization
html_theme_options.update(
    {
        "light_logo": "123D_logo_transparent_black.svg",
        "dark_logo": "123D_logo_transparent_white.svg",
        "sidebar_hide_name": True,
    }
)

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}

# This CSS should go in /home/daniel/py123d_workspace/py123d/docs/_static/custom.css
# Your conf.py already references it in html_css_files = ["custom.css"]

# If you want to add custom CSS via configuration, you can use:
html_css_files = [
    "custom.css",
]
