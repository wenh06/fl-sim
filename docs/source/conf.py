# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
import re
import sys
from pathlib import Path

import pydata_sphinx_theme
import requests
import sphinx_book_theme
import sphinx_rtd_theme
import sphinx_theme

project_root = Path(__file__).resolve().parents[2]
src_root = project_root / "fl_sim"
docs_root = Path(__file__).resolve().parents[0]

sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "fl-sim"
copyright = "2023, WEN Hao"
author = "WEN Hao"

# The full version, including alpha/beta/rc tags
release = Path(src_root / "version.py").read_text().split("=")[1].strip()[1:-1]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    # 'sphinx.ext.autosectionlabel',
    "sphinx_multiversion",
    "sphinx_emoji_favicon",
    # "sphinx_toolbox.collapse",  # replaced by dropdown of sphinx_design
    # "numpydoc",
    "sphinxcontrib.tikz",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.proof",
]

locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False  # optional.

bibtex_bibfiles = ["references.bib"]
# bibtex_bibliography_header = ".. rubric:: 参考文献"
bibtex_bibliography_header = ".. rubric:: References"
bibtex_footbibliography_header = bibtex_bibliography_header

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "np": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pd": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    # "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch_optimizer": ("https://pytorch-optimizer.readthedocs.io/en/latest/", None),
    "torch_ecg": ("https://torch-ecg.readthedocs.io/en/latest/", None),
}

autodoc_default_options = {
    "show-inheritance": True,
}

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "wenh06",  # Username
    "github_repo": "fl-sim",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
    "current_version": release,  # Version label
    "versions": [[release, f"link to {release}"]],  # Version labels
    "current_language": "en",  # Language label
    "languages": [["en", "link to en"], ["zh", "link to zh"]],  # Language labels
}

templates_path = ["_templates"]

# html_sidebars = {"*": ["versions.html"]}

exclude_patterns = []

# Napoleon settings
napoleon_custom_sections = [
    "ABOUT",
    "ISSUES",
    "Usage",
    "Citation",
    "TODO",
    "Version history",
    "Pipeline",
]
# napoleon_custom_section_rename = False # True is default for backwards compatibility.

proof_theorem_types = {
    "algorithm": "Algorithm",
    "conjecture": "Conjecture",
    "corollary": "Corollary",
    "definition": "Definition",
    "example": "Example",
    "lemma": "Lemma",
    "observation": "Observation",
    "proof": "Proof",
    "property": "Property",
    "theorem": "Theorem",
    "remark": "Remark",  # new
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
_theme_name = "sphinx_book_theme"  # "pydata_sphinx_theme", "stanford_theme", etc.

if _theme_name == "stanford_theme":
    html_theme = "stanford_theme"
    html_theme_path = [sphinx_theme.get_html_theme_path("stanford-theme")]
    html_theme_options = {
        "collapse_navigation": False,
        "display_version": True,
    }
elif _theme_name == "sphinx_rtd_theme":
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    html_theme_options = {
        "collapse_navigation": False,
        "display_version": True,
    }
elif _theme_name == "sphinx_book_theme":
    html_theme = "sphinx_book_theme"
    html_theme_path = [sphinx_book_theme.get_html_theme_path()]
    html_theme_options = {
        "repository_url": "https://github.com/wenh06/fl-sim",
        "use_repository_button": True,
        "use_issues_button": True,
        "use_edit_page_button": True,
        "use_download_button": True,
        "use_fullscreen_button": True,
        "path_to_docs": "docs/source",
        "repository_branch": "master",
    }
elif _theme_name == "pydata_sphinx_theme":
    html_theme = "pydata_sphinx_theme"
    html_theme_path = [pydata_sphinx_theme.get_html_theme_path()]
    html_theme_options = {
        "collapse_navigation": False,
        "display_version": True,
    }
else:  # builtin themes: alabaster, etc.
    html_theme = _theme_name


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

master_doc = "index"

numfig = False


_mathjax_file = "tex-chtml-full.js"


def _get_mathjax_latest_version() -> str:
    """Get the latest mathjax version.

    Returns
    -------
    str
        The latest mathjax version.

    """
    defalut_mathjax_latest_version = "3.2.2"
    url = f"https://unpkg.com/mathjax@latest/es5/{_mathjax_file}"
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            # search for the version number in r.url
            # which will be redirected to the latest version with version number
            # e.g. https://unpkg.com/mathjax@3.2.2/es5/tex-chtml-full.js
            return re.search("mathjax@([\\w\\.\\-]+)", r.url).group(1)
        else:
            return defalut_mathjax_latest_version
    except Exception:
        return defalut_mathjax_latest_version


mathjax_path = f"https://cdnjs.cloudflare.com/ajax/libs/mathjax/{_get_mathjax_latest_version()}/es5/{_mathjax_file}"
# mathjax_path = f"https://cdn.bootcdn.net/ajax/libs/mathjax/{_get_mathjax_latest_version()}/es5/{_mathjax_file}"


emoji_favicon = ":abaque:"

linkcheck_ignore = [
    r"https://doi.org/*",  # 418 Client Error
]


def setup(app):
    app.add_css_file("css/custom.css")
    app.add_css_file("css/proof.css")
