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
import sys
from pathlib import Path

import sphinx_rtd_theme
import sphinx_theme
import sphinx_book_theme
import pydata_sphinx_theme

import recommonmark  # noqa: F401
from recommonmark.transform import AutoStructify


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
    "recommonmark",
    # 'sphinx.ext.autosectionlabel',
    "sphinx_multiversion",
    # "sphinx_toolbox.collapse",  # replaced by dropdown of sphinx_design
    # "numpydoc",
    "sphinxcontrib.tikz",
    "sphinxcontrib.pseudocode",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.proof",
]

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
    "torch_ecg": ("https://deeppsp.github.io/torch_ecg/", None),
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
else:
    raise ValueError(f"Unknown theme name: {_theme_name}")
# htmlhelp_basename = "Recommonmarkdoc"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

master_doc = "index"

numfig = False

linkcheck_ignore = [
    r"https://doi.org/*",  # 418 Client Error
]


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            "url_resolver": lambda url: github_doc_root + url,  # noqa: F821
            "auto_toc_tree_section": "Contents",
        },
        True,
    )
    app.add_transform(AutoStructify)
    app.add_css_file("css/custom.css")
    app.add_css_file("css/proof.css")
