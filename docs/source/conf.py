# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from pathlib import Path

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
    "sphinx_emoji_favicon",
    "sphinxcontrib.tikz",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.proof",
    "sphinxcontrib.pseudocode2",
]

language = os.environ.get("SPHINX_LANGUAGE", "en")
locale_dirs = ["locale/"]  # path is example but recommended.
gettext_compact = False
gettext_auto_build = True

bibtex_bibfiles = ["references.bib"]
# bibtex_bibliography_header = ".. rubric:: 参考文献"
bibtex_bibliography_header = ".. rubric:: References"
bibtex_footbibliography_header = bibtex_bibliography_header

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    # "pd": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    # "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "torch_optimizer": ("https://pytorch-optimizer.readthedocs.io/en/latest/", None),
    "torch_ecg": ("https://torch-ecg.readthedocs.io/en/latest/", None),
}

autodoc_default_options = {
    "show-inheritance": True,
}

# -- Multi-version / multi-language switcher context ---------------------------
# Single-language builds (e.g. `make html-en`) show only the language switcher
# with relative links between the sibling build directories.
# `docs/build_docs.py` sets PAGES_ROOT / CURRENT_VERSION / BUILD_ALL_DOCS for
# full builds; then the switcher links across all versions and languages with
# absolute URLs under PAGES_ROOT (the GitHub Pages site root).

_pages_root = os.environ.get("PAGES_ROOT", "")
_current_version = os.environ.get("CURRENT_VERSION", "latest")
_build_all_docs = os.environ.get("BUILD_ALL_DOCS", "") == "1" and bool(_pages_root)

_language_labels = [
    ("English", "en"),
    ("简体中文", "zh_CN"),
]

if _build_all_docs:
    import yaml

    _versions_file = Path(docs_root).parent / "versions.yaml"
    _versions_cfg = {}
    if _versions_file.exists():
        _versions_cfg = yaml.safe_load(_versions_file.read_text()) or {}
    # version names double as URL path segments; "latest" tracks the master branch
    _all_versions = ["latest"] + [str(v) for v in _versions_cfg]

    def _pages_url(version: str, lang_code: str) -> str:
        return f"{_pages_root.rstrip('/')}/{version}/{lang_code}/"

    _languages = [(label, code, _pages_url(_current_version, code)) for label, code in _language_labels]
    _versions = [(v, _pages_url(v, language)) for v in _all_versions]
else:
    _languages = [(label, code, "") for label, code in _language_labels]
    _versions = []

html_context = {
    "display_github": True,
    "github_user": "wenh06",
    "github_repo": "fl-sim",
    # point the "edit on GitHub" button at the branch/tag being built
    "github_version": "master" if _current_version == "latest" else _current_version,
    "conf_py_path": "/docs/source/",
    "current_language": language,
    "current_version": _current_version,
    "languages": _languages,
    "versions": _versions,
    "build_all_docs": _build_all_docs,
}

templates_path = ["_templates"]

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
html_theme = "sphinx_book_theme"
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


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

master_doc = "index"

# numfig = False

pseudocode2_math_engine = "katex"

pseudocode2_options = {
    "lineNumber": True,  # Global default: enable line numbering
    "commentDelimiter": " //",  # Global default comment delimiter
    "noEnd": False,  # Global default: show "END" for control blocks
    "scopeLines": True,  # Global default: enable scope lines
}


_mathjax_file = "tex-chtml-full.js"
# pinned to avoid a build-time network probe; override with `make MATHJAX_VERSION=...`
# or the MATHJAX_VERSION environment variable if a newer version is wanted
_mathjax_version = os.environ.get("MATHJAX_VERSION", "3.2.2")
mathjax_path = f"https://cdnjs.cloudflare.com/ajax/libs/mathjax/{_mathjax_version}/es5/{_mathjax_file}"


emoji_favicon = ":abaque:"

linkcheck_ignore = [
    r"https://doi.org/*",  # 418 Client Error
]


def setup(app):
    app.add_css_file("css/custom.css")
    app.add_css_file("css/proof.css")
    app.add_css_file("css/codeblock.css")
    app.add_css_file("css/versions.css")
    app.add_js_file("js/codeblock.js")
    app.add_js_file("js/versions.js")
