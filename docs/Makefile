# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html-en html-zh html-all gettext locale-update Makefile

# Build English docs
html-en:
	SPHINX_LANGUAGE=en $(SPHINXBUILD) -b html -D language=en \
		"$(SOURCEDIR)" "$(BUILDDIR)/en" $(SPHINXOPTS) $(O)

# Build Chinese (Simplified) docs
html-zh:
	SPHINX_LANGUAGE=zh_CN $(SPHINXBUILD) -b html -D language=zh_CN \
		"$(SOURCEDIR)" "$(BUILDDIR)/zh_CN" $(SPHINXOPTS) $(O)

# Build both languages and add a root redirect
html-all: html-en html-zh
	@echo '<meta http-equiv="refresh" content="0; url=en/index.html">' > "$(BUILDDIR)/index.html"

# Extract translatable strings into .pot files
gettext:
	$(SPHINXBUILD) -b gettext "$(SOURCEDIR)" "$(BUILDDIR)/gettext" $(SPHINXOPTS) $(O)

# Update zh_CN .po files from freshly extracted .pot files
locale-update: gettext
	sphinx-intl update -p "$(BUILDDIR)/gettext" -l zh_CN

# Catch-all target: route all unknown targets to Sphinx.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
