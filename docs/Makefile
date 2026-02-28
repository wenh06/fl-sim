# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile
.PHONY: help Makefile gettext locale multiversion multiversion-en multiversion-zh

gettext:
	@$(SPHINXBUILD) -b gettext "$(SOURCEDIR)" "$(BUILDDIR)/gettext"

locale:
	@sphinx-intl update -p "$(BUILDDIR)/gettext" -l zh_CN

multiversion-en:
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)/multiversion/en" -D language='en'

multiversion-zh:
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)/multiversion/zh_CN" -D language='zh_CN'

multiversion: multiversion-en multiversion-zh

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
