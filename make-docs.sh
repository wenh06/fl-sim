#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR="$SCRIPT_DIR/docs"
cd $SCRIPT_DIR

# full rebuild; build_docs.py refreshes docs/pages and recreates the
# docs/build compatibility layer (served via symlink) afterwards
make clean
python3 build_docs.py --latest-only
