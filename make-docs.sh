#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR="$SCRIPT_DIR/docs"
cd $SCRIPT_DIR

make clean
python pre_build.py
make html
