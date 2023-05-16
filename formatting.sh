#!/bin/sh
black . --extend-exclude .ipynb -v --exclude="/build|dist/"
flake8 . --count --ignore="E501 W503 E203 F841 E402" --show-source --statistics --exclude=./.*,build,dist
