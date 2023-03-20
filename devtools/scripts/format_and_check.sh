#!/usr/bin/env bash
set -e -v
isort ./cvpack
black --line-length 100 ./cvpack
flake8 --ignore=E203,W503 ./cvpack
pylint --rcfile=devtools/linters/pylintrc ./cvpack
