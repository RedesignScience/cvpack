#!/bin/env bash
set -e
isort ./cvlib
black --line-length 100 ./cvlib
flake8 --ignore=E203,W503 ./cvlib
pylint --rcfile=devtools/linters/pylintrc ./cvlib
