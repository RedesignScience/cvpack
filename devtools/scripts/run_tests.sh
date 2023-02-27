#!/usr/bin/env bash

pytest -v -s --cov=cvpack --cov-report=term-missing --cov-report=html --pyargs --doctest-modules "$@" cvpack
