"""
Unit and regression test for the cvlib package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import cvlib


def test_cvlib_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "cvlib" in sys.modules
