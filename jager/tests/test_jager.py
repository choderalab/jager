"""
Unit and regression test for the jager package.
"""

# Import package, test suite, and other packages as needed
import jager
import pytest
import sys

def test_jager_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "jager" in sys.modules
