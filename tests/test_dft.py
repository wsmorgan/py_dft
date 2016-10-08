"""Tests the script accuss to the dft driver.
"""
import pytest
import os
# The virtual, pseudorandom port is setup as a session fixture in conftest.py
def get_sargs(args):
    """Returns the list of arguments parsed from sys.argv.
    """
    import sys
    sys.argv = args
    from pydft.dft import _parser_options
    return _parser_options()    

def test_examples():
    """Makes sure the script examples work properly.
    """
    argv = ["py.test", "-examples"]
    assert get_sargs(argv) is None
    
def test_run(capfd):
    """Test that a default solve works properly.
    """
    from pydft.dft import run

    args = {"N":100}
    run(args)
