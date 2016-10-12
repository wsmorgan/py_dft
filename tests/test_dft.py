"""Tests the script accuss to the dft driver.
"""
import pytest
import os
import numpy as np

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

    args = {"a":1.00,"poisson":None,"s":[3,3,3],"crystal":"sc"}
    run(args)

    args = {"a":3.00,"poisson":True,"s":[10,10,10],"crystal":"sc"}
    run(args)
    model = open("tests/test_output/potential_10_10_10.csv","r")
    temp = model.read()
    temp_f = [eval(i) for i in temp.strip().split()]
    temp2 = open("potential.csv","r").read()
    temp_f2 = [eval(i) for i in temp2.strip().split()]
    model.close()
    assert np.allclose(temp_f,temp_f2)
    os.system("rm potential.csv")
    

    with pytest.raises(ValueError):
        args = {"a":3.00,"poisson":True,"s":['3.5','10','10'],"crystal":"sc"}
        run(args)
    
    with pytest.raises(ValueError):
        args = {"a":1.00,"poisson":None,"s":[3,3,3],"crystal":"tet"}
        run(args)
