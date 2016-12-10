"""Tetst the evaluation and of the ewald nuclei interaction energy.
"""
import pytest
import numpy as np
import sys

def test_ewald():

    from pydft.ewald import ewald_energy_exact
    R = np.array([[6,0,0],[0,6,0],[0,0,6]])
    xs = np.array([[0,0,0],[1.75,0,0]])
    zs = [1,1]
    Rc = 3.16

    assert np.allclose(ewald_energy_exact(zs,xs,R,Rc=Rc),-0.33009112161146059)
    assert np.allclose(ewald_energy_exact(zs,xs,R),0.024100984308366741)
