"""Tests the Solutions for the routines used to solve the schrodinger equation.
"""

import pytest
import numpy as np
import sys

def test_diagouter():
    """Tests that the diagouter subroutine works.
    """
    from pydft.schrodinger import _diagouter

    A = np.random.normal(0,5,(10,3)) + np.random.normal(0,5,(10,3))*1j
    B = np.random.normal(0,5,(10,3)) + np.random.normal(0,5,(10,3))*1j
    out = np.dot(A,np.conj(B.T))
    assert np.allclose(_diagouter(A,B),np.diag(out))

def test_diagprod():
    """Tests that the diagonal product works.
    """

    from pydft.schrodinger import _diagprod

    a = np.random.normal(0,5,(10))
    B = np.random.normal(0,5,(10,3))

    out = np.dot(np.diag(a),B)

    assert np.allclose(_diagprod(a,B),out)
    
def test_getE():
    """Tests the getE subroutine.
    """

    from pydft.schrodinger import _getE
    from numpy.matlib import randn

    s = [6,6,4]
    R = np.array([[6,0,0],[0,6,0],[0,0,6]])
    W = np.array(randn(np.prod(s), 4) + 1j*randn(np.prod(s), 4))

    out = _getE(s,R,W)

    assert np.allclose(out.imag,0)


def test_getgrad():
    """Tests the getgrad function.
    """

    from pydft.schrodinger import _getgrad, _getE

    s = [3,3,3]
    R = np.array([[6,0,0],[0,6,0],[0,0,6]])
    Ns = 4
    np.random.seed(2004)

    W = np.random.normal(0,5,(27,4)) + np.random.normal(0,5,(27,4))*1j

    E0 = _getE(s,R,W)
    g0 = _getgrad(s,R,W)

    dW = np.random.normal(0,5,(27,4)) + np.random.normal(0,5,(27,4))*1j

    temp = range(-10,2)
    temp.reverse()
    for delta in temp:
        eps = 10**delta
        dE = 2*np.real(np.trace(np.dot(np.conj(g0.T),eps*dW)))

        diff = (_getE(s,R,W + eps*dW) -E0)/dE
        estimate = np.sqrt(len(W))*eps/abs(dE)
        print("eps",eps,"diff",diff,"error",estimate)

    assert np.allclose(diff,1,atol=1e-3)
