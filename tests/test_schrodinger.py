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

    temp = list(range(-10,2))
    temp.reverse()
    for delta in temp:
        eps = 10**delta
        dE = 2*np.real(np.trace(np.dot(np.conj(g0.T),eps*dW)))

        diff = (_getE(s,R,W + eps*dW) -E0)/dE
        estimate = np.sqrt(len(W))*eps/abs(dE)
        print("eps",eps,"diff",diff,"error",estimate)

    assert np.allclose(diff,1,atol=1e-3)

def test_Y():
    """Tests the normalization of the coefficient matrix.
    """

    from pydft.schrodinger import _Y
    from pydft.poisson import _O_operator
    from numpy.matlib import randn

    s = [6,6,4]
    R = np.array([[6,0,0],[0,6,0],[0,0,6]])
    np.random.seed(20)
    W = np.array(randn(np.prod(s), 4) + 1j*randn(np.prod(s), 4))
    W = _Y(s,R,W)

    out = np.dot(np.conj(W.T),_O_operator(s,R,W))

    assert True

def test_sd():
    """Test of the steepest decent algorithm.
    """

    from pydft.schrodinger import _sd, _Y
    from numpy.matlib import randn

    s = [6,6,4]
    R = np.array([[6,0,0],[0,6,0],[0,0,6]])
    np.random.seed(20)
    W = np.array(randn(np.prod(s), 4) + 1j*randn(np.prod(s), 4))
    W = _Y(s,R,W)

    (out, Eout) = _sd(s,R,W,Nit=275,print_test=False)

    assert np.allclose(18.9, Eout, atol=.1)


def test_H():
    """Tests the Hamiltonian.
    """

    from pydft.schrodinger import _H
    from numpy.matlib import randn
    
    s = [6,6,4]
    R = np.array([[6,0,0],[0,6,0],[0,0,6]])
    a = np.array(randn(np.prod(s), 1) + 1j*randn(np.prod(s), 1))
    b = np.array(randn(np.prod(s), 1) + 1j*randn(np.prod(s), 1))

    out1 = np.conj(np.dot(np.conj(a.T),_H(s,R,b)))
    out2 = np.dot(np.conj(b.T),_H(s,R,a))

    assert np.allclose(out1,out2)
    
# def test_getPsi():
#     """Tests the getPsi subroutine of code.
#     """

#     from pydft.schrodinger import _getPsi, _H
#     from pydft.poisson import _O_operator
#     from numpy.matlib import randn
    
#     s = [6,6,4]
#     R = np.array([[6,0,0],[0,6,0],[0,0,6]])
#     np.random.seed(20)
#     W = np.array(randn(np.prod(s), 4) + 1j*randn(np.prod(s), 4))

#     (Psi, epsilon) = _getPsi(s,R,W)

#     print(Psi.shape)

#     temp_I = np.dot(np.conj(Psi.T),_O_operator(s,R,Psi))
#     print(temp_I)
#     temp_H = np.dot(np.conj(Psi.T),_H(s,R,Psi))
#     print(temp_H)
#     print(epsilon)

#     assert False
    
