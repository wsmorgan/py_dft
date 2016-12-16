"""Contains the code needed to solve the schrodinger equation.
"""

import numpy as np

def _sho_V(r,w=2):
    """The potential for the position x of the simple harmonic oscillator.

    Args:
        r (numpy.ndarray): The location of the desired potential.
        w (float, optional): The frequency of the simple harmonic oscillator.
    Returns:
        V (float): The potential at r.
    """

    V = 0.5*w*w*r**2

    return V

def _diagouter(A,B):
    """Finds the diagonal of the outer product of two matrices.

    Args:
        A (numpy.ndarray): A matrix.
        B (numpy.ndarray): B matrix.

    Returns:
        c (np.ndarray): The diagonal of A.dag(B).

    Raises:
        ValueError: If the two matrices don't have the same shape.
    """
    # if len(A) != len(B) or len(A[0]) != len(B[0]):
    #     print(len(A),len(A[0]),len(B),len(B[0]))
    #     raise ValueError("The two input matrices must have the same shape.")

    c = []
    for i in range(len(A)):
        c_i = 0
        for n in range(len(A[0])):
            c_i += A[i][n]*np.conj(B[i][n])
        c.append(c_i)

    c = np.array(c)
    return c

def _getE(s,R,W,V = None):
    """The sum of the energies for the states present in the solution.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (numpy.ndarray): The basis vectors for the unit cell.
        W (numpy.ndarray): A matrix containing the expansion coefficients 
          for the wavefunctions

    Returns:
        E (numpy.ndarray): A vector of the energies at the sample points.

    """

    from pydft.poisson import _O_operator, _L_operator, _B_operator

    if V == None:
        V = _sho_V

    O_t = _O_operator(s,R,W)
    U = np.dot(np.conj(W.T),_O_operator(s,R,W))
    Vt = np.transpose(np.conj(_Vdual(s,R, V = V)))

    IW = _B_operator(s,R,W)
    Uinv = np.linalg.inv(U)
    IWU = _B_operator(s,R,np.dot(W,Uinv))
    n = _diagouter(IW,IWU)

    Ew = np.trace(np.dot(np.conj(np.transpose(W)),_L_operator(s,R,np.dot(W,Uinv))))

    E = (-1.)*Ew/2. + np.dot(Vt,n)
    
    return E

def _Vdual(s,R,V = None):
    """Finds the dual of the potential matrix.
    
    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (numpy.ndarray): The basis vectors for the unit cell.
        V (function, optional): The function that defines the potential.
    
    Returns:
        Vdual (numpy.ndarray): The dual of the potential matrix.
    """
    
    from pydft.poisson import _Bj_dag_operator, _O_operator, _Bj_operator, _generate_r, _find_dr

    if V == None:
        V = _sho_V

    r = _generate_r(R,s)
    dr = _find_dr(r,R)
    Vs = V(dr)
    Vdual = _Bj_dag_operator(s,R,_O_operator(s,R,_Bj_operator(s,R,Vs)))

    return Vdual

def _diagprod(a,B):
    """Finds the diagonal product of the vector a with the matrix B.

    Args:
        a (numpy.ndarray): The vector a of the diagonal product.
        B (numpy.ndarray): The matrix B of the diagonal product.

    Retruns:
        dp (numpy.ndarray): The matrix output of the diagonal product.

    Raises:
        ValueError: If the vector has more entries than there are rows
          in the matrix.
    """

    if len(a) != len(B):
        raise ValueError("The vector `a` must have the same number of entries "
                         "as there are rows in the matrix `B`." )
    dp = []

    for i in range(len(a)):
        dp.append(a[i]*np.array(B[i]))

    dp = np.array(dp)

    return dp

def _H(s,R,W,V = None):
    """Constructs the hamiltonian matrix HW.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (numpy.ndarray): The basis vectors for the unit cell.
        W (numpy.ndarray): A matrix containing the expansion coefficients 
          for the wavefunctions
        V (function, optional): The function of the potential.

    Returns:
        HW (numpy.ndarray): The hamiltonian matrix.
    """

    from pydft.poisson import _B_operator, _B_dag_operator, _L_operator
    
    if V == None:
        V = _sho_V

    LW = _L_operator(s,R,W)
    Vd = _Vdual(s,R,V = V)

    IW = _B_dag_operator(s,R,_diagprod(Vd,_B_operator(s,R,W)))

    HW = -LW/2. + IW
    
    return HW

def _getgrad(s,R,W,V = None):
    """Finds the gradient of the input matrix.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (numpy.ndarray): The basis vectors for the unit cell.
        W (numpy.ndarray): A matrix containing the expansion coefficients 
          for the wavefunctions
        V (function, optional): The function of the potential.

    Returns:
       gradW (numpy.ndarray): The gradient of the matrix.
    """
    from pydft.poisson import _O_operator

    if V == None:
        V = _sho_V

    HW = _H(s,R,W,V=V)
    Uinv = np.linalg.inv(np.dot(np.conj(np.transpose(W)),_O_operator(s,R,W)))
    rest = _O_operator(s,R,np.dot(W,np.dot(Uinv,np.dot(np.conj(W.T),HW))))
    diff = HW - rest

    gradW = np.dot(diff,Uinv)

    return np.array(gradW)

def schrodinger(R,s,V = None):
    """Solves the schrodinger equation for the given potential.

    Args:
        R (numpy.ndarray): The basis vectors for the unit cell.
        s (list of int): The number of samples points along each 
          basis vector.
        V (function, optional): The function that defines the potential.

    Returns:
    
    """

    

    

    
