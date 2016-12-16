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
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        print(len(A),len(A[0]),len(B),len(B[0]))
        raise ValueError("The two input matrices must have the same shape.")

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

    if V == None: #pragma: no cover
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

    if V == None: #pragma: no cover
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
    
    if V == None: #pragma: no cover
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

    if V == None: #pragma: no cover
        V = _sho_V

    HW = _H(s,R,W,V=V)
    Uinv = np.linalg.inv(np.dot(np.conj(np.transpose(W)),_O_operator(s,R,W)))
    rest = _O_operator(s,R,np.dot(W,np.dot(Uinv,np.dot(np.conj(W.T),HW))))
    diff = HW - rest

    gradW = np.dot(diff,Uinv)

    return np.array(gradW)

def _Y(s,R,W):
    """Normalizes the input matrix.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (numpy.ndarray): The basis vectors for the unit cell.
        W (numpy.ndarray): A matrix containing the expansion coefficients 
          for the wavefunctions

    Returns:
        Y (numpy.ndarray): The normalized expansion coefficients.
    """
    from pydft.poisson import _O_operator

    Uinv = np.sqrt(np.linalg.inv(np.dot(np.conj(W.T),_O_operator(s,R,W))))
    Y = np.dot(W,Uinv)

    return Y

def _sd(s,R,W, Nit=20 ,alpha=3*10**(-5), V = None, print_test = False):
    """An implementation of the steepest descent algorithm.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (numpy.ndarray): The basis vectors for the unit cell.
        W (numpy.ndarray): A matrix containing the expansion coefficients 
          for the wavefunctions
        Nit (int, optional): The number of iterations to go through.
        alpha (float): The weight that is to be applied to the updates 
          of the W matrix.
        V (function, optional): The function of the potential.
        print_test (boolean, optional): A boolean that indicates if the
          updated energy should be printed or not.

    Returns:
        W (numpy.ndarray): The updated coefficient matrix.
        E (float): The energy of the new wave functions.
    """
    
    if V == None: #pragma: no cover
        V = _sho_V

    for i in range(Nit):
        dW = _getgrad(s, R, W, V=V)
        W -= alpha*dW
        En = _getE(s,R,W,V=V)
        if print_test: #pragma: no cover
            print("iteration",i,"energy",En)
    return (W,En)

def _getPsi(s,R,W,V=None):
    """Finds the solutions to the schrodinger equation.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (numpy.ndarray): The basis vectors for the unit cell.
        W (numpy.ndarray): A matrix containing the expansion coefficients 
          for the wavefunctions
        V (function, optional): The function of the potential.
    
    Returns:
        Psi (numpy.ndarray): The eigensolutions to the schrodinger equation.
        epsilon (numpy.ndarray): The eigenvalues of the solutions.
    """
    
    if V == None: #pragma: no cover
        V = _sho_V

    Y = _Y(s,R,W)
    H =_H(s,R,Y,V=V)

    mu = np.dot(np.conj(Y.T),_H(s,R,Y,V=V))
    (epsilon, Psi) = np.linalg.eig(mu)

    return (Psi, np.real(epsilon))

def schrodinger(R,s, Ns = 4, Nit=400,V = None):
    """Solves the schrodinger equation for the given potential.

    Args:
        R (numpy.ndarray): The basis vectors for the unit cell.
        s (list of int): The number of samples points along each 
          basis vector.
        V (function, optional): The function that defines the potential.
        Nit (int, optional): The number of steps for the steepest decent
          algorithm.
        Ns (int, optional): The number of states to be solved for.

    Returns:
        Psi (numpy.ndarray): The wave function solutions of to the
          schrodinger equation.
        epsilon (numpy.ndarray): The energy eigenfunctions for the
          schrodigner equation.
    """

    if V == None: #pragma: no cover
        V = _sho_V
        
    W = np.random.normal(0,5,(np.prod(s),Ns)) + np.random.normal(0,5,(np.prod(s),Ns))*1j
    W = _Y(s,R,W)
    
    W = _sd(s,R,W,Nit=Nit,V=V)

    (Psi, epsilon) = _getPsi(s,R,W,V=V)

    return (Psi, epsilon)
