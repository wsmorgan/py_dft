"""This module contains methods used to solve the poisson equation."""
import numpy as np

def _generate_M(s):
    """This function finds the M matrix as described by
    'https://www.youtube.com/watch?v=FJpqaax7G-Y'.
    
    Args:
        s (list int): The number of divisions along the 3 
          basis vectors.
    Returns:
        M (numpy.ndarray): A matrix of the number of samples 
          for the space.
    """

    M = []
    for i in range(s[2]):
        for j in range(s[1]):
            for k in range(s[0]):
                M.append([k, j, i])

    return np.array(M)

def _generate_N(s):
    """This function generates the N matrix as defined by ''.

    Args: 
        s (list of int): The number of divisions along the three
          lattice vectors.

    Returns: 
        N (numpy.ndarray): The points in the cell where the
          functions will be evaluated.
    """

    N = []
    M = _generate_M(s)
    for row in M:
        n0 = row[0] - (s[0] if row[0]>s[0]/2. else 0)
        n1 = row[1] - (s[1] if row[1]>s[1]/2. else 0)
        n2 = row[2] - (s[2] if row[2]>s[2]/2. else 0)
        N.append([n0,n1,n2])

    return np.array(N)

def _generate_r(R,s):
    """Finds the space vector evaluation points for within the unit cell.

    Args:
        R (numpy.ndarray): A matrix conaining the lattice
          vectors. Each row is a different lattice vector.
        s (list of int): The number of divisions along each
          lattice vector.
    
    Returns:
        r (numpy.ndarray): A matrix of the sample points within 
          the unit cell.
    """
    M = _generate_M(s)
    S = np.diag(s)
    SiRt = np.linalg.inv(S)*np.transpose(R)
    r = np.inner(M,SiRt)
    return r

def _generate_G(R,s):
    """Finds the G vectors representation for within the unit cell.

    Args:
        R (numpy.ndarray): A matrix conaining the lattice
          vectors. Each row is a different lattice vector.
        s (list of int): The number of divisions along each
          lattice vector.
    
    Returns:
        G (numpy.ndarray): A matrix of the G vectors values within 
          the unit cell.
    """

    N = _generate_N(s)
    NRi = np.inner(N,np.linalg.inv(R))
    return 2*np.pi*NRi

def _find_Gsqu(G):
    """Computes the square of the norm of the rows in the G matrix.

    Args:
        G (numpy.ndarray): The matrix of the vectors values within
          the unit cell.

    Returns: 
        Gsqu (np.ndarray): The list of the squared norms of the
          rows of G.
    """

    Gsqu = []
    for row in G:
        Gsqu.append(np.linalg.norm(row)**2)

    return np.array(Gsqu)

def _find_dr(r,R):
    """Finds the distance between each sample point and the center of the cell.

    Args:
        r (numpp.ndarray): The list of sample points.
        R (numpy.ndarry): The basis vectors for the cell.
    """
    dr = []
    mid = sum(np.array(R))/2.
    for row in r:
        dr.append(np.linalg.norm(mid-row))

    return np.array(dr)

def _gaussian(r,sigma):
    """Finds the value of a gaussian for the evaluation points in r.

    Args:
        r (numpy.ndarray): The distance of each sample point to the 
          center of the cell.
        sigma (float): The value of sigma in the gaussian function.
    
    Returns:
        g (numpy.ndarray): The value of the gaussian evaluated at all 
          the sample points.
    """

    g = np.exp(-(np.power(r,2))/(2*sigma**2))/(2*np.pi*sigma**2)**(3/2.)

    return g

def charge_dist(s,R,coeffs,sigmas):
    """Finds the charge distribution for the system represented as
    a sumattion of gaussians.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (np.ndarray): The basis vectors for the unit cell.
        coeffs (list of float): The coefficient for each gaussian.
        sigmas (list of float): The sigma values for each gaussian.

    Returns:
       n (np.ndarray): The value of the charge distribution at each 
         point in the space.

    Raises:
        ValueError: if the argument is not an `int` of `float`.

    Examples:
        The following example finds the charge distribution contsructed 
        from two gaussians in a simple cubic unit cell. The distribution 
        is evaluated at such that there are 10 sample points in the R[0] 
        direction, 5 in the R[1] direction, and 15 in th R[2] directon. 
        The distribution is then constructed as:
        n(r) = coeff1*exp(-(r^2)/(2*sigma1^2))/(2*pi*sigma1^2)^(3/2) +
        coeff2*exp(-(r^2)/(2*sigma2^2))/(2*pi*sigma2^2)^(3/2)
        where r is the distance from each sample point to the center of
        the cell.
        >>> from pydft.poisson import charge_dist
        >>> import numpy as np
        >>> s = [10,5,15]
        >>> R = np.array([[1,0,0],[0,1,0],[0,0,1])
        >>> coeffs = [1, -1]
        >>> sigmas = [0.75, 0.5]
        >>> n = charge_dist(s,R,coeffs,sigmas)
    """

    if len(coeffs) != len(sigmas):
        raise ValueError("Number of coeffs and and sigmas must match.")
    
    r = _generate_r(R,s)
    dr = _find_dr(r,R)

    n = np.zeros(len(r))
    for i in range(len(coeffs)):
        n += coeffs[i]*_gaussian(dr,sigmas[i])

    return n

def _O_operator(s,R,v):
    """Applies the O operator to the vector v.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (np.ndarray): The basis vectors for the unit cell.
        v (np.ndarray):  1D array of the vector to operate on.

    Returns:
        result (np.ndarray): The result of O operating on v.
    """

    dim = np.prod(s)
    detR = np.linalg.det(R)
    O = np.identity(dim)*detR
    result = O*v
    
    return result

def _L_operator(s,R,v):
    """Applies the L operator to the vector v.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (np.ndarray): The basis vectors for the unit cell.
        v (np.ndarray):  1D array of the vector to operate on.

    Returns:
        result (np.ndarray): The result of L operating on v.
    """

    G = _generate_G(R,s)
    G2 = _find_Gsqu(G)
    L = -np.linalg.det(R)*np.diag(G2)
    result = L*v

    return result

def _Linv_operator(s,R,v):
    """Applies the Linv operator to the vector v.

    Args:
        s (list of int): The number of samples points along each 
          basis vector.
        R (np.ndarray): The basis vectors for the unit cell.
        v (np.ndarray):  1D array of the vector to operate on.

    Returns:
        result (np.ndarray): The result of Linv operating on v.
    """
    G = _generate_G(R,s)
    G2 = _find_Gsqu(G)
    Linv = -np.diag(1/G2*np.linalg.det(R))
    result = Linv*v

    return result

# def poisson():

#     """Calculates the solution to poisson's equation.
#     """