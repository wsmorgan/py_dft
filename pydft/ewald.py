"""Contains the code needed to find the Ewald energy of the system.
"""
import numpy as np
from itertools import product

def _find_structure_factor(x,z,G): #pragma: no cover
    """Finds the structure factors for the system.

    Args:
        x (list of float): The locations of the nuclear charges.
        z (list of float): The strength of each charge in atomic units.
        G (numpy.ndarray): The G vector for the system.

    Retruns:
        S (list of float): The structure factors for the system.
    """

    S = np.sum(np.exp(-1j*np.dot(G,np.transpose(x))),axis=1)

    return S

def _nuclear_charge_density(z,sigma,dr,sf,s,R): #pragma: no cover
    """Finds the nuclear charge density for the desired nuclei.

    Args:
        z (float): The strength of each charge.
        sigma (float): The sigma for the gaussian.
        dr (list of float): The between each point and the 
           center of the cell.
        sf (float): The structure factor for the cell.
        s (list of int): The number of samples points along each 
          basis vector.
        R (numpy.ndarray): A matrix conaining the lattice
          vectors. Each row is a different lattice vector.
    
    Returns:
        n (numpy.ndarray): The charge density for this nucleus.
    """

    from pydft.poisson import _gaussian, _B_operator, _Bj_operator
    g = z*_gaussian(dr,sigma)
    n = _B_operator(s,R,_Bj_operator(s,R,g)*sf)
    n = np.real(n)

    return n

def ewald_energy_arias(z,x,R,s,sigmas): #pragma: no cover
    """Finds the ewald energy of the system, i.e. Unn, using the
    approximation described by Thomas Arias. 

    Don't use this method. I've never gotten it to work properly and
    the exact method provides better results.

    Args:
        z (list of float): The strength of each nucleus.
        x (list of list of float): The location of each nucleus.
        sigmas (list of float): The width of the gaussians for 
          the charge distribution.
        s (list of int): The number of samples points along each 
          basis vector.
        R (numpy.ndarray): A matrix conaining the lattice
          vectors. Each row is a different lattice vector.

    Returns:
        Unum (float): The ewald energy for the interacting nuclei.

    """
    from pydft.poisson import _find_dr, _generate_r, _generate_G, poisson, _Bj_operator, _O_operator
    G = _generate_G(R,s)
    Sf = _find_structure_factor(x,z,G)
    r = _generate_r(R,s)
    dr = _find_dr(r,R)
    n = []
    phi = []
    for i in range(len(z)):
        n.append(_nuclear_charge_density(z[i],sigmas[i],dr,Sf,s,R))
        phi.append(poisson(s,R,n[i]))

    Unum = 0
    for i in range(len(z)):
        Unum += 0.5*np.real(np.dot(_Bj_operator(s,R,phi[i]),np.transpose(_O_operator(s,R,_Bj_operator(s,R,n[i])))))

    return Unum

def _find_alpha(Rc,accuracy):
    """Finds the ewald summation paramater alpha from the cutoff radius.

    Args:
        Rc (float): The cutoff radius for the summation.
        accuracy (float): The desired accuracy for the summation.

    Returns:
       alpha (float): The summation parameter alpha.
       kc (float): The cutoff radius in reciprical space.
    """

    p = np.sqrt(np.abs(np.log(accuracy)))
    alpha = p/Rc
    kc = 2*alpha*p

    return (alpha,kc)

def _phi_real(zs,rs,R,alpha, Rc):
    """Finds the real space portion of the ewald summation.

    Args:
        zs (list of int): The amount of charge on each nucleus.
        rs (list of list of float): Each entry contains the position of an atom.
        R (list of list of float): The lattice vectors.
        alpha (float): The ewald summation parameter.
        Rc (float): The cutoff radius for the summation.
    
    Returns:
        Ur (float): The energy contribution of the real portion of the sum.
    """

    from math import erfc
    Ur = 0
    # Find the real space points within the cutoff
    n = int(np.ceil(np.abs(Rc/np.linalg.norm(np.dot(R,[1,1,1]))))) + 1
    neighbs = np.array([np.array(i) for i in list(product(range(-n,n+1),repeat=3))])
    neighbs = np.dot(R,neighbs.T).T
    for n_L in neighbs:
        for a_i in range(len(zs)):
            for a_j in range(len(zs)):
                # We have to handle the central atom differently than
                # the rest of the sum.
                if np.all(n_L == 0) and (a_i != a_j):
                    rijn = np.linalg.norm(rs[a_i]-rs[a_j])
                    Ur += zs[a_i]*zs[a_j]*erfc(alpha *rijn)/rijn
                elif np.any(n_L != 0): 
                    rijn = np.linalg.norm(rs[a_i]-rs[a_j]+n_L)
                    Ur += zs[a_i]*zs[a_j]*erfc(alpha *rijn)/rijn
                    
    return Ur/2.

def _phi_recip(zs,rs,R,alpha,kc):
    """Finds the reciprocal space portion of the ewald summation.

    Args:
        zs (list of int): The amount of charge on each nucleus.
        rs (list of list of float): Each entry contains the position of an atom.
        R (list of list of float): The lattice vectors.
        alpha (float): The ewald summation parameter.
        kc (float): The cutoff radius for the summation.
    
    Returns:
        Um (float): The energy contribution of the reciprocal portion of the sum.
    """

    from cmath import exp
    
    Um = 0

    # Find the needed reciprical space points within the cutoff
    V = np.dot(R[0],np.cross(R[1],R[2]))
    k1 = 2*np.pi*np.cross(R[1],R[2])/V
    k2 = 2*np.pi*np.cross(R[2],R[0])/V
    k3 = 2*np.pi*np.cross(R[0],R[1])/V
    K = np.array([k1,k2,k3])
    m = int(np.ceil(np.abs(kc/np.linalg.norm(np.dot(K,[1,1,1])))*2*np.pi))+1
    ms = [np.dot(K,np.array(i).T) for i in list(product(list(range(-m,m+1)),repeat=3))]

    for m in ms:
        if np.any(m != 0):
            for a_i in range(len(zs)):
                for a_j in range(len(zs)):
                    Um += zs[a_i]*zs[a_j]* exp(-(np.pi*np.pi*np.dot(m,m)/(alpha*alpha))+2*np.pi*1j*np.dot(m,rs[a_i]-rs[a_j]))/np.dot(m,m)
    
    return Um/(2*np.pi*V)

def _phi_rem(zs,alpha):
    """Finds the energy missed by the real and reciprocal sums.

    Args:
        zs (list of int): The amount of charge on each nucleus.
        alpha (float): The ewald summation parameter.
    
    Returns:
        Uo (float): The energy missing energy controbution.
    """
        
    Uo = 0
    for z in zs:
        Uo += z*z

    Uo = -alpha*Uo/(2*np.sqrt(np.pi))

    return Uo

def ewald_energy_exact(z,x,R, Rc = None,accuracy = 1e-2):
    """Finds the ewald energy using the exact approach as described in
    these papers:

    Plain Ewald and PME by Thierry Matthey.

    Ewald summation techniques in perspective: a survey by Abdulnour
    Y. Toukmaji and John A. Board Jr.

    Args:
        z (list of float): The strength of each nucleus.
        x (list of list of float): The location of each nucleus.
        R (numpy.ndarray): A matrix conaining the lattice
          vectors. Each row is a different lattice vector.
        Rc (float): (Optional) The cutoff radius for the sum.

    Returns: 
        Unum (float): The ewald energy for the interacting nuclei.
    """

    V = np.dot(R[0],np.cross(R[1],R[2]))
    # if the cutoff radius isn't given then we assume that we want all
    # the atom's within 1 lattice parameter of the center of the cell.
    if Rc == None:
        Rc = V**(1./3)

    (alpha,kc) = _find_alpha(Rc,accuracy)

    Ur = _phi_real(z,x,R,alpha,Rc)
    Um = _phi_recip(z,x,R,alpha,kc)
    Uo = _phi_rem(z,alpha)

    Unum = Ur+np.real(Um)+Uo

    return Unum
