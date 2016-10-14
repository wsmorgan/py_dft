"""Tests the evaluation and generation of the values for the poisson equation.
"""

import pytest
import numpy as np
import sys

def test_generate_M():
    """Tests that the generate M subroutine works.
    """
    from pydft.poisson import _generate_M

    assert  np.alltrue(_generate_M([3,2,1]) == np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                                               [0, 1, 0], [1, 1, 0], [2, 1, 0]]))
    assert np.alltrue(_generate_M([1,2,3]) == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1],
                                               [0, 1, 1], [0, 0, 2], [0, 1, 2]]))
    assert np.alltrue(_generate_M([3,3,3]) == np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0],
                                               [0, 1, 0], [1, 1, 0], [2, 1, 0],
                                               [0, 2, 0], [1, 2, 0], [2, 2, 0],
                                               [0, 0, 1], [1, 0, 1], [2, 0, 1],
                                               [0, 1, 1], [1, 1, 1], [2, 1, 1],
                                               [0, 2, 1], [1, 2, 1], [2, 2, 1],
                                               [0, 0, 2], [1, 0, 2], [2, 0, 2],
                                               [0, 1, 2], [1, 1, 2], [2, 1, 2],
                                               [0, 2, 2], [1, 2, 2], [2, 2, 2]]))
def test_generate_N():
    """Tests that the generate_N subroutine constructs the correct grid.
    """

    from pydft.poisson import _generate_N

    assert np.alltrue(_generate_N([3,2,1]) ==
                      np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
                                [1, 1, 0], [-1, 1, 0]]))
    assert np.alltrue(_generate_N([1,2,3]) == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1],
                                                        [0, 1, 1], [0, 0, -1], [0, 1, -1]]))
    assert np.alltrue(_generate_N([3,3,3]) ==
                      np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [1, 1, 0],
                                [-1, 1, 0], [0, -1, 0], [1, -1, 0], [-1, -1, 0], [0, 0, 1],
                                [1, 0, 1], [-1, 0, 1], [0, 1, 1], [1, 1, 1], [-1, 1, 1],
                                [0, -1, 1], [1, -1, 1], [-1, -1, 1], [0, 0, -1], [1, 0, -1],
                                [-1, 0, -1], [0, 1, -1], [1, 1, -1], [-1, 1, -1], [0, -1, -1],
                                [1, -1, -1], [-1, -1, -1]]))

def test_generate_r():
    """Tests that the generate_r subroutine constructs the correct matrix.
    """

    from pydft.poisson import _generate_r

    R = [[1,0,0],[0,1,0],[0,0,1]]
    S = [3,3,3]
    assert np.allclose(_generate_r(R,S),np.array([[0.0, 0.0, 0.0], [0.3333333333333333, 0.0, 0.0], [0.6666666666666666, 0.0, 0.0], [0.0, 0.3333333333333333, 0.0], [0.3333333333333333, 0.3333333333333333, 0.0], [0.6666666666666666, 0.3333333333333333, 0.0], [0.0, 0.6666666666666666, 0.0], [0.3333333333333333, 0.6666666666666666, 0.0], [0.6666666666666666, 0.6666666666666666, 0.0], [0.0, 0.0, 0.3333333333333333], [0.3333333333333333, 0.0, 0.3333333333333333], [0.6666666666666666, 0.0, 0.3333333333333333], [0.0, 0.3333333333333333, 0.3333333333333333], [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], [0.6666666666666666, 0.3333333333333333, 0.3333333333333333], [0.0, 0.6666666666666666, 0.3333333333333333], [0.3333333333333333, 0.6666666666666666, 0.3333333333333333], [0.6666666666666666, 0.6666666666666666, 0.3333333333333333], [0.0, 0.0, 0.6666666666666666], [0.3333333333333333, 0.0, 0.6666666666666666], [0.6666666666666666, 0.0, 0.6666666666666666], [0.0, 0.3333333333333333, 0.6666666666666666], [0.3333333333333333, 0.3333333333333333, 0.6666666666666666], [0.6666666666666666, 0.3333333333333333, 0.6666666666666666], [0.0, 0.6666666666666666, 0.6666666666666666], [0.3333333333333333, 0.6666666666666666, 0.6666666666666666], [0.6666666666666666, 0.6666666666666666, 0.6666666666666666]]))

    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    S = [3,2,1]
    assert np.allclose(_generate_r(R,S),np.array([[0.0, 0.0, 0.0], [0.16666666666666666, 0.0, 0.0], [0.3333333333333333, 0.0, 0.0], [0.0, -0.25, 0.0], [0.16666666666666666, -0.25, 0.0], [0.3333333333333333, -0.25, 0.0]]))

    R = [[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]]
    S = [1,2,3]
    assert np.allclose(_generate_r(R,S),np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.16666666666666666], [0.0, 0.0, 0.16666666666666666], [0.0, 0.0, 0.3333333333333333], [0.0, 0.0, 0.3333333333333333]]))

def test_generate_G():
    """Tests the construction of the G vectors for the system.
    """
    from pydft.poisson import _generate_G

    R = [[1,0,0,],[0,1,0],[0,0,1]]
    s = [1,2,3]
    assert np.allclose(_generate_G(R,s),np.array([[0.0, 0.0, 0.0], [0.0, 6.283185307179586, 0.0], [0.0, 0.0, 6.283185307179586], [0.0, 6.283185307179586, 6.283185307179586], [0.0, 0.0, -6.283185307179586], [0.0, 6.283185307179586, -6.283185307179586]]))

    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    s = [3,2,1]
    assert np.allclose(_generate_G(R,s),np.array([[0.0, 0.0, 0.0], [6.283185307179586, 6.283185307179586, 0.0], [-6.283185307179586, -6.283185307179586, 0.0], [6.283185307179586, 0.0, 6.283185307179586], [12.566370614359172, 6.283185307179586, 6.283185307179586], [0.0, -6.283185307179586, 6.283185307179586]]))

    R = [[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]]
    s = [3,3,3]
    assert np.allclose(_generate_G(R,s),np.array([[0.0, 0.0, 0.0], [6.283185307179586, 6.283185307179586, -6.283185307179586], [-6.283185307179586, -6.283185307179586, 6.283185307179586], [6.283185307179586, -6.283185307179586, 6.283185307179586], [12.566370614359172, 0.0, 0.0], [0.0, -12.566370614359172, 12.566370614359172], [-6.283185307179586, 6.283185307179586, -6.283185307179586], [0.0, 12.566370614359172, -12.566370614359172], [-12.566370614359172, 0.0, 0.0], [-6.283185307179586, 6.283185307179586, 6.283185307179586], [0.0, 12.566370614359172, 0.0], [-12.566370614359172, 0.0, 12.566370614359172], [0.0, 0.0, 12.566370614359172], [6.283185307179586, 6.283185307179586, 6.283185307179586], [-6.283185307179586, -6.283185307179586, 18.84955592153876], [-12.566370614359172, 12.566370614359172, 0.0], [-6.283185307179586, 18.84955592153876, -6.283185307179586], [-18.84955592153876, 6.283185307179586, 6.283185307179586], [6.283185307179586, -6.283185307179586, -6.283185307179586], [12.566370614359172, 0.0, -12.566370614359172], [0.0, -12.566370614359172, 0.0], [12.566370614359172, -12.566370614359172, 0.0], [18.84955592153876, -6.283185307179586, -6.283185307179586], [6.283185307179586, -18.84955592153876, 6.283185307179586], [0.0, 0.0, -12.566370614359172], [6.283185307179586, 6.283185307179586, -18.84955592153876], [-6.283185307179586, -6.283185307179586, -6.283185307179586]]))
    
def test_find_Gsqu():
    """Tests the codes finding of the squared norm of the G vector.
    """

    from pydft.poisson import _find_Gsqu, _generate_G

    R = [[1,0,0,],[0,1,0],[0,0,1]]
    s = [3,2,1]
    assert np.allclose(_find_Gsqu(_generate_G(R,s)),np.array([0.0, 39.478417604357432, 39.478417604357432, 39.478417604357432, 78.956835208714864, 78.956835208714864]))

    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    s = [1,2,3]
    assert np.allclose(_find_Gsqu(_generate_G(R,s)),np.array([0.0, 78.956835208714864, 78.956835208714864, 236.87050562614459, 78.956835208714864, 78.956835208714864]))

    R = [[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]]
    s = [3,3,3]
    assert np.allclose(_find_Gsqu(_generate_G(R,s)),np.array([0.0, 118.43525281307228, 118.43525281307228, 118.43525281307228, 157.91367041742973, 315.82734083485946, 118.43525281307228, 315.82734083485946, 157.91367041742973, 118.43525281307228, 157.91367041742973, 315.82734083485946, 157.91367041742973, 118.43525281307228, 434.26259364793174, 315.82734083485946, 434.26259364793174, 434.26259364793174, 118.43525281307228, 315.82734083485946, 157.91367041742973, 315.82734083485946, 434.26259364793174, 434.26259364793174, 157.91367041742973, 434.26259364793174, 118.43525281307228]))    
def test_find_dr():
    """Tests that the find dr subroutine finds the correct distances.
    """
    from pydft.poisson import _find_dr, _generate_r
    R = [[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]]
    s = [3,3,3]
    r = _generate_r(R,s)
    assert np.allclose(_find_dr(r,R),np.array([0.8660254037844386, 0.78173595997057166, 0.72648315725677892, 0.8660254037844386, 0.78173595997057166, 0.72648315725677892, 0.8660254037844386, 0.78173595997057166, 0.72648315725677892, 0.78173595997057166, 0.68718427093627688, 0.62360956446232363, 0.78173595997057166, 0.68718427093627688, 0.62360956446232363, 0.78173595997057166, 0.68718427093627688, 0.62360956446232363, 0.72648315725677892, 0.62360956446232363, 0.55277079839256671, 0.72648315725677892, 0.62360956446232363, 0.55277079839256671, 0.72648315725677892, 0.62360956446232363, 0.55277079839256671]))

    R = [[1,0,0,],[0,1,0],[0,0,1]]
    s = [1,2,3]
    r = _generate_r(R,s)
    assert np.allclose(_find_dr(r,R),np.array([0.8660254037844386, 0.70710678118654757, 0.72648315725677892, 0.52704627669472992, 0.72648315725677892, 0.52704627669472992]))
    
    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    s = [3,2,1]
    r = _generate_r(R,s)
    assert np.allclose(_find_dr(r,R),np.array([0.4330127018922193, 0.36324157862838946, 0.36324157862838946, 0.61237243569579447, 0.56519416526043897, 0.56519416526043897]))
    
def test_gaussian():
    """Tests the evaluation of the gaussian function.
    """
    
    from pydft.poisson import _gaussian, _generate_r, _find_dr
    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    s = [1,2,3]
    sigma = 0.5
    r = _generate_r(R,s)
    dr = _find_dr(r,R)
    assert np.allclose(_gaussian(dr,sigma),np.array([ 0.34910796,  0.23993816,  0.3901348 ,  0.26813547,  0.3901348 ,0.26813547]))

    R = [[1,0,0,],[0,1,0],[0,0,1]]
    s = [3,2,1]
    sigma = 0.25
    r = _generate_r(R,s)
    dr = _find_dr(r,R)
    assert np.allclose(_gaussian(dr,sigma),np.array([0.010072639249693521, 0.059596720089735565, 0.059596720089735565, 0.074427296480276031, 0.44036350805532332, 0.44036350805532332]))

    R = [[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]]
    s = [3,3,3]
    sigma = 0.75
    r = _generate_r(R,s)
    dr = _find_dr(r,R)
    assert np.allclose(_gaussian(dr,sigma),np.array([0.077271039142547598, 0.087424539903614443, 0.094146313078274091, 0.077271039142547598, 0.087424539903614443, 0.094146313078274091, 0.077271039142547598, 0.087424539903614443, 0.094146313078274091, 0.087424539903614443, 0.098912221993792224, 0.1065172436636445, 0.087424539903614443, 0.098912221993792224, 0.1065172436636445, 0.087424539903614443, 0.098912221993792224, 0.1065172436636445, 0.094146313078274091, 0.1065172436636445, 0.11470698937905052, 0.094146313078274091, 0.1065172436636445, 0.11470698937905052, 0.094146313078274091, 0.1065172436636445, 0.11470698937905052]))
    
    R = [[6,0,0],[0,6,0],[0,0,6]]
    s = [20,25,30]
    sigma = 0.75
    r = _generate_r(R,s)
    dr = _find_dr(r,R)
    g = _gaussian(dr,sigma)
    intg = sum(g*np.linalg.det(R)/float(np.prod(s)))
    assert np.allclose([1],[intg],atol=1E-3)

    R = [[6,0,0],[0,6,0],[0,0,6]]
    s = [20,25,30]
    sigma = 0.25
    r = _generate_r(R,s)
    dr = _find_dr(r,R)
    g = _gaussian(dr,sigma)
    intg = sum(g*np.linalg.det(R)/float(np.prod(s)))
    assert np.allclose([1],[intg],atol=1E-3)    

def test_charge_dist():
    """Tests the construction of the charge distribution function.
    """
    from pydft.poisson import charge_dist, _generate_r

    R = [[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]]
    s = [3,3,3]
    sigmas = [0.75,0.25]
    coeffs = [1,-1]
    assert np.allclose(charge_dist(s,R,coeffs,sigmas),np.array([0.067198399892854074, 0.056826563571350616, 0.034549592988538526, 0.067198399892854074, 0.056826563571350616, 0.034549592988538526, 0.067198399892854074, 0.056826563571350616, 0.034549592988538526, 0.056826563571350616, 0.0059637769615466657, -0.074521606788759701, 0.056826563571350616, 0.0059637769615466657, -0.074521606788759701, 0.056826563571350616, 0.0059637769615466657, -0.074521606788759701, 0.034549592988538526, -0.074521606788759701, -0.23790854240050377, 0.034549592988538526, -0.074521606788759701, -0.23790854240050377, 0.034549592988538526, -0.074521606788759701, -0.23790854240050377]))

    R = [[1,0,0],[0,1,0],[0,0,1]]
    s = [2,2,2]
    sigmas = [0.5,0.025,0.3]
    coeffs = [1,-1,0.5]
    assert np.allclose(charge_dist(s,R,sigmas,coeffs),np.array([0.056911838376348944, 0.08201987696049548, 0.08201987696049548, 0.12184330048684332, 0.08201987696049548, 0.12184330048684332, 0.12184330048684332, 0.18571888510765483]))

    
    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    s = [1,3,1]
    sigmas = [0.75,0.25]
    coeffs = [0,0]
    assert np.allclose(charge_dist(s,R,coeffs,sigmas),np.array([0.0, 0.0, 0.0]))

    R = [[6,0,0],[0,6,0],[0,0,6]]
    s = [20,25,30]
    sigmas = [0.75,0.25]
    coeffs = [1,-1]
    n = charge_dist(s,R,coeffs,sigmas)
    sumn = sum(n*np.linalg.det(R)/float(np.prod(s)))
    assert np.allclose([0],[sumn],atol=1E-3)

    R = [[6,0,0],[0,6,0],[0,0,6]]
    s = [20,25,30]
    sigmas = [0.75,0.25]
    coeffs = [1]
    with pytest.raises(ValueError):
        charge_dist(s,R,coeffs,sigmas)

def test_O():
    """Test the generation of the O operator.
    """

    from pydft.poisson import _O_operator

    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    s = [2,4,5]
    out = np.identity(np.prod(s))*np.linalg.det(R)
    v = np.random.normal(0,0.1,40)
    assert np.allclose(_O_operator(s,R,v),np.dot(out,v))
    
    R = [[1,0,0],[0,2,0],[0,0,4]]
    s = [1,5,6]
    out = np.identity(np.prod(s))*np.linalg.det(R)
    v = np.random.normal(0.1,0.5,30)
    assert np.allclose(_O_operator(s,R,v),np.dot(out,v))
    assert np.allclose(_O_operator(s,R,v)/v,np.linalg.det(R))

    R = [[2.5,2.5,-2.5],[0.5,-0.5,0.5],[-1.5,1.5,1.5]]
    s = [10,2,6]
    out = np.identity(np.prod(s))*np.linalg.det(R)
    v = np.random.normal(0,0.25,120)
    assert np.allclose(_O_operator(s,R,v),np.dot(out,v))

def test_L():
    """Tests the L operator.
    """
    from pydft.poisson import _L_operator, _generate_G, _find_Gsqu

    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    s = [2,4,5]
    G = _generate_G(R,s)
    G2 = _find_Gsqu(G)
    L = -np.linalg.det(R)*np.diag(G2)
    v = np.random.normal(0,0.1,40)
    assert np.allclose(_L_operator(s,R,v),np.dot(L,v))

    R = [[1,0,0],[0,2,0],[0,0,4]]
    s = [1,5,6]
    G = _generate_G(R,s)
    G2 = _find_Gsqu(G)
    L = -np.linalg.det(R)*np.diag(G2)
    v = np.random.normal(0.1,0.5,30)
    assert np.allclose(_L_operator(s,R,v),np.dot(L,v))
    assert np.allclose(_L_operator(s,R,v)/v,-np.linalg.det(R)*G2)

    R = [[2.5,2.5,-2.5],[0.5,-0.5,0.5],[-1.5,1.5,1.5]]
    s = [10,2,6]
    G = _generate_G(R,s)
    G2 = _find_Gsqu(G)
    L = -np.linalg.det(R)*np.diag(G2)
    v = np.random.normal(0,0.25,120)
    assert np.allclose(_L_operator(s,R,v),np.dot(L,v))

def test_Linv():
    """Tests the Linv operator.
    """
    from pydft.poisson import _Linv_operator, _generate_G, _find_Gsqu, _L_operator

    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    s = [2,4,5]
    G = _generate_G(R,s)
    G2 = _find_Gsqu(G)
    G2[0] = 1.0
    Linv = -np.diag(1/(G2*np.linalg.det(R)))
    Linv[0][0] = -0.0
    v = np.random.normal(0,0.1,40)
    assert np.allclose(_Linv_operator(s,R,v),np.dot(Linv,v))

    R = [[1,0,0],[0,2,0],[0,0,4]]
    s = [1,5,6]
    G = _generate_G(R,s)
    G2 = _find_Gsqu(G)
    G2[0] = 1.0
    Linv = -np.diag(1/(G2*np.linalg.det(R)))
    Linv[0][0] = -0.0
    v = np.random.normal(0.1,0.5,30)
    assert np.allclose(_Linv_operator(s,R,v),np.dot(Linv,v))
    assert np.allclose(_L_operator(s,R,_Linv_operator(s,R,v))[1:],v[1:])

    R = [[2.5,2.5,-2.5],[0.5,-0.5,0.5],[-1.5,1.5,1.5]]
    s = [10,2,6]
    G = _generate_G(R,s)
    G2 = _find_Gsqu(G)
    G2[0] = 1.0
    Linv = -np.diag(1/(G2*np.linalg.det(R)))
    Linv[0][0] = -0.0
    v = np.random.normal(0,0.25,120)
    assert np.allclose(_Linv_operator(s,R,v),np.dot(Linv,v))
    
def test_B():
    """Tests of the B operator.
    """

    from pydft.poisson import _B_operator, _generate_G, _generate_r

    s = [1,2,3]
    R = [[2.5,2.5,-2.5],[0.5,-0.5,0.5],[-1.5,1.5,1.5]]
    # G = _generate_G(R,s)
    # r = _generate_r(R,s)
    # B = np.exp(1j*np.dot(G,np.transpose(r)))
    v = np.random.normal(0,0.5,6)
    out = np.fft.fftn(v.reshape(s,order="F")).reshape(np.prod(s),order="F")
    assert np.allclose(_B_operator(s,R,v), out)

    s = [10,2,3]
    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    # G = _generate_G(R,s)
    # r = _generate_r(R,s)
    # B = np.exp(1j*np.dot(G,np.transpose(r)))
    v = np.random.normal(0,0.5,60)
    out = np.fft.fftn(v.reshape(s,order="F")).reshape(np.prod(s),order="F")
    assert np.allclose(_B_operator(s,R,v),out)
    
    s = [5,5,2]
    R = [[6.0,0.0,0.0],[0.0,6.0,0.0],[0.0,0.0,6.0]]
    # G = _generate_G(R,s)
    # r = _generate_r(R,s)
    # B = np.exp(1j*np.dot(G,np.transpose(r)))
    v = np.random.normal(0,0.25,50)
    out = np.fft.fftn(v.reshape(s,order="F")).reshape(np.prod(s),order="F")
    assert np.allclose(_B_operator(s,R,v),out)

def test_Bj():
    """Tests of the B conjugate transpose operator.
    """
    from pydft.poisson import _Bj_operator, _generate_G, _generate_r, _B_operator

    s = [5,5,2]
    R = [[6.0,0.0,0.0],[0.0,6.0,0.0],[0.0,0.0,6.0]]
    # G = _generate_G(R,s)
    # r = _generate_r(R,s)
    # B = np.exp(1j*np.dot(G,np.transpose(r)))
    v = np.random.normal(0,0.25,50)
    # Bj = np.transpose(B.conjugate())/np.prod(s)
    out = np.fft.ifftn(v.reshape(s,order="F")).reshape(np.prod(s),order="F")
    assert np.allclose(_Bj_operator(s,R,v),out)
    assert np.allclose(_B_operator(s,R,_Bj_operator(s,R,v)),v)

    s = [1,2,3]
    R = [[2.5,2.5,-2.5],[0.5,-0.5,0.5],[-1.5,1.5,1.5]]
    # G = _generate_G(R,s)
    # r = _generate_r(R,s)
    # B = np.exp(1j*np.dot(G,np.transpose(r)))
    v = np.random.normal(0,0.5,6)
    # Bj = np.transpose(B.conjugate())/np.prod(s)
    out = np.fft.ifftn(v.reshape(s,order="F")).reshape(np.prod(s),order="F")
    assert np.allclose(_Bj_operator(s,R,v),out)

    s = [10,2,3]
    R = [[0.5,0.5,-0.5],[0.5,-0.5,0.5],[-0.5,0.5,0.5]]
    # G = _generate_G(R,s)
    # r = _generate_r(R,s)
    # B = np.exp(1j*np.dot(G,np.transpose(r)))
    v = np.random.normal(0,0.5,60)
    # Bj = np.transpose(B.conjugate())/np.prod(s)
    out = np.fft.ifftn(v.reshape(s,order="F")).reshape(np.prod(s),order="F")
    assert np.allclose(_Bj_operator(s,R,v),out)
    
def test_poission():
    """Tests the solution to poissons equation
    """

    from pydft.poisson import poisson, charge_dist, _Bj_operator, _O_operator, _B_operator, _Linv_operator
    R = [[6,0,0],[0,6,0],[0,0,6]]
    s = [20,15,15]
    coefs = [-1,1]
    sigmas = [0.75,0.5]
    n = charge_dist(s,R,coefs,sigmas)
    phi = _B_operator(s,R,_Linv_operator(s,R,-4*np.pi*_O_operator(s,R,_Bj_operator(s,R,n))))
    assert np.allclose(phi,poisson(s,R,n))
    
    phi = np.real(phi)
    Unum=0.5*np.real(np.dot(_Bj_operator(s,R,phi),np.transpose(_O_operator(s,R,_Bj_operator(s,R,n)))))
    Uanal=((1/sigmas[0]+1/sigmas[1])/2-np.sqrt(2)/np.sqrt(sigmas[0]**2+sigmas[1]**2))/np.sqrt(np.pi)
    assert np.allclose(Unum,Uanal,atol=1e-4)
