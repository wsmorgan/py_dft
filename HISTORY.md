# Revision History for "py_dft"

## Revision 0.4.0
- Added code to implement the full solution to the schrodiger
  equation. (The function _getPsi in schrodinger.py still has
  bugs in it).

## Revision 0.3.0
- Added the first three 4 sectios of assignment 2 to the
  schrodinger.py script (_getE, _H, _getgrad, _diagprod, and
  _diagouter).
- Updated the _O_operator, _L_operator, _B_operator, _Bj_operator in
  poisson.py.
- Added _B_dagg_operator and _Bj_dagg_operator to poisson.py.

## Revision 0.2.0
- Added ewald.py which does ewald summation.

## Revision 0.1.2
- Switched from a regular fourire transform to the fast fourier
  transform for finding the B and Bj operators.

## Revision 0.1.1
- Fixed an error with unit tests.
- Removed an print statement from dft.py.

## Revision 0.1.0
- Finished implementing the poisson solver in poisson.py.
- Fixed a bug in how Linv was calculated.
- Updated and added unit tests.
- Finished the driver.

## Revision 0.0.4
- Fixed the errors in B and Bj so that B(Bj(v)) = v for a simple cubic
  lattice.
- Fixed other malfunctioning unit tests.

## Revision 0.0.3
- Added the remaining operators and implemented the poisson solver.
- There are still bugs in the complex conjugate of B (Bj) operator
  that need to be solved.

## Revision 0.0.2
- Added code and tests for the L, O, and Linv operators.

## Revision 0.0.1
-Added poisson.py which solves the poisson equation.
-Adde dft.py which will be the driver for our code.

## Revision 0.0.0

The initial commit to the repo.

This includes the basis package that has a single module that defines
the potentials from a cfg file.