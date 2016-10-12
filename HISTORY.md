# Revision History for "py_dft"

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