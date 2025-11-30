# Math Library Development Plan

This document outlines the step-by-step development process for the `math` library, prioritizing minimal functional modules.

## Phase 1: Core Infrastructure (Foundation)
**Goal**: Establish the basic building blocks required by all other modules.

1.  **Constants (`core/constants.hpp`)**
    -   Define mathematical constants (PI, E, etc.) with high precision.
    -   *Verification*: Static assertions or simple print tests.
2.  **Utilities (`core/utils.hpp`)**
    -   Implement floating-point comparison helpers (epsilon checks).
    -   Implement basic angle conversions (degrees <-> radians).
    -   *Verification*: Unit tests for comparison edge cases.
3.  **Basic Functions (`core/functions.hpp`)**
    -   Wrap standard library math functions or implement custom versions if needed (e.g., clamp, lerp).
    -   *Verification*: Unit tests against expected values.

## Phase 2: Linear Algebra (Vectors & Matrices)
**Goal**: Implement data structures for numerical computation.

4.  **Vector Class (`linalg/vector.hpp`)**
    -   Implement a template `Vector<T, N>` class.
    -   Add operations: addition, subtraction, scalar multiplication, dot product, norm.
    -   *Verification*: Unit tests for vector arithmetic.
5.  **Matrix Class (`linalg/matrix.hpp`)**
    -   Implement a template `Matrix<T, R, C>` class.
    -   Add operations: addition, subtraction, multiplication (matrix-matrix, matrix-vector).
    -   Implement identity matrix generation.
    -   *Verification*: Unit tests for matrix operations.

## Phase 3: Advanced Linear Algebra (Solvers & Decompositions)
**Goal**: Add capabilities to solve linear systems.

6.  **Decomposition (`linalg/decomposition.hpp`)**
    -   Implement LU Decomposition.
    -   (Optional) QR or Cholesky if time permits.
    -   *Verification*: Verify $A = LU$.
7.  **Linear Solvers (`linalg/solver.hpp`)**
    -   Implement a solver for $Ax = b$ using LU decomposition.
    -   implement Inverse calculation.
    -   *Verification*: Solve known systems and check residuals.

## Phase 4: Calculus (Numerical Methods)
**Goal**: Implement numerical differentiation and integration.

8.  **Differentiation (`calculus/differentiation.hpp`)**
    -   Implement numerical derivative (finite difference: forward, backward, central).
    -   *Verification*: Differentiate polynomials and compare with analytical results.
9.  **Integration (`calculus/integration.hpp`)**
    -   Implement Trapezoidal rule and Simpson's rule.
    -   *Verification*: Integrate standard functions over known intervals.
10. **ODE Solvers (`calculus/ode.hpp`)**
    -   Implement Euler method and Runge-Kutta 4 (RK4).
    -   *Verification*: Solve basic ODEs (e.g., exponential growth).

## Development Workflow
For each step:
1.  **Implement**: Write code in `include/math/...`.
2.  **Test**: Create a specific test file in `tests/`.
3.  **Build**: Run `cmake -B build -G Ninja` and `cmake --build build`.
4.  **Verify**: Run `ctest --test-dir build` or execute the specific test binary.
