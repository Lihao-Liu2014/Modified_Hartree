#ifndef RADIAL_OPERATOR_H
#define RADIAL_OPERATOR_H

#include <vector>
#include <cmath>
#include <stdexcept>

/*
    Radial logarithmic grid and tridiagonal Hamiltonian builder
    for the transformed radial function

        h(u) = r^{3/2} φ_{nℓ}(r),  r = e^u.

    We work on a uniform u-grid:
        u_i = u_min + i * du,  i = 0..Nu-1,
        r_i = exp(u_i).

    We enforce Dirichlet boundary conditions:
        h(u_min) = h(u_max) = 0.

    The differential operator (no Fock term) from Eq. (10.92) is:

      ε h(u) =
        { -1/2 [ (1/r) d^2/du^2 (1/r) - (ℓ+1/2)^2 / r^2 ]
          - Z/r + v_H_eff(r) } h(u).

    On the grid we restrict ourselves to the *interior* points
        i = 1..Nu-2,
    so the matrix dimension is N = Nu-2.

    For a given ℓ and potential V_eff(r_i) = -Z/r_i + v_H_eff(r_i),
    we construct a symmetric tridiagonal matrix with diagonals:

        a[j] = A_{j, j-1},   (sub-diagonal)
        b[j] = A_{j, j},     (main diagonal)
        c[j] = A_{j, j+1},   (super-diagonal)

    where j = 0..N-1 index the *interior* grid points:

        i = j + 1  ↔  u_i, r_i.

    You can then use these with tri::solve_tridiagonal.
*/

struct RadialGrid {
    int    Nu;      // number of u-points (including boundaries)
    double u_min;
    double u_max;
    double du;
    std::vector<double> u; // u_i
    std::vector<double> r; // r_i = exp(u_i)
};

// Create a logarithmic grid
inline RadialGrid make_log_grid(int Nu, double u_min, double u_max)
{
    if (Nu < 3) {
        throw std::runtime_error("make_log_grid: Nu must be >= 3");
    }

    RadialGrid g;
    g.Nu   = Nu;
    g.u_min = u_min;
    g.u_max = u_max;
    g.du   = (u_max - u_min) / (Nu - 1);

    g.u.resize(Nu);
    g.r.resize(Nu);

    for (int i = 0; i < Nu; ++i) {
        g.u[i] = u_min + i * g.du;
        g.r[i] = std::exp(g.u[i]);
    }
    return g;
}

/*
    Build ONLY the kinetic + centrifugal part of the operator
    (no potential) as a tridiagonal matrix.

    This corresponds to

        T_ℓ h(u) = -1/2 [ (1/r) d^2/du^2 (1/r h) - (ℓ+1/2)^2 / r^2 h ]

    Using central differences for the second derivative:

        f_i     = h_i / r_i
        f''_i ~ (f_{i+1} - 2 f_i + f_{i-1}) / du^2

    Restrict to interior points i = 1..Nu-2 ⇒ dimension N = Nu-2
    and define j = i-1 (0..N-1) as the interior index.
*/
inline void build_kinetic_tridiagonal(const RadialGrid& g,
                                      int l,
                                      std::vector<double>& a, // size N-1
                                      std::vector<double>& b, // size N
                                      std::vector<double>& c) // size N-1
{
    const int Nu = g.Nu;
    const int N  = Nu - 2;
    if (N <= 0) {
        throw std::runtime_error("build_kinetic_tridiagonal: grid too small");
    }

    a.resize(N - 1);
    b.resize(N);
    c.resize(N - 1);

    const double du2 = g.du * g.du;
    const double lp = l + 0.5; // ℓ + 1/2

    // interior points: i = 1..Nu-2, j = i-1
    for (int j = 0; j < N; ++j) {
        int i = j + 1;
        double ri   = g.r[i];
        double ri2  = ri * ri;

        // coefficients from expanding T_ℓ acting on h
        // see analysis: T(h)_i = A_{i,i-1} h_{i-1} + A_{i,i} h_i + A_{i,i+1} h_{i+1}
        // with
        //   A_{i,i-1} = -1/2 * (1 / (ri * du^2 * r_{i-1}))
        //   A_{i,i+1} = -1/2 * (1 / (ri * du^2 * r_{i+1}))
        //   A_{i,i}   =  1/(ri^2 du^2) + 0.5 * (ℓ+1/2)^2 / ri^2

        double A_im1 = 0.0;
        double A_ip1 = 0.0;

        if (i > 1) {
            double rim1 = g.r[i - 1];
            A_im1 = -0.5 * (1.0 / (ri * du2 * rim1));
        }
        if (i < Nu - 2) {
            double rip1 = g.r[i + 1];
            A_ip1 = -0.5 * (1.0 / (ri * du2 * rip1));
        }
        double A_ii = 1.0 / (ri2 * du2) + 0.5 * (lp * lp) / ri2;

        // store in tridiagonal form
        b[j] = A_ii;
        if (j > 0)     a[j - 1] = A_im1; // sub-diagonal at (j, j-1)
        if (j < N - 1) c[j]     = A_ip1; // super-diagonal at (j, j+1)
    }
}

/*
    Add a LOCAL potential V_eff(r) (array size Nu) to the
    diagonal of the kinetic tridiagonal to get the full Hamiltonian.

    V_eff(i) should contain the *total* effective potential at grid point i:
        V_eff(i) = -Z / r_i + v_H_eff^{(subshell)}(r_i)
    or any other local potential you want to use.

    We again work only with interior points i = 1..Nu-2.
*/
inline void add_potential_to_tridiagonal(const RadialGrid& g,
                                         const std::vector<double>& V_eff, // size Nu
                                         std::vector<double>& b)           // main diag of size N = Nu-2
{
    const int Nu = g.Nu;
    const int N  = Nu - 2;

    if ((int)V_eff.size() != Nu || (int)b.size() != N) {
        throw std::runtime_error("add_potential_to_tridiagonal: size mismatch");
    }

    for (int j = 0; j < N; ++j) {
        int i = j + 1; // interior index
        b[j] += V_eff[i];
    }
}

/*
    Utility: expand an interior-vector x[j], j=0..N-1 (N=Nu-2),
    into a full grid vector h[i], i=0..Nu-1, with Dirichlet
    boundary conditions h[0]=h[Nu-1]=0.

    Use this to reconstruct the full radial function:

        h_full = expand_to_full_grid(x)
        φ(r_i) = h_full[i] / r_i^{3/2}
*/
inline std::vector<double> expand_to_full_grid(const RadialGrid& g,
                                               const std::vector<double>& x)
{
    const int Nu = g.Nu;
    const int N  = Nu - 2;
    if ((int)x.size() != N) {
        throw std::runtime_error("expand_to_full_grid: size mismatch");
    }

    std::vector<double> h(Nu, 0.0);
    for (int j = 0; j < N; ++j) {
        int i = j + 1; // interior
        h[i] = x[j];
    }
    // h[0] and h[Nu-1] remain 0
    return h;
}

#endif // RADIAL_OPERATOR_H