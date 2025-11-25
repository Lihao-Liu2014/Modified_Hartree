#ifndef EIGENSOLVER_BY_NODES_H
#define EIGENSOLVER_BY_NODES_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

#include "radial_operator.h"

// You need Eigen installed and available in the include path.
#include <Eigen/Dense>

namespace mh {

// Info for one eigenstate on the grid
struct NodeLabeledState {
    double energy;                  // eigenvalue (a.u.)
    int nodes;                      // number of radial nodes
    std::vector<double> h;          // h(u) on full grid (size Nu)
    std::vector<double> phi;        // φ(r) on full grid (size Nu)
};

// Count radial nodes (sign changes) in φ(r)
// threshold: ignore |φ| < threshold to avoid noise at origin/tail
inline int count_radial_nodes(const std::vector<double>& phi,
                              double threshold = 1e-6)
{
    const int Nu = (int)phi.size();
    int nodes = 0;

    // Skip leading tiny values near origin
    int i = 0;
    while (i < Nu && std::fabs(phi[i]) < threshold) ++i;
    if (i >= Nu) return 0;

    double prev = phi[i];
    for (++i; i < Nu; ++i) {
        double val = phi[i];
        if (std::fabs(val) < threshold) continue;

        if ((prev > 0.0 && val < 0.0) || (prev < 0.0 && val > 0.0)) {
            ++nodes;
            prev = val;
        } else {
            prev = val;
        }
    }
    return nodes;
}

/*
    Solve the tridiagonal Hamiltonian for a given ℓ and effective potential,
    compute all eigenpairs, and select the bound state corresponding to
    subshell (n, l) by counting radial nodes.

    Inputs:
      grid : RadialGrid (with Nu points)
      a,b,c: tridiagonal coefficients for the Hamiltonian on *interior* points
             size(b) = N = Nu-2, size(a) = size(c) = N-1
      n, l : target subshell quantum numbers

    Output:
      NodeLabeledState containing energy ε, h(u), φ(r), and node count.

    Throws if it cannot find a bound state with the desired node count.
*/
inline NodeLabeledState
solve_subshell_by_nodes(const RadialGrid& grid,
                        const std::vector<double>& a,
                        const std::vector<double>& b,
                        const std::vector<double>& c,
                        int n, int l)
{
    const int Nu = grid.Nu;
    const int N  = (int)b.size();

    if (N != Nu - 2 || (int)a.size() != N - 1 || (int)c.size() != N - 1) {
        throw std::runtime_error("solve_subshell_by_nodes: size mismatch");
    }
    if (n <= 0 || l < 0 || n <= l) {
        throw std::runtime_error("solve_subshell_by_nodes: invalid (n,l)");
    }

    const int target_nodes = n - l - 1;

    // --- 1. Build dense symmetric matrix H from tridiagonal (a,b,c) ---

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N);
    for (int j = 0; j < N; ++j) {
        H(j, j) = b[j];
        if (j < N - 1) {
            H(j, j + 1) = c[j];
            H(j + 1, j) = a[j];  // symmetric
        }
    }

    // --- 2. Diagonalize H: H v_k = λ_k v_k ---

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("solve_subshell_by_nodes: eigen decomposition failed");
    }

    Eigen::VectorXd evals = es.eigenvalues();      // λ_k, sorted ascending
    Eigen::MatrixXd evecs = es.eigenvectors();     // columns are v_k

    // --- 3. Loop over eigenstates, compute nodes, and pick the right one ---

    int best_index   = -1;
    double best_energy = 0.0;

    for (int k = 0; k < N; ++k) {
        double E = evals[k];

        // Only consider bound states
        if (E >= 0.0) continue;

        // Extract interior eigenvector
        std::vector<double> x_k(N);
        for (int j = 0; j < N; ++j) {
            x_k[j] = evecs(j, k);
        }

        // Expand to full grid h[i] with Dirichlet boundaries
        std::vector<double> h_full = expand_to_full_grid(grid, x_k);

        // Convert to φ(r) = h / r^{3/2}
        std::vector<double> phi_full(Nu);
        for (int i = 0; i < Nu; ++i) {
            double r = grid.r[i];
            phi_full[i] = h_full[i] / std::pow(r, 1.5);
        }

        // Count radial nodes
        int nodes = count_radial_nodes(phi_full);

        if (nodes == target_nodes) {
            // Among states with correct node count, pick the most bound (smallest E)
            if (best_index < 0 || E < best_energy) {
                best_index   = k;
                best_energy  = E;
            }
        }
    }

    if (best_index < 0) {
        throw std::runtime_error("solve_subshell_by_nodes: "
                                 "no bound state with desired node count "
                                 "(target_nodes = " + std::to_string(target_nodes) + ")");
    }

    // --- 4. Build and normalize the chosen state ---

    NodeLabeledState result;
    result.energy = best_energy;

    // Fetch chosen eigenvector
    std::vector<double> x_best(N);
    for (int j = 0; j < N; ++j) {
        x_best[j] = evecs(j, best_index);
    }

    // Full h(u)
    result.h = expand_to_full_grid(grid, x_best);

    // Normalize in ∫ du h^2 = 1
    double sum = 0.0;
    for (int i = 0; i < Nu; ++i) {
        sum += result.h[i] * result.h[i] * grid.du;
    }
    if (sum > 0.0) {
        double inv_norm = 1.0 / std::sqrt(sum);
        for (int i = 0; i < Nu; ++i) result.h[i] *= inv_norm;
    }

    // Build φ(r) on full grid
    result.phi.resize(Nu);
    for (int i = 0; i < Nu; ++i) {
        double r = grid.r[i];
        result.phi[i] = result.h[i] / std::pow(r, 1.5);
    }

    // Count nodes of the normalized state (for info)
    result.nodes = count_radial_nodes(result.phi);

    return result;
}

} // namespace mh

#endif // EIGENSOLVER_BY_NODES_H
