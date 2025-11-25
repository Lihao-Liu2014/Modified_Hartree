#ifndef MODIFIED_HARTREE_H
#define MODIFIED_HARTREE_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include "radial_operator.h"
#include "eigen_tridiagonal_groundstate.h"

// Small helpers
namespace mh {

static const double PI = 3.14159265358979323846;

// One subshell result: n, l, N electrons, energy, radial functions
struct OrbitalResult {
    int n;
    int l;
    int N;                             // occupancy
    double energy;                     // ε_nl in atomic units
    std::vector<double> h;             // h(u) = r^{3/2} φ(r) on full grid (size Nu)
    std::vector<double> phi;           // φ(r) on full grid (size Nu)
};

// Internal subshell representation
struct Subshell {
    int n;
    int l;
    int N;
    double factor;                     // (N-1)/N
    double energy;
    std::vector<double> h;             // full-grid h(u)
};

// Cache for kinetic tridiagonal per ℓ
struct LChannel {
    int l;
    std::vector<double> a_kin;
    std::vector<double> b_kin;
    std::vector<double> c_kin;
};

class ModifiedHartreeAtom {
public:
    // Constructor:
    //  Z       : nuclear charge
    //  Nu      : number of u-grid points (>= 3)
    //  u_min   : minimum u (e.g. -8 or -10)
    //  u_max   : maximum u (e.g. +5 or +8)
    //  subshells: vector of (n,l,N_electrons), e.g. {(1,0,2),(2,0,2),(2,1,6)}
    ModifiedHartreeAtom(int Z,
                        int Nu,
                        double u_min,
                        double u_max,
                        const std::vector<std::tuple<int,int,int>>& subshells)
        : Z_(Z),
          grid_(make_log_grid(Nu, u_min, u_max))
    {
        if (Z_ <= 0) {
            throw std::runtime_error("ModifiedHartreeAtom: Z must be > 0");
        }
        if (subshells.empty()) {
            throw std::runtime_error("ModifiedHartreeAtom: need at least one subshell");
        }

        // Initialize subshell list and crude hydrogenic guesses for h(u)
        for (const auto& t : subshells) {
            int n = std::get<0>(t);
            int l = std::get<1>(t);
            int N = std::get<2>(t);
            if (n <= 0 || l < 0) {
                throw std::runtime_error("ModifiedHartreeAtom: invalid (n,l)");
            }
            if (N <= 0) {
                throw std::runtime_error("ModifiedHartreeAtom: N must be > 0");
            }

            Subshell s;
            s.n = n;
            s.l = l;
            s.N = N;
            s.factor = double(N - 1) / double(N);
            s.energy = 0.0;
            s.h.assign(grid_.Nu, 0.0);

            // crude hydrogenic-like initial guess:
            // φ(r) ~ r^l exp(-Z_eff r / n), with Z_eff ≈ Z
            double Z_eff = double(Z_);
            for (int i = 0; i < grid_.Nu; ++i) {
                double r  = grid_.r[i];
                double phi_guess = std::pow(r, l) * std::exp(-Z_eff * r / double(n));
                // h = r^{3/2} φ
                s.h[i] = std::pow(r, 1.5) * phi_guess;
            }

            // normalize in ∫ du h^2 = 1
            normalize_h(s.h);

            subshells_.push_back(s);
        }

        // Build kinetic tridiagonal for each distinct ℓ
        build_l_channels();

        vH_total_final_.assign(grid_.Nu, 0.0);
        vH_eff_final_.assign(subshells_.size(),
                            std::vector<double>(grid_.Nu, 0.0));        
    }

    // Run SCF loop
    void run_scf(int max_scf_iter = 1000,
                 double tol_energy  = 1e-8)
    {
        const int Nu = grid_.Nu;
        const int n_sub = (int)subshells_.size();

        // Pre-allocate arrays used each iteration
        std::vector<std::vector<double>> rho_sub(n_sub,
                                                 std::vector<double>(Nu, 0.0));
        std::vector<double> rho_total(Nu, 0.0);

        std::vector<std::vector<double>> vH_sub(n_sub,
                                                std::vector<double>(Nu, 0.0));
        std::vector<double> vH_total(Nu, 0.0);

        std::vector<std::vector<double>> vH_eff(n_sub,
                                                std::vector<double>(Nu, 0.0));

        std::vector<double> energies_old(n_sub, 0.0);

        std::vector<std::vector<double>> vH_eff_old(
            n_sub, std::vector<double>(Nu, 0.0));
        bool first_iter = true;

        for (int iter = 0; iter < max_scf_iter; ++iter) {

            // --- 1. Normalize all orbitals ---
            for (auto& s : subshells_) {
                normalize_h(s.h);
            }

            // --- 2. Densities ---
            std::fill(rho_total.begin(), rho_total.end(), 0.0);
            for (int s_idx = 0; s_idx < n_sub; ++s_idx) {
                auto& s = subshells_[s_idx];
                auto& rho = rho_sub[s_idx];

                for (int i = 0; i < Nu; ++i) {
                    double r = grid_.r[i];
                    double phi = s.h[i] / std::pow(r, 1.5); // φ(r) = h / r^{3/2}
                    rho[i] = s.N * phi * phi / (4.0 * PI);
                    rho_total[i] += rho[i];
                }
            }

            // --- 3. Hartree potentials (total and per subshell) ---
            compute_vH_from_rho(rho_total, vH_total);

            for (int s_idx = 0; s_idx < n_sub; ++s_idx) {
                compute_vH_from_rho(rho_sub[s_idx], vH_sub[s_idx]);
            }

            // --- 4. Effective modified Hartree potential for each subshell (raw) ---
            std::vector<std::vector<double>> vH_eff_new(
                n_sub, std::vector<double>(Nu, 0.0));

            for (int s_idx = 0; s_idx < n_sub; ++s_idx) {
                auto& s      = subshells_[s_idx];
                auto& vH_s   = vH_sub[s_idx];
                auto& vH_eff_s_new = vH_eff_new[s_idx];

                for (int i = 0; i < Nu; ++i) {
                    // vH_eff_new = vH_total - (1/N_s) * vH_self
                    vH_eff_s_new[i] = vH_total[i] - (1.0 / double(s.N)) * vH_s[i];
                }
            }

            // --- 4b. Linear mixing of effective potentials ---
            // vH_eff_mixed = (1 - alpha) * vH_eff_old + alpha * vH_eff_new

            double alpha = 0.3;  // try 0.2–0.5; smaller = more damping

            if (first_iter) {
                // no old potential yet: just take the new one
                for (int s_idx = 0; s_idx < n_sub; ++s_idx) {
                    vH_eff[s_idx]     = vH_eff_new[s_idx];
                    vH_eff_old[s_idx] = vH_eff_new[s_idx];
                }
                first_iter = false;
            } else {
                for (int s_idx = 0; s_idx < n_sub; ++s_idx) {
                    auto& v_old = vH_eff_old[s_idx];
                    auto& v_new = vH_eff_new[s_idx];
                    auto& v_mix = vH_eff[s_idx];

                    for (int i = 0; i < Nu; ++i) {
                        v_mix[i] = (1.0 - alpha) * v_old[i]
                                + alpha         * v_new[i];
                    }

                    // keep mixed as "old" for next iteration
                    v_old = v_mix;
                }
            }

            vH_total_final_ = vH_total;
            for (int s_idx = 0; s_idx < n_sub; ++s_idx) {
                vH_eff_final_[s_idx] = vH_eff[s_idx];
            }


            // save old energies BEFORE solving eigenproblems
            for (int s_idx = 0; s_idx < n_sub; ++s_idx) {
                energies_old[s_idx] = subshells_[s_idx].energy;
            }


            // --- 5. Solve for each subshell --- 
            for (int s_idx = 0; s_idx < n_sub; ++s_idx) {
            auto &s = subshells_[s_idx];

            std::vector<double> V_eff(Nu);
            for (int i = 0; i < Nu; ++i) {
                V_eff[i] = -double(Z_) / grid_.r[i] + vH_eff[s_idx][i];
            }

            const LChannel &chan = get_channel_for_l(s.l);
            std::vector<double> a = chan.a_kin;
            std::vector<double> b = chan.b_kin;
            std::vector<double> c = chan.c_kin;
            add_potential_to_tridiagonal(grid_, V_eff, b);

            NodeLabeledState state =
                solve_subshell_by_nodes(grid_, a, b, c, s.n, s.l);

            s.energy = state.energy;
            s.h      = state.h;
        }


            // --- check convergence ---
            double max_delta = 0.0;
            for (int s_idx = 0; s_idx < n_sub; ++s_idx) {
                double d = std::fabs(subshells_[s_idx].energy - energies_old[s_idx]);
                if (d > max_delta) max_delta = d;
            }

            std::cout << "SCF iter " << iter
                    << "  max Δε = " << max_delta << "\n" << "  energies: ";

            for (const auto& s : subshells_) {
                std::cout << s.energy << " ";
            }
            std::cout << "\n";

            if (iter > 0 && max_delta < tol_energy) {
                break;
            
            }
        }
    }

    // Output all orbitals (energies + h + φ)
    std::vector<OrbitalResult> get_orbitals() const
    {
        std::vector<OrbitalResult> res;
        res.reserve(subshells_.size());

        for (const auto& s : subshells_) {
            OrbitalResult o;
            o.n = s.n;
            o.l = s.l;
            o.N = s.N;
            o.energy = s.energy;
            o.h = s.h;
            o.phi.resize(grid_.Nu);

            for (int i = 0; i < grid_.Nu; ++i) {
                double r = grid_.r[i];
                o.phi[i] = s.h[i] / std::pow(r, 1.5);
            }
            res.push_back(std::move(o));
        }
        return res;
    }

    const RadialGrid& grid() const { return grid_; }

    const std::vector<double>& get_total_hartree_potential() const {
        return vH_total_final_;
    }

    // Effective Hartree (exchange-corrected) potential for subshell s_idx
    const std::vector<double>& get_effective_potential(int subshell_index) const {
        return vH_eff_final_.at(subshell_index);
    }

    int num_subshells() const { return static_cast<int>(subshells_.size()); }    

private:
    int Z_;
    RadialGrid grid_;
    std::vector<Subshell> subshells_;
    std::vector<LChannel> l_channels_;
    std::vector<double> vH_total_final_;                 // size Nu
    std::vector<std::vector<double>> vH_eff_final_;      // [n_sub][Nu]

    // --- Helpers ---

    void normalize_h(std::vector<double>& h)
    {
        double sum = 0.0;
        for (int i = 0; i < grid_.Nu; ++i) {
            sum += h[i] * h[i] * grid_.du; // ∫ du h^2
        }
        if (sum <= 0.0) return;
        double inv_norm = 1.0 / std::sqrt(sum);
        for (double& hi : h) hi *= inv_norm;
    }

    // Build kinetic tridiagonal for each distinct ℓ
    void build_l_channels()
    {
        // collect distinct l values
        std::vector<int> ls;
        for (const auto& s : subshells_) {
            if (std::find(ls.begin(), ls.end(), s.l) == ls.end()) {
                ls.push_back(s.l);
            }
        }
        std::sort(ls.begin(), ls.end());

        for (int l : ls) {
            LChannel chan;
            chan.l = l;
            build_kinetic_tridiagonal(grid_, l,
                                      chan.a_kin,
                                      chan.b_kin,
                                      chan.c_kin);
            l_channels_.push_back(std::move(chan));
        }
    }

    // Find the kinetic cache for a given ℓ
    const LChannel& get_channel_for_l(int l) const
    {
        for (const auto& chan : l_channels_) {
            if (chan.l == l) return chan;
        }
        throw std::runtime_error("get_channel_for_l: missing l-channel");
    }

    // Compute Hartree potential from a spherically symmetric density rho(r)
    // using: v_H(r) = 4π ∫ dr' r'^2 rho(r') / max(r,r')
    void compute_vH_from_rho(const std::vector<double>& rho,
                             std::vector<double>& vH) const
    {
        const int Nu = grid_.Nu;
        if ((int)rho.size() != Nu) {
            throw std::runtime_error("compute_vH_from_rho: size mismatch");
        }
        vH.assign(Nu, 0.0);

        for (int i = 0; i < Nu; ++i) {
            double ri = grid_.r[i];
            double sum = 0.0;

            for (int j = 0; j < Nu; ++j) {
                double rj = grid_.r[j];
                double r_great = (ri > rj) ? ri : rj;
                double dr = rj * grid_.du; // dr = r du
                sum += (rj * rj) * rho[j] / r_great * dr;
            }
            vH[i] = 4.0 * PI * sum;
        }
    }

    // Extract interior points (i=1..Nu-2) from full-grid h into x
    void extract_interior(const std::vector<double>& h_full,
                          std::vector<double>& x) const
    {
        const int Nu = grid_.Nu;
        const int N  = Nu - 2;
        x.resize(N);
        for (int j = 0; j < N; ++j) {
            int i = j + 1; // interior
            x[j] = h_full[i];
        }
    }
};

} // namespace mh

#endif // MODIFIED_HARTREE_H
