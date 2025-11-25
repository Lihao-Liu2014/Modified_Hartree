#include <iostream>
#include <fstream>
#include <iomanip>
#include <tuple>
#include <vector>
#include <string>

#include "modified_hartree.h"

using namespace std;

// Helper: run one atom and write results
void run_atom(int Z,
              const std::string& name,
              int Nu,
              double u_min,
              double u_max,
              const std::vector<std::tuple<int,int,int>>& subshells)
{
    cout << "========================================\n";
    cout << " Atom: " << name << "  (Z = " << Z << ")\n";
    cout << "========================================\n";

    // Build atom and run SCF
    mh::ModifiedHartreeAtom atom(Z, Nu, u_min, u_max, subshells);
    atom.run_scf(1000, 1e-8);

    auto orbitals = atom.get_orbitals();
    const auto& grid = atom.grid();

    // Print energies
    cout << fixed << setprecision(8);
    for (const auto& o : orbitals) {
        double removal_e_au = -o.energy;           // Koopmans-like approximation
        double removal_e_eV = removal_e_au * 27.211386245988; // a.u. → eV

        cout << "  Subshell n=" << o.n
             << " l=" << o.l
             << " N=" << o.N
             << "   ε = " << setw(14) << o.energy << " a.u."
             << "   I ≈ " << setw(14) << removal_e_eV << " eV\n";
    }

    // Write radial wavefunctions for each subshell to files
    // File format: r   phi(r)
    for (const auto& o : orbitals) {
        // Build filename like: He_n1_l0.dat
        std::string filename = name + "_n" + std::to_string(o.n)
                                     + "_l" + std::to_string(o.l) + ".dat";
        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "Could not open file " << filename << " for writing\n";
            continue;
        }

        ofs << "# " << name << "  Z=" << Z
            << "  n=" << o.n << "  l=" << o.l
            << "  N=" << o.N << "\n";
        ofs << "# energy (a.u.) = " << o.energy << "\n";
        ofs << "# columns: r   phi(r)\n";

        ofs << std::scientific << std::setprecision(10);
        for (int i = 0; i < grid.Nu; ++i) {
            ofs << grid.r[i] << "  " << o.phi[i] << "\n";
        }
        ofs.close();
        cout << "  Wrote radial wavefunction to " << filename << "\n";
    }

    // Write total Hartree potential and effective potentials for each subshell to files
    for (std::size_t s_idx = 0; s_idx < orbitals.size(); ++s_idx) {
        const auto& o = orbitals[s_idx];
        const auto& V_H = atom.get_total_hartree_potential();
        const auto& Veff = atom.get_effective_potential(static_cast<int>(s_idx));
        // Write total Hartree potential
        {
            // Build filename like: He_VH.dat
            std::string filename = name + "_VH.dat";
            std::ofstream ofs(filename);
            if (!ofs) {
                std::cerr << "Could not open file " << filename << " for writing\n";
                continue;
            }

            ofs << "# " << name << "  Z=" << Z << "\n";
            ofs << "# columns: r   V_H(r)\n";

            ofs << std::scientific << std::setprecision(10);
            for (int i = 0; i < grid.Nu; ++i) {
                ofs << grid.r[i] << "  " << V_H[i] << "\n";
            }
            ofs.close();
            cout << "  Wrote total Hartree potential to " << filename << "\n";
        }


        // Build filename like: He_Veff_n1_l0.dat
        std::string filename = name + "_Veff_n" + std::to_string(o.n)
                                     + "_l" + std::to_string(o.l) + ".dat";
        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "Could not open file " << filename << " for writing\n";
            continue;
        }

        ofs << "# " << name << "  Z=" << Z
            << "  n=" << o.n << "  l=" << o.l
            << "  N=" << o.N << "\n";
        ofs << "# columns: r   V_eff(r)\n";

        ofs << std::scientific << std::setprecision(10);
        for (int i = 0; i < grid.Nu; ++i) {
            ofs << grid.r[i] << "  " << Veff[i] << "\n";
        }
        ofs.close();
        cout << "  Wrote effective potential to " << filename << "\n";
    }

    cout << "\n";
}

int main()
{
    // Grid parameters (same for all three atoms, you can tune these)
    int    Nu    = 1500;   // number of grid points
    double u_min = -8.0;   // ln(r_min)
    double u_max =  6.0;   // ln(r_max)

    {
        std::vector<std::tuple<int,int,int>> subshells = {
            {1, 0, 2},  // 1s^2
            {2, 0, 2},  // 2s^2
            {2, 1, 6},   // 2p^6
            {3, 0, 2},  // 3s^2
        };
        run_atom(12, "Mg", Nu, u_min, u_max, subshells);
    }


    // --- Helium: He, Z=2, configuration 1s^2 ---
    {
        std::vector<std::tuple<int,int,int>> subshells = {
            {1, 0, 2}   // 1s^2
        };
        run_atom(2, "He", Nu, u_min, u_max, subshells);
    }

    // --- Neon: Ne, Z=10, configuration 1s^2 2s^2 2p^6 ---
    {
        std::vector<std::tuple<int,int,int>> subshells = {
            {1, 0, 2},  // 1s^2
            {2, 0, 2},  // 2s^2
            {2, 1, 6}   // 2p^6
        };
        run_atom(10, "Ne", Nu, u_min, u_max, subshells);
    }

    // --- Argon: Ar, Z=18, configuration 1s^2 2s^2 2p^6 3s^2 3p^6 ---
    {
        std::vector<std::tuple<int,int,int>> subshells = {
            {1, 0, 2},  // 1s^2
            {2, 0, 2},  // 2s^2
            {2, 1, 6},  // 2p^6
            {3, 0, 2},  // 3s^2
            {3, 1, 6}   // 3p^6
        };
        run_atom(18, "Ar", Nu, u_min, u_max, subshells);
    }



    return 0;
}
