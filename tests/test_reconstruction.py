import pytest
from hessym.hessian import *
from hessym.symmetry import octa_symm_group
from hessym.util import *

class TestNiReconstruction():
    """
    Tests that the force constants corresponding to each NN shell are correctly reconstructed after symmetry reduction
    by comparing to LAMMPS-computed Hessian for FCC Ni. For supercells of sizes 2x2x2, 3x3x3, and 4x4x4, this test:
    1) Performs symmetry reduction to identify the minimal parameter set to represent the Hessian
    2) Populates the symmetry-reduced Hessian by assigning values taken from the LAMMPS-computed Hessian to this minimal parameter set 
    3) Iterates over all nearest neighbor shells and check that all force constants in that shell are identical to the corresponding elements in the LAMMPS-computed Hessian
    """
    def test_222_supercell(self):
        print("Test that all interactions up to 3rd NN shell in reconstructed Hessian match LAMMPS Hessian for 2x2x2 supercell...")
        octa = octa_symm_group()
        positions, pbc_dim = load_lammps_positions("data/lammps_eam_positions_222")
        hessian = load_lammps_hessian("data/lammps_eam_dynmat_222.dat", 58.69)
        H = SymbolicHessian(positions, pbc_dim, octa, 3)
        H.validate(hessian)

    def test_333_supercell(self):
        print("Test that all interactions up to 7th NN shell in reconstructed Hessian match LAMMPS Hessian for 3x3x3 supercell...")
        octa = octa_symm_group()
        positions, pbc_dim = load_lammps_positions("data/lammps_eam_positions_333")
        hessian = load_lammps_hessian("data/lammps_eam_dynmat_333.dat", 58.69)
        H = SymbolicHessian(positions, pbc_dim, octa, 7)
        H.validate(hessian)

    def test_444_supercell(self):
        print("Test that all interactions up to 8th NN shell in reconstructed Hessian match LAMMPS Hessian for 4x4x4 supercell...")
        octa = octa_symm_group()
        positions, pbc_dim = load_lammps_positions("data/lammps_eam_positions_444")
        hessian = load_lammps_hessian("data/lammps_eam_dynmat_444.dat", 58.69)
        H = SymbolicHessian(positions, pbc_dim, octa, 8)
        H.validate(hessian)