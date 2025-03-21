# `hessym`: A package for Hessian crystal symmetry analysis

The **Hessian** matrix of a crystalline material contains vital information needed to describe the thermodynamics of the system; however, because Hessian size scales
quadratically with the number of atoms in a crystal, direct calculations of the Hessian are computationally infeasible for all but the simplest of supercells.

The symmetry group of a crystal places constraints on the allowed numerical values of each entry in the Hessian, which can drastically reduce the number of independent parameters
in the matrix. If these independent parameters are identified, we need only explicitly compute a subset of
force constants in order to fully populate the Hessian.

`hessym` is a lightweight package for performing symmetry analysis and efficient approximations of Helmholtz free energy, phonon dispersions, and phonon density of states via reconstructed Hessians.

Installation:
```
# Install prerequisites
pip install -r requirements.txt

# Build hessym package
pip install -e .
```

To run test cases, `cd` to `tests` directory:
```
python -m pytest -s
```