"""
Utilities for Hessian symmetry reduction calculation
"""
import numpy as np
import sympy
import string
from tqdm import tqdm
import yaml

class SymbolicHessian:
    """
    Class for assigning symbolic labels to non-zero entries of a Hessian matrix as constrained by the symmetry relations of the point group
    """
    def __init__(self, positions, pbc_dim, symm_group, n_neighbors):
        """
        Initialize a symbolic Hessian for a crystal belonging to `symm_group` with atoms placed at `positions` within a (pbc_dim)**3 supercell
        Treat all Hessian elements corresponding to atoms that are more than `n_neighbors` neighboring shells apart as 0
        Params:
            positions (np.array): Atomic positions. If the SymbolicHessian is to be validated against a numerical Hessian computed by LAMMPS/VASP, these positions should correspond to the exact simulation setup
            pbc_dim (float): Length over which to apply periodic boundary conditions (i.e., supercell length)
            symm_group (symm_group): Symmetry group object describing the crystal
            n_neighbors (int): Number of nearest neighbors to treat as having non-zero force constants
        Attr:
            self.positions (np.array): Atomic positions
            self.n_atoms (int): Number of atoms in the system
            self.symm_group (symm_group): Symmetry group object describing the crystal
            self.n_neighbors (int): Number of nearest neighbors treated as having non-zero force constnats
            self.pbc_dim (float): Length scale of periodic boundary conditions
            self.threshold_distances (np.array): Array containing the distance between an atom and its first NNs, second NNs, etc.
            self.threshold_distance (float): Maximum distance between atoms for their interaction to be described by non-zero force constant
            self.hessian (np.array): 3`n_atoms x 3`n_atoms` array containing symmetry-constrained symbolic force constants
            self.relative_neighbor_positions (np.array): Array of position vectors between any given atom and all of its neighbors within `self.threshold_distance`, accounting for PBC
            self.relative_pos_to_submatrix_mapping (SymbolMapping): Mapper relating an atom to the set of 3x3 symbolic submatrices describing the force constants it shares with its neighbors
            self.parameters (set): Set of independent symbolic parameters in the Hessian
            self.mask (np.array): Boolean mask of size 3`n_atoms x 3`n_atoms` that is true at Hessian indices corresponding to interactions within the `n_neighbors` shell
            self.nn_mask (np.array): Integer array of size 3`n_atoms x 3`n_atoms` that contains the neighbor number relating atoms at the corresponding Hessian indices (eg 1st NN, 2nd NN, etc). Interactions between atoms that are greater than `n_neighbors` apart are masked with -1
            self.recon (np.array): Array containing the most recently-reconstructed numerical Hessian 
        """
        self.positions = positions
        self.n_atoms = len(self.positions)
        self.symm_group = symm_group
        self.n_neighbors = n_neighbors
        self.pbc_dim = pbc_dim
        self.threshold_distances = self.get_nearest_neighbor_distances(self.positions, self.n_neighbors, self.pbc_dim)
        self.threshold_distance = self.threshold_distances[-1]
        self.mask = np.full((3*self.n_atoms, 3*self.n_atoms), fill_value=False, dtype=bool)
        self.nn_mask = np.full((3*self.n_atoms, 3*self.n_atoms), fill_value=-1, dtype=int) # -1 == no NN identified

        # We treat the force constants between atoms that are outside of the `n_neighbor` radius as 0
        self.hessian = np.full((3*self.n_atoms, 3*self.n_atoms), fill_value=0, dtype=object)

        # Compute all possible NN position vectors, considering only NN within `n_neighbors`
        self.relative_neighbor_positions = self.get_relative_neighbor_positions(self.positions[0, :])
        # Assign a set of mapping labels for all Hessian submatrices corresponding to these position vectors
        self.relative_pos_to_submatrix_mapping = SymbolMapping(self.symm_group, self.relative_neighbor_positions, self.threshold_distances)
        self.parameters = self.relative_pos_to_submatrix_mapping.parameters
        
        # Fill the Hessian. Iterate over all atoms to 1) treat the current atom as the central reference point 
        # 2) identify its neighbors and the correct Hessian submatrices for each 3) insert into the appropriate position in the Hessian
        # TODO: Hessian is symmetric; only need to iterate over top diagonal
        for atom_idx, pos in enumerate(self.positions):
            hessian_submatrices, hessian_nearest_neighbor_nums, insertion_idx = self.get_submatrices_for_all_neighbors(pos)
            self.hessian[3*atom_idx:3*atom_idx+3, insertion_idx] = hessian_submatrices
            self.nn_mask[3*atom_idx:3*atom_idx+3, insertion_idx] = hessian_nearest_neighbor_nums
            self.mask[3*atom_idx:3*atom_idx+3, insertion_idx] = True

        # This attribute stores the most recent reconstruction generated by the object
        self.recon = np.zeros(self.hessian.shape)

    def __repr__(self):
        return np.array_repr(self.hessian)
        
    def apply_pbc(self, relative_positions, box_dim):
        """
        Applies periodic boundary conditions to atomic positions measured from the central atom of the simulation box
        Params:
            relative_postions (np.array): Atomic positions rescaled to place the central atom at (0, 0, 0)
            box_dim: (np.array): Length of simulation box along the x, y, and z axes 
        Return:
            relative_positions (np.array): Position coordinates of all atoms in periodic simulation box 
        """
        pbc_relative_positions = np.zeros(relative_positions.shape)
        for dim in range(3):
            dim_length = box_dim[dim]
            pbc_relative_positions[:, dim] = relative_positions[:, dim] - dim_length*np.round(relative_positions[:, dim]/dim_length)
        return pbc_relative_positions

    
    def get_nearest_neighbor_distances(self, positions, n_neighbors, box_dim):
        """
        Returns the distances between an atom and its set of first nearest neighbors, second nearest neighbors, etc. up to `n_neighbors`
        Params:
            positions (np.array): Position coordinates of all atoms in simulation box
            n_neighbors (int): Maximum nearest neighbor shell to consider (i.e., 3rd NN, 4th NN, etc)
            box_dim (np.array): Length of simulation box along the x, y, and z axes 
        Return:
            neighbor_distances (np.array): Array containing the distance between an atom and its first NN, second NN, etc.
        """
        central_atom = positions[0, :]
        relative_positions = positions - central_atom
        relative_positions = self.apply_pbc(relative_positions, box_dim)
        distances = np.linalg.norm(relative_positions, axis=1)
        
        # Numerical error will cause some equal distances to be mistakenly treated as unique
        # Correct this by scaling, rounding, then rescaling
        unique_distances = np.unique(np.round(distances*1e6)/1e6)
        
        # Index beginning from 1 so the central atom is not counted as its own neighbor
        neighbor_distances = np.sort(unique_distances)[1:n_neighbors+1]

        return neighbor_distances + 1e-6

    def get_relative_neighbor_positions(self, central_pos, return_neighbor_idx=False):
        """
        Returns the positions of atoms in the supercell that are `self.n_neighbors`-nearest or closer neighbors of the atom at `central_pos`,
        expressed relative to the coordinates of `central_pos` and considering periodic boundary conditions
        Params:
            central_pos (np.array): Position coordinate of central atom
            return_neighbor_idx (bool): If True, return the indices at which the central atom's neighbors are stored in `self.positions`
        Return:
            neighbor_positions_relative_to_central (np.array): Array of position coordinates for the central atom's neighbors, measured wrt the central atom at (0, 0, 0)
        """
        positions_relative_to_central = self.apply_pbc(self.positions - central_pos, self.pbc_dim)
        distances_relative_to_central = np.linalg.norm(positions_relative_to_central, axis=1)
        neighbor_idx = np.where(distances_relative_to_central <= self.threshold_distance)[0]
        neighbor_positions_relative_to_central = positions_relative_to_central[neighbor_idx, :]
        if return_neighbor_idx:
            return neighbor_idx, neighbor_positions_relative_to_central
        return neighbor_positions_relative_to_central
    
    def get_submatrices_for_all_neighbors(self, central_pos):
        """
        Returns the non-zero Hessian submatrices for the force constants between the atom at `central_pos` and all other atoms in the supercell, as well
        as the indices at which these submatrices should be inserted into `self.hessian`
        Params:
            central_pos (np.array): Position coordinate of central atom
        Return:
            stacked_relative_neighbor_submatrices (np.array): Set of all non-zero Hessian submatrices for force constants involving the atom at `central_pos`, formatted for insertion into `self.hessian`
            hessian_idx (list): List of column indices at which the values of `stacked_relative_neighbor_submatrices` should be inserted into `self.hessian`
        """
        neighbor_idx, relative_neighbor_positions = self.get_relative_neighbor_positions(central_pos, return_neighbor_idx=True)
        
        # NumPy indexing magic to cleanly get a list of indices for neighboring atom submatrices in Hessian
        atom_idx_to_hessian_idx = np.arange(3*self.n_atoms).reshape(3, self.n_atoms, order="F")
        hessian_idx = list(atom_idx_to_hessian_idx[:, neighbor_idx].flatten(order="F"))
        # Now, get the submatrices themselves
        relative_neighbor_submatrices = np.apply_along_axis(self.relative_pos_to_submatrix_mapping.get_symbolic_submatrix, 1, relative_neighbor_positions)
        relative_neighbor_nums = [np.ones((3, 3))*self.get_parameter_nearest_neighbor_num(list(sympy.Matrix(submatrix).free_symbols)[0]) for submatrix in relative_neighbor_submatrices]
        stacked_relative_neighbor_submatrices = np.hstack(relative_neighbor_submatrices)
        stacked_nearest_neighbor_nums = np.hstack(relative_neighbor_nums)

        return stacked_relative_neighbor_submatrices, stacked_nearest_neighbor_nums, hessian_idx
    
    def get_nn_hessian_idx(self, central_pos, n_neighbor):
        """
        Returns the Hessian indices at which elements describing the interactions between the atom at `central_pos` and its `n_neighbor`th NNs can be found
        Args:
            central_pos (np.array): Position vector of central atom to consider
            n_neighbor (int): Nearest neighbor number of interaction to consider (i.e. 1st NN, 2nd NN, etc)
        Return:
            hessian_row_idx (np.array): Hessian row indices for desired interactions
            hessian_col_idx (np.array): Hessian col indices for desired interactions
        """
        positions_relative_to_central = self.apply_pbc(self.positions - central_pos, self.pbc_dim)
        distances_relative_to_central = np.linalg.norm(positions_relative_to_central, axis=1)
        neighbor_idx = np.where(np.isclose(distances_relative_to_central, self.threshold_distances[n_neighbor-1]))[0]
        atom_idx_to_hessian_idx = np.arange(3*self.n_atoms).reshape(3, self.n_atoms, order="F")
        hessian_col_idx = np.array(list(atom_idx_to_hessian_idx[:, neighbor_idx].flatten(order="F")))
        atom_idx = np.where(np.all(self.positions == central_pos, axis=1))[0][0]
        hessian_row_idx = np.array([3*atom_idx, 3*atom_idx+1, 3*atom_idx+2])[:, np.newaxis]
        return hessian_row_idx, hessian_col_idx


    def get_parameter_indices(self, verbose=False):
        """
        Returns the Hessian indices at which each independent symbolic parameter can be found (since each parameter is located at many positions in `self.hessian`, arbitrarily choose an index)
        Return:
            param_map (dict): Dictionary mapping between symbolic parameters and their index within a Hessian matrix
            verbose (bool, optional): If True, print tqdm progress bar
        """
        param_map = {}
        for param in tqdm(self.parameters, disable=not verbose, desc="Extracting numerical values from Hessian"):
            pos_indices = np.where(self.hessian[:3, :]== param) 
            param_map[param] = (pos_indices[0][0], pos_indices[1][0])
            
            # The negative of each parameter might not always exist in the Hessian; only retrieve indices if it does
            try:
                neg_indices = np.where(self.hessian[:3, :] == -param)
                param_map[-param] = (neg_indices[0][0], neg_indices[1][0])
            except:
                pass
            
        return param_map
    
    def get_parameter_nearest_neighbor_num(self, parameter):
        """
        Given a parameter symbol, returns the nearest neighbor shell for which the parameter describes interactions
        Params:
            parameter (sympy.Symbol): Symbolic parameter of Hessian
        """
        self_interaction_parameter = chr(self.n_neighbors+97)
        parameter_symbol = str(parameter)[0]
        if parameter_symbol != self_interaction_parameter:
            return ord(parameter_symbol)-96
        else:
            return 0
    
    def reconstruct_hessian(self, true_hessian=None, symbol_mapping=None, verbose=False):
        """
        Reconstructs an approximate Hessian by substituting either 1) a subset of values from `true_hessian` into the symbolic representation
        or 2) a dictionary of parameter value pairs given by `symbol_mapping`
        Params:
            true_hessian (np.array, optional): A 3`self.n_atoms` x 3`self.n_atoms` numerical Hessian computed by LAMMPS/VASP
            symbol_mapping (dict, optional): A dictionary mapping each parameter in `self.parameters` to a numerical value
            verbose (bool, optional): If True, print tqdm progress bar
        Return:
            reconstructed_hessian (np.array): A Hessian matrix reconstructed using only the parameters contained in `self.parameters`
            symbol_mapping (dict): A dictionary mapping each parameter in `self.parameters` to a numerical value (if `symbol_mapping` was passed as an argument, simply repeats the mapping)
        """
        assert true_hessian is not None or symbol_mapping is not None, "Must pass symbol mapping information to reconstruct Hessian"
        reconstructed_hessian = np.zeros(self.hessian.shape)

        if true_hessian is not None:
            parameter_indices_map = self.get_parameter_indices(verbose=verbose)
            symbol_mapping = {}
            
            for param in parameter_indices_map:
                val = true_hessian[parameter_indices_map[param]]
                symbol_mapping[param] = val
            atom_idx = 0
            for pos in tqdm(self.positions, disable=not verbose, desc="Populating Hessian"):
                hessian_submatrices, _, insertion_idx = self.get_submatrices_for_all_neighbors(pos)
                # TODO: Refactor – Sympy.subs is convenient, but slow
                reconstructed_hessian[3*atom_idx:3*atom_idx+3, insertion_idx] = sympy.Matrix(hessian_submatrices).subs(symbol_mapping)
                atom_idx += 1

        elif symbol_mapping is not None:
            assert all(param in symbol_mapping for param in self.parameters), f"Reconstruction requires numeric values for all independent parameters: {self.parameters}"
            atom_idx = 0
            for pos in tqdm(self.positions, disable=not verbose, desc="Populating Hessian"):
                hessian_submatrices, _, insertion_idx = self.get_submatrices_for_all_neighbors(pos)
                # TODO: Refactor – Sympy.subs is convenient, but slow
                reconstructed_hessian[3*atom_idx:3*atom_idx+3, insertion_idx] = sympy.Matrix(hessian_submatrices).subs(symbol_mapping)
                atom_idx += 1

        # Apply translational invariance condition. Treat the self-interaction as an independent term and set it equal to the negative sum over 
        # the submatrices of all other atoms
        for atom_idx in range(self.n_atoms):
            # Indices of all columns representing non-self-interactions
            non_self_interaction_idx =np.delete(np.arange(3*self.n_atoms), np.arange(3*atom_idx, 3*atom_idx+3))
            # Sum over 3x3 submatrices of all non-self-interactions for atom at `atom_idx`
            trans_invariance_sum = np.sum(reconstructed_hessian[3*atom_idx:(3*atom_idx+3), non_self_interaction_idx].reshape(3, 3, self.n_atoms-1, order="F"), axis=-1)
            # Insert into self-interaction submatrix
            reconstructed_hessian[3*atom_idx:3*atom_idx+3, 3*atom_idx:3*atom_idx+3] = -trans_invariance_sum

        self.recon = reconstructed_hessian
        print("Hessian reconstruction attribute updated")
        return reconstructed_hessian, symbol_mapping
    
    def compute_free_energy(self, atomic_mass, T, true_hessian=None, symbol_mapping=None, mode="classical", verbose=False):
        """
        Computes free energy in eV using Hessian reconstructed from numeric data given by `true_hessian` or `symbol_mapping`. If no numeric data is passed, computes free energy
        using the Hessian data stored in `self.recon`, if available
        Assumes Hessian is measured in eV/A^2, atomic masses are measured in amu
        Params:
            atomic_mass (float): Atomic mass in amu
            T (float or np.array of floats): Temperature in K
            true_hessian (np.array, optional): A 3`self.n_atoms` x 3`self.n_atoms` numerical Hessian computed by LAMMPS/VASP
            symbol_mapping (dict, optional): A dictionary mapping each parameter in `self.parameters` to a numerical value
            mode (str): String "classical" or "quantum" free energy calculation
            verbose (bool): 
        Return:
            F (float): Helmholtz free energy in eV
            F_per_atom (float): Helmholtz free energy per atom
        """
        assert true_hessian is not None or symbol_mapping is not None or self.recon is not None, "Free energy calculation requires numerical Hessian reconstruction"
        if true_hessian is None and symbol_mapping is None:
            reconstructed_hessian = np.copy(self.recon)
        else:
            reconstructed_hessian, _ = np.copy(self.reconstruct_hessian(true_hessian, symbol_mapping, verbose=verbose))
        kB = 8.617343e-5  # eV/K
        hbar = 6.58211899e-16 # eV s
        Na = 6.02214179e23    # mol^-1

        # Convert to kg/s^2 (1.6*10**-19 * 10**20)
        reconstructed_hessian *= 16.022
        # Eigendecomposition. Eigenvalues are in units of kg/s^2
        eigenval, eigenvec = np.linalg.eigh(reconstructed_hessian)
        # Eigenfrequencies in units of s^-2 (convert amu to kg). Neglect 3 zero eigenmodes
        omega = np.sqrt(eigenval[3:]/(atomic_mass*1.66054*10**(-27)))
        
        if not np.isscalar(T):
            # Reformat temperature array to allow vectorized calculation
            T_arr = np.repeat(T, len(omega), axis=0).reshape(len(T), len(omega))
        else:
            T_arr = T

        # Helmholtz free energy in units of eV
        if mode == "classical":
            F = -kB*T*np.sum(np.log(kB*T_arr/(hbar*omega)), axis=-1)
        elif mode == "quantum":
            F = 0.5*hbar*np.sum(omega) + kB*T*np.sum(np.log(1-np.exp(-hbar*omega/(kB*T_arr))), axis=-1)

        F_per_atom = F/self.n_atoms
        return F, F_per_atom

    def validate(self, true_hessian):
        """
        For testing. Beginning with 1st nearest neighbors, iterates over NN shells to check for equality between the reconstructed Hessian and `true_hessian`
        Params:
            true_hessian (np.array): A 3`self.n_atoms` x 3`self.n_atoms` numerical Hessian computed by LAMMPS/VASP
        """
        # Mask all entries in the true Hessian that are outside the NN window being considered
        nn_hessian = np.copy(true_hessian)
        nn_hessian[~self.mask] = 0

        recon, recon_mapping = self.reconstruct_hessian(true_hessian, verbose=True)
        recon_correct_for_all_nn = []
        # Note: enforcing translational invariance will cause inconsistency in the self-interaction; skip this assessment for now
        for nn in range(1, self.n_neighbors+1):
            nn_idx = np.where(self.nn_mask == nn)
            recon_vals, true_vals = recon[nn_idx], true_hessian[nn_idx]
            recon_correct_for_nn = np.sum(~np.isclose(recon_vals, true_vals)) == 0
            nn_parameter_map = {symbol:recon_mapping[symbol] for symbol in recon_mapping if self.get_parameter_nearest_neighbor_num(symbol)==nn}
            # print(f"{str(nn) + ' NN ' if nn > 0 else 'Self-'}interactions parameters: {nn_parameter_map}")
            # print(f"{str(nn) + ' NN ' if nn > 0 else 'Self-'}interactions correct: {test_case_result(recon_correct_for_nn)}")
            assert recon_correct_for_nn, f"Reconstruction incorrect for {nn} NN shell"
            recon_correct_for_all_nn.append(recon_correct_for_nn)
        # print("-----------------------------------------------------------------------------------")
        # print(f"Reconstruction fully correct: {test_case_result(np.all(recon_correct_for_all_nn))}")
        assert np.all(recon_correct_for_all_nn), "Reconstruction contains errors"
        return np.all(recon_correct_for_all_nn)

    def write_phonopy(self, load_filepath, write_filepath, true_hessian=None):
        """
        Writes symmetry-reduced Hessian matrix to Phonopy input YAML for use in phonon DOS and dispersion calculations
        Args:
            load_filepath (str): Filepath to `phonopy_disp.yaml` file used to generate unreduced Hessian
            write_filepath (str): Filepath for writing symmetry-reduced Hessian
            true_hessian (np.array): A 3`self.n_atoms` x 3`self.n_atoms` numerical Hessian computed by LAMMPS/VASP
        """
        assert true_hessian is not None or self.recon is not None, "Phonopy deck generation requires numerical Hessian reconstruction"
        if true_hessian is not None:
            recon, recon_mapping = self.reconstruct_hessian(true_hessian, verbose=True)
        else:
            recon = self.recon

        formatted_force_constants = []
        # TODO: Refactor to be more efficient
        for row in range(self.n_atoms):
            for col in range(self.n_atoms):
                formatted_force_constants.append(recon[3*row:3*row+3, 3*col:3*col+3].tolist())

        with open(load_filepath) as f:
            yaml_file = yaml.safe_load(f)

        yaml_n_atoms = len(yaml_file["supercell"]["points"])
        assert len(formatted_force_constants) == yaml_n_atoms**2, "Reduced Hessian size must match size of Phonopy input"

        # If force constants were computed using LAMMPS, we overwrite a displacement-generating Phonopy YAML
        # This requires massaging the YAML structure slightly
        if "displacements" in yaml_file.keys():
            yaml_file.pop("displacements")
        if "force_constants" not in yaml_file.keys():
            yaml_file["force_constants"] = {"format":"full", "shape":[yaml_n_atoms, yaml_n_atoms]}

        yaml_file["force_constants"]["elements"] = formatted_force_constants

        with open(write_filepath, "w") as f:
            yaml.dump(yaml_file, f, default_flow_style=None, sort_keys=False)

        print(f"Symmetry-reduced force constants written to {write_filepath}")
        print(f"Generate phonon DOS by: `phonopy-load -p --readfc {write_filepath}`")


class SymbolMapping:
    """
    Stores labeled submatrices of the Hessian for each possible interaction between neighboring atoms of type i and j
    """
    def __init__(self, symm_group, relative_neighbor_positions, threshold_distances):
        """
        Initialize labeled submatrices from a given set of position vectors between neighboring atoms

        We take some central atom at (0, 0, 0) and consider its neighbors at `relative_neighbor_positions`. Each `n`th nearest neighbor to the central atom
        is a distance of `threshold_distances[n-1]` away from the central atom. We assign a symbolic 3x3 submatrix to describe the force constants between the central
        atom and each of its neighbors.
        
        We set labels for the submatrices under the assumptions:
        - The overall Hessian must be symmetric. This means that the submatrix representing the interaction between central atom i
          and neighboring atom j should be the transpose of the submatrix representing the interaction between central atom j and
          neighboring atom i
        - Each submatrix describing the interaction between central atom i and neighboring atom j should be symmetric TODO: is this true for all systems?
        - All atoms are the same species, so the choice of what atom to treat as the central position is arbitrary
        
        Params:
            relative_neighbor_positions (np.array): Array containing all possible position vectors between a central atom at (0, 0, 0) and its neighbors
            threshold_distances (np.array): Array containing the distance between an atom and its first NN, second NN, etc.
        Attr:
            self.positions (np.array): Array containing all possible position vectors between a central atom at (0, 0, 0) and its neighbors
            self.symbolic_submatrices (list): List of symbolic submatrices describing the force constants between the central atom and its neighbor at the same index of `self.positions`
            self.parameters (set): Set of independent symbolic parameters in the Hessian
        """
        all_symbols = list(string.ascii_lowercase)
        
        _, unique_atomic_site_idx = np.unique(relative_neighbor_positions, axis=0, return_index=True)
        unique_atomic_sites = relative_neighbor_positions[np.sort(unique_atomic_site_idx), :]
        atomic_site_distances = np.linalg.norm(unique_atomic_sites, axis=1)
        
        self.positions = np.zeros((2*len(unique_atomic_sites), 3))
        self.symbolic_submatrices = []
        self.parameters = set()
        self.pos_insertion_idx = 0
        
        # Iterate over nearest neighbor shells. Because all atoms in a NN shell are equidistant from the central atom, they share the same set
        # of force constant magnitudes, permuted according to their relative position from the center. Solve the symmetry constraints applied
        # to the submatrix describing interactions between the central atom and some reference atom in the current shell, then, determine how all
        # other atoms in the shell are related to this reference atom by symmetry
        for shell_idx, shell_distance in enumerate(threshold_distances):
            sites_in_shell = unique_atomic_sites[np.isclose(atomic_site_distances, shell_distance), :]
            ref_atom_in_shell = sites_in_shell[0, :]
            shell_symbols = sympy.symbols('%s0:6'%all_symbols[shell_idx])
            ref_atom_submatrix = sympy.Matrix([[shell_symbols[0], shell_symbols[5], shell_symbols[4]],
                                               [shell_symbols[5], shell_symbols[1], shell_symbols[3]],
                                               [shell_symbols[4], shell_symbols[3], shell_symbols[2]]])
            #TODO: Note that this submatrix form places *no* assumptions on the matrix structure. For FCC, submatrices are symmetric so this formulation can be further simplified

            all_constraints = []
            # Check for symmetry constraints on the submatrix
            for M in symm_group.symmetry_matrices[1:]:
                # If ref atom is invariant under symmetry operation, apply a constraint
                if np.all(np.isclose(M@ref_atom_in_shell, ref_atom_in_shell)): #TODO: okay to check for equality here?
                    constraints = sympy.nsimplify(M@ref_atom_submatrix@M.T - ref_atom_submatrix)
                    all_constraints.extend(constraints)
            # Solve the constraint system
            shell_solution = sympy.solve(all_constraints, shell_symbols)
            ref_atom_submatrix = ref_atom_submatrix.subs(shell_solution)
            self.add_atomic_site(ref_atom_in_shell, ref_atom_submatrix)
            
            # Then, find how all other atoms in shell transform from the ref atom
            for atom_in_shell in sites_in_shell[1:, :]:
                for M in symm_group.symmetry_matrices[1:]:
                    if np.all(np.isclose(M@ref_atom_in_shell, atom_in_shell)):
                        shell_atom_submatrix = sympy.nsimplify(M@ref_atom_submatrix@M.T)
                        self.add_atomic_site(atom_in_shell, shell_atom_submatrix)
                        break

        # Finally, handle the central atom
        central_atom_symbols = sympy.symbols('%s0:6'%all_symbols[shell_idx+1])
        central_atom_submatrix = sympy.Matrix([[central_atom_symbols[0], 0, 0, ],
                                               [0, central_atom_symbols[0], 0],
                                               [0, 0, central_atom_symbols[0]]])
        # TODO: In FCC, `central_atom_submatrix` is simply a multiple of identity so the array above can be simplified. In HCP, the self-interaction submatrix is not diagonal, so analysis will fail here
        self.add_atomic_site(np.array([0, 0, 0]), central_atom_submatrix)

        # Remove any unused indices of the positions array
        final_used_idx = np.argmax(np.all(np.isclose(self.positions, 0), axis=1))
        self.positions = self.positions[:final_used_idx+1]

    def add_atomic_site(self, relative_neighbor_position, symbolic_submatrix):
        if not np.any(np.all(np.isclose(self.positions, relative_neighbor_position), axis=1)) or np.all(relative_neighbor_position == [0, 0, 0]):
            self.positions[self.pos_insertion_idx, :] =  relative_neighbor_position
            self.symbolic_submatrices.append(symbolic_submatrix)

            if not np.all(relative_neighbor_position == [0, 0, 0]):
                # Avoid duplicate entry for zero vector
                self.positions[self.pos_insertion_idx+1, :] =  -relative_neighbor_position
                self.symbolic_submatrices.append(symbolic_submatrix.T)

            [self.parameters.add(symbol) for symbol in symbolic_submatrix.free_symbols]

            self.pos_insertion_idx += 2

    def get_symbolic_submatrix(self, relative_neighbor_position):
        """
        Retrieve the symbolic submatrix that describes the interaction between a central atom and its neighbor at relative
        position `relative_neighbor_position`

        Params:
            relative_neighbor_position (np.array): Array containing the position of a neighboring atom relative to a central
            atom defined to have position (0, 0, 0)
        """
        atomic_site_idx = np.where(np.all(np.isclose(self.positions, relative_neighbor_position), axis=1))[0]
        if len(atomic_site_idx) == 0:
            raise ValueError(f"No equivalent atomic site found for neighbor at {relative_neighbor_position}")
        else:
            return self.symbolic_submatrices[atomic_site_idx[0]]


def compute_free_energy(atomic_mass, T, hessian, mode="classical"):
    """
    Computes free energy in eV using Hessian reconstructed from numeric data given by `true_hessian` or `symbol_mapping`.
    Assumes Hessian is measured in eV/A^2, atomic masses are measured in amu
    Params:
        atomic_mass (float): Atomic mass in amu
        T (float or np.array of floats): Temperature in K
        true_hessian (np.array, optional): A 3`self.n_atoms` x 3`self.n_atoms` numerical Hessian computed by LAMMPS/VASP
        symbol_mapping (dict, optional): A dictionary mapping each parameter in `self.parameters` to a numerical value
        mode (str): String "classical" or "quantum" free energy calculation
    Return:
        F (float): Helmholtz free energy in eV
        F_per_atom (float): Helmholtz free energy per atom
    """
    assert mode in ["classical", "quantum"]
    kB = 8.617343e-5  # eV/K
    hbar = 6.58211899e-16 # eV s
    Na = 6.02214179e23    # mol^-1

    # Convert to kg/s^2 (1.6*10**-19 * 10**20)
    si_units_hessian = hessian*16.022
    # Eigendecomposition. Eigenvalues are in units of kg/s^2
    eigenval, eigenvec = np.linalg.eigh(si_units_hessian)
    # Eigenfrequencies in units of s^-2 (convert amu to kg). Neglect 3 zero eigenmodes
    omega = np.sqrt(eigenval[3:]/(atomic_mass*1.66054*10**(-27)))

    if not np.isscalar(T):
        # Reformat temperature array to allow vectorized calculation
        T_arr = np.repeat(T, len(omega), axis=0).reshape(len(T), len(omega))
    else:
        T_arr = T

    # Helmholtz free energy in units of eV
    if mode == "classical":
        F = -kB*T*np.sum(np.log(kB*T_arr/(hbar*omega)), axis=-1)
    elif mode == "quantum":
        F = 0.5*hbar*np.sum(omega) + kB*T*np.sum(np.log(1-np.exp(-hbar*omega/(kB*T_arr))), axis=-1)

    F_per_atom = F/len(si_units_hessian)
    return F, F_per_atom


def print_result(is_passed):
    """
    Prints pass/fail messages for test case result `is_passed`
    """
    if is_passed:
        return "\033[92mPASSED\033[0m"
    else:
        return "\033[91mFAILED\033[0m"