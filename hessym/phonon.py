import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import seaborn as sns
from .util import read_dat
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy import Phonopy

class Calculator:
    def __init__(self, hessian, positions, element="Ni"):
        """
        Initialize calculator object for phonon calculations
        Params:
            H (np.array): Hessian matrix
            positions (np.array): Atomic coordinates of system
            element (str): String name of element being modeled
        """
        self.n_atoms = len(positions)
        self.hessian = hessian
        self.unitcell = PhonopyAtoms(symbols=[element]*self.n_atoms,
                                     cell=np.eye(3),
                                     scaled_positions=positions)
        self.calculator = Phonopy(self.unitcell)
        self.calculator.force_constants = self.hessian.reshape(self.n_atoms, 3, self.n_atoms, 3).transpose(0, 2, 3, 1)
        self.calculator.save()

    def gen_dispersion(self, path, path_labels, label="", c="tab:blue", npoints=50):
        """
        Generate phonon dispersion relation
        """
        qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=npoints)
        self.calculator.run_band_structure(qpoints, path_connections=connections, labels=path_labels)
        self.calculator.plot_band_structure()

    def gen_dos(self, freq_range=[-1, 11], interval=0.05, smear=0.05, label="", c="tab:blue", xlim=None):
        """
        Generate phonon DOS
        """
        self.calculator.run_mesh([32, 32, 32])
        self.calculator.run_total_dos(freq_min=freq_range[0], freq_max=freq_range[1], sigma=smear, freq_pitch=interval)
        frequencies, dos = self.calculator.get_total_DOS()
        plt.plot(frequencies, dos, label=label, c=c)
        plt.xlabel("Frequency")
        plt.ylabel("DOS")
        if xlim is not None:
            plt.xlim(*xlim);
        return frequencies, dos


class DOS:
    def __init__(self, hessian, positions, element="Ni"):
        """
        Initialize calculator object for DOS
        Params:
            H (np.array): Hessian matrix
            positions (np.array): Atomic coordinates of system
            element (str): String name of element being modeled
            freq_range (list[int]): Range of frequency vaues over which DOS should be generated
            interval (float): Frequency interval for plotting DOS points
            smear (float): Smearing width to use in smoothing DOS curve
        """
        self.n_atoms = len(positions)
        self.unitcell = PhonopyAtoms(symbols=[element]*self.n_atoms,
                                     cell=np.eye(3),
                                     scaled_positions=positions)
        self.calculator = Phonopy(self.unitcell)
        self.hessian = hessian

    def plot(self, freq_range=[-1, 11], interval=0.05, smear=0.05, label="", c="tab:blue", xlim=None):
        """
        Plot DOS
        """
        # Compute phonons
        self.calculator.force_constants = self.hessian.reshape(self.n_atoms, 3, self.n_atoms, 3).transpose(0, 2, 3, 1)
        self.calculator.save()
        self.calculator.run_mesh([32, 32, 32])
        self.calculator.run_total_dos(freq_min=freq_range[0], freq_max=freq_range[1], sigma=smear, freq_pitch=interval)
        frequencies, dos = self.calculator.get_total_DOS()
        plt.plot(frequencies, dos, label=label, c=c)
        plt.xlabel("Frequency")
        plt.ylabel("DOS")
        if xlim is not None:
            plt.xlim(*xlim);
        return frequencies, dos
    


def plot_dos(filepath, title="", label="", c="tab:blue"):
    """
    Plots the phonon DOS located at `filepath`
    """
    dos_df = read_dat(filepath)
    dos_df.columns = ["freq", "dos"]
    
    plt.plot(dos_df["freq"], dos_df["dos"], c=c, label=label)
    plt.xlabel("Frequency")
    plt.ylabel("Phonon DOS")
    if label:
        plt.legend()
    plt.title(title)

    return dos_df


def plot_dispersion(filepath, title="", k_labels=[r"$\Gamma$", "X", "W", "K", r"$\Gamma$"], label="", c="tab:blue"):
    """
    Plots the phonon dispersion located at `filepath`
    """
    with open(filepath) as f:
        k_point_data = f.readlines()[1]
    k_points = np.array(re.sub(r"[^0-9. ]", "", k_point_data).lstrip().rstrip().split(" "), dtype=float)
    dispersion_df = read_dat(filepath)
    dispersion_df.columns = ["k", "freq"]
    bands = np.split(dispersion_df, np.where(dispersion_df["k"].diff() < 0)[0], axis=0)

    for band in bands[:-1]:
        plt.plot(band["k"], band["freq"], c=c)
    plt.plot(bands[-1]["k"], bands[-1]["freq"], c=c, label=label)
    
    plt.ylabel("Frequency")
    plt.xticks(k_points, labels=k_labels)
    plt.axhline(0, c="k", linestyle="--")
    plt.title(title)
    if label:
        plt.legend()

    return bands


def plot_dos_by_nn(directory, filename, title="", return_ax=False, palette="viridis_r", savefig=False, savepath=None):
    """
    Generates an overlaid DOS plot for all data files in `dir` with names matching `filename` and computes the RMSE of each reconstruction relative to the DOS computed from the full Hessian
    Assumes that all relevant files take the form `filename`_{x}NN, whre `x` indicates the number of nearest neighbors treated as non-zero in phonon calculations
    Also assumes that the DOS computed from the full Hessian is named `filename`
    """
    all_files = np.array(os.listdir(directory))
    files_mask = [bool(re.match(filename, file)) for file in all_files]
    nn_files = sorted(all_files[files_mask])
    max_nn = len(nn_files)
    
    cmap = sns.color_palette(palette, max_nn-1)
    
    dos_fig, dos_ax = plt.subplots(dpi=160)
    full_dos = plot_dos(f"{directory}/{nn_files[-1]}")
    rmse = []
    for i, nn_file in enumerate(nn_files[:-1]):
        nn = int(re.findall(r"(\d)NN", nn_file)[0])
        dos = plot_dos(f"{directory}/{nn_file}", label=f"{nn} NN", c=cmap[i])
        rmse.append(np.sqrt(np.mean((dos["dos"] - full_dos["dos"])**2)))
    plot_dos(f"{directory}/{nn_files[-1]}", label=f"Full Hessian", c="tab:red")
    plt.xlabel(r"Frequency")
    plt.ylabel("Phonon DOS")
    plt.legend()
    if savefig:
        plt.savefig(f"{savepath}/dos")
    
    rmse_fig, rmse_ax = plt.subplots(figsize=(8, 4))
    plt.bar(range(1, max_nn), rmse, color=cmap)
    plt.xlabel("NN Considered", fontsize=15)
    plt.ylabel("RMSE", fontsize=14)
    plt.xticks(range(1, max_nn), fontsize=20)
    if savefig:
        plt.savefig(f"{savepath}/dos_rmse")

    if return_ax:
        return rmse, (dos_fig, dos_ax), (rmse_fig, rmse_ax)
    else:
        return rmse


def plot_dispersion_by_nn(directory, filename, title="", k_labels=[r"$\Gamma$", "X", "W", "K", r"$\Gamma$"]):
    """
    Generates an overlaid dispersion plot for all data files in `dir` with names matching `filename` and computes the RMSE of each reconstruction relative to the dispersion computed from the full Hessian
    Assumes that all relevant files take the form `filename`_{x}NN, whre `x` indicates the number of nearest neighbors treated as non-zero in phonon calculations
    Also assumes that the dispersion computed from the full Hessian is named `filename`
    """
    all_files = np.array(os.listdir(directory))
    files_mask = [bool(re.match(filename, file)) for file in all_files]
    nn_files = sorted(all_files[files_mask])
    max_nn = len(nn_files)
    
    cmap = sns.color_palette("viridis_r", max_nn-1)
    
    plt.figure(dpi=160)
    full_dispersion = plot_dispersion(f"{directory}/{nn_files[-1]}")
    full_dispersion_vals = np.zeros((len(full_dispersion), len(full_dispersion[0])))
    for band_idx, band in enumerate(full_dispersion):
        full_dispersion_vals[band_idx, :] = band["freq"].values
    rmse = []
    for i, nn_file in enumerate(nn_files[:-1]):
        nn = int(re.findall(r"(\d)NN", nn_file)[0])
        dispersion = plot_dispersion(f"{directory}/{nn_file}", k_labels=k_labels, label=f"{nn} NN", c=cmap[i])
        dispersion_vals = np.zeros((len(dispersion), len(dispersion[0])))
        for band_idx, band in enumerate(dispersion):
            dispersion_vals[band_idx, :] = band["freq"].values
        rmse.append(np.sqrt(np.mean((full_dispersion_vals - dispersion_vals)**2)))
    plot_dispersion(f"{directory}/{nn_files[-1]}", c="tab:red", label="Full Hessian")
    
    plt.figure(figsize=(7, 4))
    plt.bar(range(1, max_nn), rmse)
    plt.xlabel("NN Considered", fontsize=15)
    plt.ylabel("RMSE", fontsize=14)
    plt.xticks(range(1, max_nn), fontsize=20)
    return rmse