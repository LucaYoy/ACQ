import sys; sys.path.append("/home/andreu/QITE")
import numpy as np
import scipy.sparse as sp

from matplotlib import pyplot as plt

from qiskit.quantum_info import random_statevector as psirand
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate,HamiltonianGate
from qiskit.quantum_info import SparsePauliOp,Pauli
from qiskit.synthesis.evolution import LieTrotter

import pickle
from pathlib import Path
from scipy.sparse.linalg import eigsh

# Import your modules (adjust paths as needed)
import Hamiltonian as ham
import Evolution_var as evol
import PauliStrings as ps

def load_true_energy_and_state_OLD(J, h, n_qubits):
    """
    Load the exact ground state and energy for the TFIM.
    If they do not exist, compute and save them automatically.
    """
    folder = Path(f"TrueEnergies/TFIM/J{J:.1f}_h{h:.1f}/{n_qubits}")
    gs_file = folder / "ground_state.pkl"
    energy_file = folder / "ground_state_energy.pkl"

    # If files don't exist, compute and save them
    if not gs_file.exists() or not energy_file.exists():
        print(f"[INFO] Ground state files not found in {folder}. Computing ground state...")

        # Build Hamiltonian
        H, _ = ham.TFIM(J, h, n_qubits)

        # Compute exact eigenvalues and eigenvectors
        energies, vectors = np.linalg.eigh(H.todense())

        # Ground state
        E_gs = energies[0]
        psi_gs = vectors[:, 0]

        # Create folder
        folder.mkdir(parents=True, exist_ok=True)

        # Save ground state vector
        with open(gs_file, "wb") as f:
            pickle.dump(psi_gs, f)

        # Save ground state energy
        with open(energy_file, "wb") as f:
            pickle.dump(E_gs, f)

        print(f"[INFO] Ground state computed and saved in {folder}")

        return E_gs, psi_gs

    # Load with pickle
    with open(gs_file, "rb") as f:
        psi_gs = pickle.load(f)
    with open(energy_file, "rb") as f:
        E_gs = pickle.load(f)

    print(f"[INFO] Loaded ground state and energy from {folder}")
    return E_gs, psi_gs



def load_true_energy_and_state(J, h, n_qubits, tol=1e-12, maxiter=10000):
    """
    Load or compute the ground-state energy and eigenvector of the TFIM Hamiltonian.

    Uses sparse iterative diagonalization (eigsh) to find the lowest eigenpair only.
    The result is effectively exact up to the specified tolerance.

    Parameters
    ----------
    J : float
        Coupling constant.
    h : float
        Transverse field strength.
    n_qubits : int
        Number of spins/qubits in the TFIM chain.
    tol : float, optional
        Convergence tolerance for eigsh (default 1e-12).
    maxiter : int, optional
        Maximum number of Lanczos iterations (default 10000).

    Returns
    -------
    E_gs : float
        Ground-state energy.
    psi_gs : np.ndarray
        Ground-state eigenvector.
    """
    folder = Path(f"TrueEnergies/TFIM/J{J:.1f}_h{h:.1f}/{n_qubits}")
    gs_file = folder / "ground_state.pkl"
    energy_file = folder / "ground_state_energy.pkl"
    # Build sparse TFIM Hamiltonian
    H, _ = ham.TFIM(J, h, n_qubits)


    if not gs_file.exists() or not energy_file.exists():
        print(f"[INFO] Ground state files not found in {folder}. Computing ground state...")

        # Build sparse TFIM Hamiltonian
        H, _ = ham.TFIM(J, h, n_qubits)

        # Compute ground state using sparse iterative solver
        E_gs, psi_gs = eigsh(H, k=1, which='SA', tol=tol, maxiter=maxiter)
        #E_gs, psi_gs = float(E_gs[0]), psi_gs[:, 0]
        E_gs = float(E_gs[0])
        psi_gs = psi_gs[:, [0]]   # shape (2**n, 1)

        # Create output folder and save results
        folder.mkdir(parents=True, exist_ok=True)
        with open(gs_file, "wb") as f:
            pickle.dump(psi_gs, f)
        with open(energy_file, "wb") as f:
            pickle.dump(E_gs, f)

        print(f"[INFO] Ground state computed and saved in {folder}")
    else:
        # Load from existing pickle files
        with open(gs_file, "rb") as f:
            psi_gs = pickle.load(f)
        with open(energy_file, "rb") as f:
            E_gs = pickle.load(f)
        print(f"[INFO] Loaded ground state and energy from {folder}")

    # Optional: verify residual norm to confirm accuracy
    residual = np.linalg.norm(H @ psi_gs - E_gs * psi_gs)
    print(f"[DEBUG] Residual norm: {residual:.3e}")

    return E_gs, psi_gs



#def load_true_energy_and_state(J, h, n_qubits):
#    folder = Path(f"TrueEnergies/TFIM/J{J:.1f}_h{h:.1f}/{n_qubits}")
#    gs_file = folder / "ground_state.pkl"
#    energy_file = folder / "ground_state_energy.pkl"

#    if not gs_file.exists() or not energy_file.exists():
#        raise FileNotFoundError(f"True energy/state files not found in {folder}")

    # Load with pickle
#    with open(gs_file, "rb") as f:
#        psi_gs = pickle.load(f)
#    with open(energy_file, "rb") as f:
#        E_gs = pickle.load(f)
#
#    return E_gs, psi_gs

def save_data(J, h, n_qubits, D, T, EQ, Eadap_F_U, psi_QITE, psi_adap_F_U, a, indU_F_U, F_QITE, F_adap_F_U, varE_QITE, varE_adap_F_U):
    """
    Save evolution data to a pickle file, creating folders if necessary.
    """
    psi_QITE_csc = psi_QITE.tocsc()
    psi_adap_F_U_csc = psi_adap_F_U.tocsc()

    evolution_data = {
        'QITE': {
            'energies': EQ,
            'states': {
                'data': psi_QITE_csc.data,
                'indices': psi_QITE_csc.indices,
                'indptr': psi_QITE_csc.indptr,
                'shape': psi_QITE_csc.shape
            },
            'coefficients': a
        },
        'adaptive_QITE_fuse_U': {
            'energies': Eadap_F_U,
            'states': {
                'data': psi_adap_F_U_csc.data,
                'indices': psi_adap_F_U_csc.indices,
                'indptr': psi_adap_F_U_csc.indptr,
                'shape': psi_adap_F_U_csc.shape
            },
            'recalculation_indices': indU_F_U
        },
        'fidelities': {
            'QITE': F_QITE,
            'adaptive_fuse_U': F_adap_F_U
        },
        'energy_variances': {
            'QITE': varE_QITE,
            'adaptive_fuse_U': varE_adap_F_U
        }
    }

    # Define folder and ensure it exists
    folder = Path(f'results/TFIM/J{J:.1f}_h{h:.1f}/{n_qubits}')
    folder.mkdir(parents=True, exist_ok=True)

    # Define file path using the Path object
    filename = folder / f'evolution_data_N{n_qubits}_D{D}_T{T}.pickle'

    # Save the pickle file
    with open(filename, 'wb') as f:
        pickle.dump(evolution_data, f)

    print(f"Evolution data saved to {filename}")


def run_evolution(n_qubits, dt, N_steps, D, T, J, h):
    """
    Run QITE and adaptive QITE with fuse U evolution, using the true ground state for reference.
    """
    # --- Load true ground state and energy ---
    EH, psigs = load_true_energy_and_state(J, h, n_qubits)

    # --- Hamiltonian ---
    H, H_trot = ham.TFIM(J, h, n_qubits, T=T)

    # --- Initial state ---
    psi0 = np.zeros((2**n_qubits, 1), dtype=complex)
    psi0[0] = 1
    psi_0 = sp.csc_matrix(psi0 / np.linalg.norm(psi0))

    # --- QITE evolution ---
    EQ, varE_QITE, F_QITE, psi_QITE, a = evol.QITE(
        n_qubits, H, H_trot, D, psi_0, N_steps, dt, psigs,
        vervose=False, stopping_criteria=True
    )

    # --- Adaptive QITE with fuse U ---
    Eadap_F_U, varE_FUSE, F_FUSE, psi_adap_F_U, indU_F_U = evol.adaptive_QITE_fuse(
        n_qubits, H, H_trot, D, psi_0, N_steps, dt, psigs, stopping_criteria=True
    )

    # --- Save results ---
    #save_data(n_qubits, D, T, EQ, Eadap_F_U, psi_QITE, psi_adap_F_U, a, indU_F_U, F_QITE, F_FUSE, varE_QITE, varE_FUSE)

    return EQ, varE_QITE, F_QITE, psi_QITE, a, Eadap_F_U, varE_FUSE, F_FUSE, psi_adap_F_U, indU_F_U



# -----------------------------
# Example usage
# -----------------------------
#if __name__ == "__main__":
    # Parameters for test
#    n_qubits = 5
#    dt = 0.2
#    N_steps = 50
#    D = 2
#    T = 2
#    J = 1.0
#    h = 0.5

#    print("Running QITE and adaptive QITE test...")
#    EQ, varE_QITE, F_QITE, psi_QITE, a, Eadap_F_U, varE_adap_F_U, F_adap_F_U, psi_adap_F_U, indU_F_U = run_evolution(
#        n_qubits, dt, N_steps, D, T, J, h
#    )
#    save_data(J,h,n_qubits,D,T,EQ,Eadap_F_U,psi_QITE,psi_adap_F_U,a,indU_F_U,F_QITE,F_adap_F_U,varE_QITE,varE_adap_F_U)
#    print("Test complete! Evolution data saved.")
