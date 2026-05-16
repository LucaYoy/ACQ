import sys
sys.path.append("../")

import pickle
import importlib
import numpy as np
import scipy.sparse as sp

import Hamiltonian as ham
import Evolution as evol
import PauliStrings as ps

from qrisp import QuantumVariable, x
from qrisp.operators import X, Y, Z

import run_qite
importlib.reload(run_qite)
from run_qite import build_acq_circuit_qrisp, get_statevector, compute_moments


def fidelity_pure(psi, phi):
    """
    Input values should be column vectors.
    """
    F = np.abs(psi.conj().T @ phi) ** 2
    return F[0, 0]


def Heisenberg_qr(J, n_qubits):
    H = 0
    for i in range(n_qubits - 1):
        H += J * (X(i) * X(i + 1) + Y(i) * Y(i + 1) + Z(i) * Z(i + 1))
    return H


def U_10(qv):
    for i in range(qv.size):
        if i % 2 == 0:
            x(qv[i])


def main():
    n_qubits = 8
    J = 1

    D = 8
    dt = 0.1
    N = 10
    T = 8

    print(f"Running ACQ for Heisenberg model 1010 with N={n_qubits}, J={J}, D={D}, dt={dt}, N={N}, T={T}")
    H_qr = Heisenberg_qr(J, n_qubits)
    print("Computing eigs for fidelity reference")
    EH, VH = np.linalg.eigh(H_qr.to_array())
    psigs = VH[:, 0:1]

    def state_prep_10():
        qv = QuantumVariable(n_qubits)
        U_10(qv)
        return qv

    print("Computing ACQ for 1010 initial state")
    H_acq, H_trot = ham.Heisenberg_OBC_DN(J, n_qubits, sparse=True)

    qc_10 = state_prep_10().qs.compile()
    psi_0_10 = sp.csc_matrix(np.real_if_close(get_statevector(qc_10, n_qubits).reshape(-1, 1)))

    E_ACQ, psi_ACQ, indx_acq, times_acq, a_acq = evol.ACQ_QC(
        n_qubits,
        H_acq,
        H_trot,
        D,
        psi_0_10,
        N,
        dt,
        methodLS="LU",
        OBC=True,
    )

    simulation_data_acq = {
        "E_ACQ": E_ACQ,
        "psi_ACQ": psi_ACQ,
        "indx_acq": indx_acq,
        "times_acq": times_acq,
        "a_acq": a_acq,
    }
    with open(f"Heisenberg_N{n_qubits}_J{J}_ACQ_D{D}_1010_results_simulation.pkl", "wb") as f:
        pickle.dump(simulation_data_acq, f)

    num_paulis_real, PD_real, _ = ps.real_OBC(H_trot, D, n_qubits, PDstr=True)
    steps_acq = len(a_acq)
    circuit_ops = {}
    circuits = {}

    for step_acq in range(steps_acq + 1):
        qc = build_acq_circuit_qrisp(
            n_qubits,
            D,
            T,
            PD_real,
            a_acq,
            times_acq,
            step_acq,
            U_10,
            debug=False,
        )
        tqc = qc.transpile(basis_gates=["cz", "u"])
        circuits[step_acq] = qc
        circuit_ops[step_acq] = tqc.count_ops()
        print(f"Step {step_acq}: {tqc.count_ops()}")

    energies = [
        compute_moments(get_statevector(circuits[step], n_qubits), H_acq)[0]
        for step in range(steps_acq + 1)
    ]
    fidelities = [
        fidelity_pure(psigs, get_statevector(circuits[step], n_qubits).reshape(-1, 1))
        for step in range(steps_acq + 1)
    ]

    circuit_data_10 = {
        "circuits": circuits,
        "circuit_ops": circuit_ops,
        "energies": energies,
        "fidelities": fidelities,
    }

    with open(f"Heisenberg_N{n_qubits}_J{J}_ACQ_D{D}_1010_results_transpile.pkl", "wb") as f:
        pickle.dump(circuit_data_10, f)


if __name__ == "__main__":
    main()
