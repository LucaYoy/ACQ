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
from qrisp.vqe import VQEProblem
from qrisp.vqe.problems.heisenberg import create_heisenberg_init_function
from qrisp.operators import X, Y, Z
import networkx as nx

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


def H_0(J, n_qubits):
    H = 0
    for i in range(0, n_qubits - 1, 2):
        H += J * (X(i) * X(i + 1) + Y(i) * Y(i + 1) + Z(i) * Z(i + 1))
    return H


def H_1(J, n_qubits):
    H = 0
    for i in range(1, n_qubits - 1, 2):
        H += J * (X(i) * X(i + 1) + Y(i) * Y(i + 1) + Z(i) * Z(i + 1))
    return H


def main():
    n_qubits = 8
    J = 1

    D = 8
    dt = 0.1
    N = 10
    T = 8

    print(f'running ACQ hva for {n_qubits} qubits, J={J}, D={D}, dt={dt}, N={N}, T={T}')

    H_qr = Heisenberg_qr(J, n_qubits)
    print("Computing eigs for fidelity reference")
    EH, VH = np.linalg.eigh(H_qr.to_array())
    psigs = VH[:, 0:1]

    def ansatz_hva(qv, params):
        H_1(J, n_qubits).trotterization(method="commuting")(qv, t=params[1])
        H_0(J, n_qubits).trotterization(method="commuting")(qv, t=params[0])

    G = nx.Graph()
    G.add_edges_from([(k, k + 1) for k in range(n_qubits - 1)])
    M = nx.maximal_matching(G)
    U_singlet = create_heisenberg_init_function(M)

    HVA = VQEProblem(H_qr, ansatz_hva, 2, init_function=U_singlet)
    U_HVA = HVA.train_function(QuantumVariable(n_qubits), depth=1, max_iter=100)

    def state_prep_hva():
        qv = QuantumVariable(n_qubits)
        U_HVA(qv)
        return qv

    print("Computing ACQ for HVA initial state")
    H_acq, H_trot = ham.Heisenberg_OBC_DN(J, n_qubits, sparse=True)

    qc_hva = state_prep_hva().qs.compile()
    psi_0_hva = sp.csc_matrix(get_statevector(qc_hva, n_qubits).reshape(-1, 1))

    E_ACQ, psi_ACQ, indx_acq, times_acq, a_acq = evol.ACQ_QC(
        n_qubits,
        H_acq,
        H_trot,
        D,
        psi_0_hva,
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
    with open(f"Heisenberg_N{n_qubits}_J{J}_ACQ_D{D}_hva_results_simulation.pkl", "wb") as f:
        pickle.dump(simulation_data_acq, f)

    num_paulis_gen, PD_gen, _ = ps.general_OBC(H_trot, D, n_qubits, PDstr=True)

    steps_acq = len(a_acq)
    circuit_ops = {}
    circuits = {}

    for step_acq in range(steps_acq + 1):
        qc = build_acq_circuit_qrisp(
            n_qubits,
            D,
            T,
            PD_gen,
            a_acq,
            times_acq,
            step_acq,
            U_HVA,
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

    circuit_data_hva = {
        "circuits": circuits,
        "circuit_ops": circuit_ops,
        "energies": energies,
        "fidelities": fidelities,
    }

    with open(f"Heisenberg_N{n_qubits}_J{J}_ACQ_D{D}_hva_results_transpile.pkl", "wb") as f:
        pickle.dump(circuit_data_hva, f)
    

if __name__ == "__main__":
    main()
