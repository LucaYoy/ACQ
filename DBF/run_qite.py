from typing import List, Optional, Sequence

from qrisp import QuantumVariable
from qrisp.algorithms.qite import QITE # https://qrisp.eu/reference/Algorithms/QITE.html
from qrisp.operators import X, Y, Z

import numpy as np
import sympy as sp
import scipy.sparse as spy


from qiskit.quantum_info import SparsePauliOp,Pauli


def cyclic_range(start: int, stop: int, n: int) -> List[int]:
    """Return cyclic indices in [start, stop) on a ring of size n."""
    if start < stop:
        return list(range(start, stop))
    return list(range(start, n)) + list(range(0, stop))


def support_of_Ak(k: int, D: int, T: int, n_qubits: int) -> List[int]:
    """Return support qubits for the k-th T-local Hamiltonian piece."""
    a = D - T
    half = a // 2
    offset = a % 2

    start = (k - half) % n_qubits
    stop = (k + T + half + offset) % n_qubits
    return cyclic_range(start, stop, n_qubits)


def _qrisp_term_from_pauli(pauli_reduced: str, support: Sequence[int]):
    """Build a Qrisp operator term for one reduced Pauli string on a support."""
    term = None
    for local_idx, p in enumerate(pauli_reduced):
        q = support[local_idx]
        if p == "I":
            continue
        if p == "X":
            factor = X(q)
        elif p == "Y":
            factor = Y(q)
        elif p == "Z":
            factor = Z(q)
        else:
            raise ValueError(f"Unsupported Pauli label '{p}' in string '{pauli_reduced}'.")
        term = factor if term is None else term * factor
    return term


def _qrisp_generator_from_coeffs(
    pauli_strings_reduced: Sequence[str],
    coeffs: np.ndarray,
    support: Sequence[int],
    atol: float,
):
    """Build A_k = sum_j a_j P_j as a Qrisp operator for a fixed support."""
    generator = 0
    for pstr, coeff in zip(pauli_strings_reduced, coeffs):
        term = _qrisp_term_from_pauli(pstr, support)
        if term is None:
            continue

        # ACQ/QITE coefficients should be real for a Hermitian generator.
        if abs(np.imag(coeff)) > atol:
            raise ValueError(
                "Coefficient has non-negligible imaginary part in ACQ generator: "
                f"{coeff}. Increase atol only if this is numerical noise."
            )
        coeff_real = float(np.real(coeff))
        if abs(coeff_real) <= atol:
            continue
        generator += coeff_real * term
    return generator


def build_acq_circuit_qrisp(
    n_qubits: int,
    D: int,
    T: int,
    PD: List[List[str]],
    a: List[np.ndarray],
    times: List[float],
    steps: int,
    U_0: Optional[callable] = None,
    trotter_steps: int = 1,
    trotter_method: str = "commuting",
    atol: float = 1e-10,
    alpha: float = 1.0,
    debug: bool = False,
):
    """
    Qrisp equivalent of ACQ circuit construction from transpile_circuit_qite_adap.

    It applies, for each adaptive step and each Hamiltonian piece k:
        exp(-i * A_k(step) * times[step])
    where A_k(step) = sum_j a[step][k, j] P_{k, j}, restricted to local support D.

    Args:
        n_qubits: Number of qubits.
        D: Domain size used for each local generator.
        T: Locality of each Trotter Hamiltonian piece.
        PD: Pauli decomposition strings, shape [Nk][num_paulis].
        a: Adaptive coefficients, one array per step with shape [Nk, num_paulis].
        times: Adaptive evolution time per step.
        steps: Number of adaptive steps to apply.
        U_0: Optional initial state preparation callable, U_0(qv).
        trotter_steps: Number of Trotter steps in qrisp operator evolution.
        alpha: Scaling factor for evolution times in qrisp operator evolution.
        trotter_method: Qrisp trotterization method (e.g. "commuting").
        atol: Tolerance for discarding tiny coefficients / imaginary noise.

    Returns:
        Compiled Qrisp circuit implementing the ACQ sequence.
    """
    if steps > len(a) or steps > len(times):
        raise ValueError(
            f"steps={steps} is inconsistent with len(a)={len(a)} and len(times)={len(times)}."
        )

    qv = QuantumVariable(n_qubits)
    if U_0 is not None:
        U_0(qv)

    # Capture the unitary of the initial state preparation, if any.
    # The final circuit unitary includes this block, so the exact target below
    # must compose it in the same order when comparing full-circuit matrices.
    u_prep = qv.qs.get_unitary()

    u_true = np.eye(2**n_qubits)

    for step in range(steps):
        A_sum = 0
        #A_sum_qrisp = 0
        t_step = float(times[step])
        for k in range(len(PD)):
            A_k = SparsePauliOp(PD[k], a[step][k, :])
            A_sum += A_k

            support = support_of_Ak(k, D, T, n_qubits)
            PD_k_reduced = ["".join(pstr[q] for q in support) for pstr in PD[k]]

            A_k_reduced = _qrisp_generator_from_coeffs(PD_k_reduced, a[step][k, :], support, atol)
            A_k_reduced.trotterization(method=trotter_method)(qv, t_step/alpha, trotter_steps)
            #A_sum_qrisp += A_k_reduced

        A_sum = A_sum.simplify()
        u_true = spy.linalg.expm(-1j * A_sum.to_matrix()*times[step]) @ u_true


    if debug:
        # Build the full exact target including the initial state preparation.
        # Full unitary: apply the preparation first, then evolution: U_full = U_evol @ U_prep
        u_target_right = u_true @ u_prep

        # Calculate error using trace norm
        qc_unitary = qv.qs.get_unitary()
        dim = qc_unitary.shape[0]
        error = (qc_unitary - u_target_right)
        trace_norm = np.linalg.norm(error) # Frobenius norm
        print(f"Trace norm error between Qrisp circuit unitary and target ACQ unitary: {trace_norm:e}")

        # Compare the actually prepared state on |0...0> as well. This is the quantity
        # most directly relevant if U_0 is only meant to prepare an initial state.
        try:
            zero = np.zeros((dim,), dtype=complex)
            zero[0] = 1.0
            psi_qc = qv.qs.statevector_array()
            psi_target_right = u_target_right @ zero
            print(f"fidelity: {np.abs(np.vdot(psi_qc, psi_target_right))**2:e}")
        except Exception as e:
            print('Statevector diagnostics failed:', e)

    qc = qv.qs.compile()
    return qc


def run_QITE(H, U_0, exp_H, s_values, steps, method='GC', use_statevectors=False):

    H_matrix = H.to_sparse_matrix()

    theta = sp.Symbol('theta')
    optimal_s = [theta]
    optimal_energies = []
    circuits = []
    statevectors = []

    N = H.find_minimal_qubit_amount()
    qv = QuantumVariable(N)
    U_0(qv)
    qc = qv.qs.compile()
    circuits.append(qc)

    if use_statevectors:
        E_0, _, _ = compute_moments(get_statevector(qc, N), H_matrix)
    else:
        E_0 = H.get_measurement(qv, precision=0.01, precompiled_qc=qc, diagonalisation_method='commuting')

    optimal_energies.append(E_0)

    # Find optimal evolution times s_k with 20-point grid search
    for k in range(1,steps+1):
        
        print(f"Step {k}/{steps}...")
        # Perform k steps of QITE
        def state_prep():
            qv = QuantumVariable(H.find_minimal_qubit_amount())
            QITE(qv, U_0, exp_H, optimal_s, k, method=method)
            return qv
        
        qv = state_prep()
        qc = qv.qs.compile()
        circuits.append(qc)


        # Find optimal evolution time 
        # Use "precompliled_qc" keyword argument to avoid repeated compilation of the DB-QITE circuit
        if use_statevectors:
            energies = [compute_moments(get_statevector(qc, N, subs_dic={theta:s_}), H_matrix)[0] for s_ in s_values]
        else:
            energies = [H.expectation_value(state_prep, precision=0.01, diagonalisation_method='commuting', subs_dic={theta:s_}, precompiled_qc=qc)() for s_ in s_values]

        index = np.argmin(energies)
        s_min = s_values[index]

        optimal_s.insert(-1,s_min)
        optimal_energies.append(energies[index])


    evolution_times = [sum(optimal_s[i] for i in range(k)) for k in range(steps+1)]

    circuit_ops = {}
    circuit_qubits = {}
    circuit_depth = {}

    # Collect data for circuits and statevectors
    for k, qc in enumerate(circuits):
        # We need to bind the symbolic parameter theta to transpile to specific basis gates
        if k>0:
            qc = qc.bind_parameters(subs_dic={theta:optimal_s[k-1]})   
 
        tqc = qc.transpile(basis_gates=["cz","u"])
        circuit_ops[k] = tqc.count_ops()
        circuit_qubits[k] = tqc.num_qubits()
        circuit_depth[k] = tqc.depth()

        statevectors.append(get_statevector(qc, N))

    circuit_data = [circuit_ops, circuit_qubits, circuit_depth]

    result_dict = {'evolution_times':evolution_times, 'optimal_energies':optimal_energies, 'circuit_data':circuit_data, 'statevectors':statevectors}

    return result_dict


def get_statevector(qc, n, subs_dic=None):
    if subs_dic is not None:
        bqc = qc.bind_parameters(subs_dic) 
    else:
        bqc = qc
    
    for i in range(bqc.num_qubits() - n):
        bqc.qubits.insert(0, bqc.qubits.pop(-1))

    statevector = bqc.statevector_array()[:2**n]
    statevector = statevector/np.linalg.norm(statevector)

    return statevector

def compute_moments(psi, H):
    E = (psi.conj().T @ H.dot(psi)).real
    S = (psi.conj().T @ (H @ H).dot(psi)).real
    return E, S, S - E**2