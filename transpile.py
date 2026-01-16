import numpy as np
from qiskit.quantum_info import SparsePauliOp,Pauli
from qiskit.circuit.library import PauliEvolutionGate,HamiltonianGate
import scipy.sparse as sp
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit_ibm_runtime.fake_provider import FakeValenciaV2,FakeBelemV2,FakeQuitoV2
from qiskit.quantum_info import Operator
from qiskit.synthesis import qs_decomposition,LieTrotter

def cyclic_range(start, stop, n):
    if start < stop:
        return list(range(start, stop))
    else:
        # wrap around: [start, n) ∪ [0, stop)
        return list(range(start, n)) + list(range(0, stop))

def support_of_Ak(k,D,T,n_qubits):
    a = D - T
    half = a // 2
    offset = a % 2  # 0 if even, 1 if odd

    start = (k - half) % n_qubits
    stop  = (k + T + half + offset) % n_qubits

    support = cyclic_range(start, stop, n_qubits)
    return support


def transpile_circuit_qite(n_qubits,D,T,PD,a,dt,steps,backend,op_level,PEvolution=False):
    # set up the quantum circuit and backend
    qc_v = QuantumCircuit(n_qubits)
    v_true = np.eye(2**n_qubits)

    # loop through the Pauli operators and construct the circuits
    for step in range(steps):
        for k in range(len(PD)):
            
            support = support_of_Ak(k,D,T,n_qubits)
            PD_k_reduced = [''.join(string[i] for i in support) for string in PD[k]]
            #print(PD_k_reduced,PD[k],support)
            Ak_reduced = SparsePauliOp(PD_k_reduced,a[step,k,:])
            Ak = SparsePauliOp(PD[k],a[step,k,:]).to_matrix()

            qiskit_support = [n_qubits-1-i for i in support[::-1]]  #reverse order for qiskit
            if PEvolution:
                qc_v.append(PauliEvolutionGate(Ak_reduced,dt),qiskit_support) #use support to reduce qubits
            else:
                qc_v.append(HamiltonianGate(Ak_reduced,dt),qiskit_support) #use support to reduce qubits
            
            v_true = sp.linalg.expm(-1j *Ak*dt) @ v_true
            
    # transpile the circuits to the native gates of the backend
    tr_qc_v = transpile(qc_v,backend,seed_transpiler=40,optimization_level=op_level)
    
    # Calculate error using trace norm
    tr_unitary = Operator(tr_qc_v)
    error = (tr_unitary.data - v_true) 
    trace_norm = np.linalg.norm(error)  #trace norm

    return tr_qc_v, v_true, trace_norm

def transpile_circuit_qite_adap(n_qubits,D,T,PD,a,times,steps,backend,op_level,PEvolution=False):
    # set up the quantum circuit and backend
    qc_u = QuantumCircuit(n_qubits)
    A_sum = 0
    u_true = np.eye(2**n_qubits)

    for step in range(steps):
        # loop through the Pauli operators and construct the circuits 
        for k in range(len(PD)):
            #compute these for true matrix
            Ak = SparsePauliOp(PD[k],a[step][k,:])        
            A_sum += Ak

            #undo compression for transpling in blocks of support D
            support = support_of_Ak(k,D,T,n_qubits)
            PD_k_reduced = [''.join(string[i] for i in support) for string in PD[k]]
            Ak_reduced = SparsePauliOp(PD_k_reduced,a[step][k,:])
            #print(PD_k_reduced,PD[k],support)

            qiskit_support = [n_qubits-1-i for i in support[::-1]]  #reverse order for qiskit
            if PEvolution:
                qc_u.append(PauliEvolutionGate(Ak_reduced,times[step]),qiskit_support) #use support to reduce qubits
            else:
                qc_u.append(HamiltonianGate(Ak_reduced,times[step]),qiskit_support) #use support to reduce qubits

        
        # Update the true ACQ unitary evolution
        A_sum = A_sum.simplify()
        u_true = sp.linalg.expm(-1j * A_sum.to_matrix()*times[step]) @ u_true
        A_sum = 0

    # transpile the circuits to the native gates of the backend
    tr_qc_u = transpile(qc_u,backend,seed_transpiler=40,optimization_level=op_level)
    
    # Calculate error using trace norm
    tr_unitary = Operator(tr_qc_u)
    error = (tr_unitary.data - u_true)
    trace_norm = np.linalg.norm(error) #trace norm
    #print(Operator(u_true).equiv(Operator(tr_qc_u)))

    return tr_qc_u, u_true, trace_norm