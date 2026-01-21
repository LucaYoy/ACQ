# ACQ: Adaptive time Compressed QITE — Repository Overview

This repository contains code and notebooks for Quantum Imaginary Time Evolution (QITE) and Adaptive time Compressed QITE (ACQ), along with tools to build and evaluate transpiled circuits on realistic backends. ACQ is an algorithm that reuses the same effective unitary across multiple time steps, recomputing it only when the energy stops decreasing, which can reduce compilation overhead while preserving convergence.

The ACQ method implemented here follows the preprint “Adaptive time Compressed QITE (ACQ) and its geometrical
interpretation” and provides a practical path from Hamiltonian models to transpiled circuits, and benchmarking.

## What’s Inside

- QITE and ACQ simulation routines (energy tracking, coefficient recovery, fused unitary construction)
- Trotterization utilities for common spin models (TFIM, Cluster-Ising, Heisenberg)
- Pauli string generators (general and real-valued optimizations)
- Circuit construction and transpilation helpers
- Notebooks to reproduce figures and metrics (energy, fidelity, variance, trace-norm error)

## Repository Structure

- ACQ/
	- [Evolution.py](Evolution.py): Implements `QITE()` and `ACQ()`.
		- `QITE()`: Iterative imaginary-time update using per-piece generators.
		- `ACQ()`: Adaptive strategy that reuses a fused unitary until energy stops decreasing.
	- [Hamiltonian.py](Hamiltonian.py): Trotterization helpers for models:
		- `TFIM()`: Transverse Field Ising Model
		- `ClusterIsing()`: Cluster-Ising model
		- `Heisenberg()`: Heisenberg model
	- [PauliStrings.py](PauliStrings.py): Pauli operator bases for local domains `D`.
		- `general()`: full 4^D basis
		- `real()`: reduced basis for real states/Hamiltonians
	- [transpile.py](transpile.py): Circuit builders and transpilation utilities.
		- `transpile_circuit_qite()`: step-wise QITE circuits and exact-unitary comparison
		- `transpile_circuit_qite_adap()`: ACQ circuits using fused generators and times
	- Notebooks:
		- [transpile_sim.ipynb](transpile_sim.ipynb): End-to-end experiment: build circuits, transpile across optimization levels, and plot energy/fidelity/variance/error.

## Installation

Create a fresh environment and install Python dependencies. Qiskit, SciPy, NumPy, and Matplotlib are required; Qiskit Aer and fake backends are used for transpilation experiments.

## Quick Start (Python)

Run a minimal ACQ experiment on TFIM:

```python
import numpy as np
import scipy.sparse as sp
from Hamiltonian import TFIM
from Evolution import ACQ

n_qubits, T, D = 5, 2, 2
J, h, dt, N = 0.5, 1.0, 0.1, 50

H, H_trot = TFIM(J, h, n_qubits, T=T)
psi0 = np.zeros((2**n_qubits, 1), dtype=complex)
psi0[0] = 1
psi_0 = sp.csc_matrix(psi0 / np.linalg.norm(psi0))

E, psi_QITE, indices, times, a = ACQ(n_qubits, H, H_trot, D, psi_0, N, dt)
print("Final energy:", E[np.nonzero(E)].tolist()[-1])
```
## Key Concepts

- **Imaginary Time Evolution (QITE):** Approximates `exp(-H·dt)` via unitary approximation updates built from local Pauli bases over domain `D`.
- **Adaptive time Compressed QITE (ACQ):** Computes a fused generator from current state; reuses it for multiple steps while energy decreases, then recomputes as needed.
- **Trotterization:** Decomposes `H` into local pieces (`T`-local). See [Hamiltonian.py](Hamiltonian.py).
- **Pauli Bases:** Choose `general()` or `real()` depending on Hamiltonian/state realness to reduce measurements and runtime.
- **Transpilation & Metrics:** Compare ideal vs transpiled circuits with [transpile.py](transpile.py) utilities and notebook plots.

## Citing ACQ

If you use this code, please cite the ACQ preprint included with the repository.

## Acknowledgments

- Built on top of Qiskit, NumPy, SciPy, and Matplotlib.
- Includes fake backends and Aer simulators for reproducible transpilation experiments.

