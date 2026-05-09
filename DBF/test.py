import numpy as np
from qiskit.quantum_info import SparsePauliOp

I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
pd = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

s = 'XYZ'
mat = pd[s[0]]
for p in s[1:]:
    mat = np.kron(mat, pd[p])

qmat = SparsePauliOp(s).to_matrix()
qmat_rev = SparsePauliOp(s[::-1]).to_matrix()

print('Pauli String:', s)
print('Qiskit normal == standard math kronecker:', np.allclose(mat, qmat))
print('Qiskit reversed == standard math kronecker:', np.allclose(mat, qmat_rev))

from qiskit.quantum_info import Statevector, SparsePauliOp

# Start with state |000>
sv = Statevector.from_label('000')

# Apply "XII"
op = SparsePauliOp("XII")
new_sv = sv.evolve(op)

print(new_sv.to_dict())