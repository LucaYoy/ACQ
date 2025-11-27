# %%
import Hamiltonian as ham
import numpy as np
import scipy.io as sio

# %%
n_qubits=range(4,16)
J=1
h=0.5

for n in n_qubits:
    H,H_trot=ham.TFIM(J,h,n)
    print('Computing Eigs for N=',n)
    EH,VH = np.linalg.eigh(H.todense())
    psigs=VH[:,0]
    psiE1=VH[:,1]
    E_gs=EH[0]

    savedict = {
        'N' : n_qubits,
        'J' : J,
        'h' : h,
        'E0' : E_gs,
        'psi0' : psigs,
        'psi1' : psiE1
    }
    sio.savemat('eigsTFIM/N%d_J%1.1f_h%1.1f.mat'%(n,J,h), savedict)


