import Hamiltonian as ham
import scipy.sparse as sp
import numpy as np
import Evolution_sim as evol
import scipy.io as sio

def fidelity_pure(psi,phi):
    '''
    input values should be column vectors
    '''
    F=np.abs(psi.conj().T@phi)**2
    return F[0,0]

def variance(psi,H):
    V=np.real((psi.conj().T@H@H@psi)-(psi.conj().T@H@psi)**2)
    return V[0,0]


#Hamiltonian Params Traverse Field Ising model
n_qubits=range(6,15)
D=range(2,7,2)
J=1
h=0.5
T=2

#evolution parameters
dt=0.1
N=100 #max number of steps
tmax=dt*N
t=0 + np.arange(0, N+1) *dt

# %%
for n in n_qubits:
    #initial state
    psi0np=np.zeros((2**n,1),dtype=complex)
    psi0np[0]=1
    psi_0=sp.csc_matrix(psi0np/np.linalg.norm(psi0np))
    H,H_trot=ham.TFIM(J,h,n,T=T)
    eig=sio.loadmat('eigsTFIM/N%d_J%1.1f_h%1.1f.mat'%(n,J,h))
    psigs=eig['psi0']
    for d in D:
        print("Computing QITE for N=",n,"D=",d)
        EQ,psi_QITE,a = evol.QITE(n,H,H_trot,d,psi_0,N,dt,vervose=False)
        NQ=len(EQ)
        var_Q=np.zeros(NQ)
        F_Q=np.zeros(NQ)
        for i in range(NQ):
            F_Q[i]=fidelity_pure(psigs,psi_QITE[:,i].todense())
            var_Q[i]=variance(psi_QITE[:,i],H)
        savedict = {
            'N' : n_qubits,
            'D' : D,
            'T' : T,
            'a' : a[0:NQ,:,:],
            'psi0' : psi_0,
            'psiQITE' : psi_QITE[:,0:NQ],
            'J' : J,
            'h' : h,
            't' : t[0:NQ],
            'E' : EQ,
            'var' : var_Q,
            'F' : F_Q
        }
        sio.savemat('runs_QITE/QITE_TFIM_J%1.1f_h%1.1f_N%d_D%d_T%d.mat'%(J,h,n,d,T), savedict)


