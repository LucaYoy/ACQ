import numpy as np
import scipy
import cmath 
import scipy.sparse as sp
import PauliStrings as pauli_strings

def QITE(n_qubits,H,H_trot,D,psi_0,N,dt,vervose=False):

    #checking whick method to obtain pauli strings is used
    if np.isreal(H.data).all() and np.isreal(psi_0.data).all():
        #print("Using Real Pauli Strings") 
        num_paulis,PD,fail = pauli_strings.real(H_trot,D,n_qubits)
    else:
        #print("Using General Pauli Strings")
        num_paulis,PD,fail = pauli_strings.general(H_trot,D,n_qubits)

    psi_out=sp.lil_matrix((2**n_qubits,N+1),dtype=complex)
    psi_out[:,0]=psi_0.copy()
    psi_QITE=psi_0.copy()
    a=np.zeros((N,H_trot.Nk,num_paulis),dtype=complex)
    E_QITE=np.zeros(N+1)
    E_QITE[0]=np.real((psi_QITE.getH()@H@psi_QITE).trace())
    for i in range(0,N):
        #ara en aquest pas de temps anem a calcular els coeffs per a cada qubit
        for l in range(H_trot.Nk):
            #print('Time step',i+1,'/',N-1,'acting on qubit',l)
            #computing matrix S and coeffcient b (steps 1-3 in the algorithm)
            X=sp.lil_matrix((2**n_qubits,num_paulis),dtype=complex)
            b=np.zeros((num_paulis),dtype=complex)
            aux=np.real((psi_QITE.getH()@sp.linalg.expm(-2*H_trot.Hk[l]*dt)@psi_QITE).trace())
            c=cmath.sqrt(aux)
            expHdt=sp.linalg.expm(-H_trot.Hk[l]*dt)
            for j in range(num_paulis):
                b[j] = -1j*(psi_QITE.getH()@(expHdt@PD[l][j]-PD[l][j]@expHdt)@psi_QITE).trace()/c/dt
                X[:,j]=PD[l][j]@psi_QITE
            S=(X.getH()@X).todense()
           
            #obtencio coefficients a
            #invS_ex=np.linalg.pinv(S+S.T)
            #a[i,l]=np.real(invS_ex@b)

            #least square solution of equation (S+S^T)*a = b (step 4 in algorithm)
            a[i,l]=(scipy.linalg.lstsq(S+S.T,b,lapack_driver='gelsd'))[0]
            
            #construction of the evolution operator (steps 5 and 6 in the algorithm)
            operator=sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
            for j in range(num_paulis):
                operator+=a[i,l,j]*PD[l][j]
            psi_QITE = sp.linalg.expm(-1j*operator*dt)@psi_QITE
            
        #Enegy of the state at time step i
        if np.real((psi_QITE.getH()@H@psi_QITE).trace())>E_QITE[i]:
            print('Energy doubly increased at step',i)
            break
        E_QITE[i+1] = np.real((psi_QITE.getH()@H@psi_QITE).trace())
        psi_out[:,i+1]=psi_QITE.copy()
        if vervose:
            print("Step",i+1,"/",N,"with energy",E_QITE[i+1])
    return E_QITE[0:i+1],psi_out[:,0:i+1],a[0:i+1,:,:]

def compute_fuse_U(n_qubits,H_trot,num_paulis,PD,psi_QITE,dt,lstsq=True):
    a=np.zeros((H_trot.Nk,num_paulis),dtype=complex)
    A_sum = sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
    for l in range(H_trot.Nk):
        X=sp.lil_matrix((2**n_qubits,num_paulis),dtype=complex)
        b=np.zeros((num_paulis),dtype=complex)
        aux=np.real((psi_QITE.getH()@sp.linalg.expm(-2*H_trot.Hk[l]*dt)@psi_QITE).trace())
        c=cmath.sqrt(aux)
        expHdt=sp.linalg.expm(-H_trot.Hk[l]*dt)
        for j in range(num_paulis):
            b[j] = -1j*(psi_QITE.getH()@(expHdt@PD[l][j]-PD[l][j]@expHdt)@psi_QITE).trace()/c/dt
            X[:,j]=PD[l][j]@psi_QITE
        S=(X.getH()@X).todense()
        
        if lstsq:
            #least square solution of equation (S+S^T)*a = b (step 4 in algorithm)
            a[l]=(scipy.linalg.lstsq(S+S.T,b,lapack_driver='gelsd'))[0]
        else:
            #pseudo-inverse method
            invS_ex=scipy.linalg.pinvh(S+S.transpose())
            a[l]=np.real(invS_ex@b).flatten()
        
        #construction of the evolution operator (steps 5 and 6 in the algorithm)
        operator=sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
        for j in range(num_paulis):
            operator+=a[l,j]*PD[l][j]
        psi_QITE = sp.linalg.expm(-1j*operator*dt)@psi_QITE
        A_sum += operator
        
    return A_sum, lambda t: sp.linalg.expm(-1j*A_sum*t)

        
def ACQ(n_qubits,H,H_trot,D,psi_0,N,dt,failstop=True,expm=False):
    #checking whick method to obtain pauli strings is used
    if np.isreal(H.data).all() and np.isreal(psi_0.data).all():
        print("Using Real Pauli Strings") 
        num_paulis,PD,fail = pauli_strings.real(H_trot,D,n_qubits)
    else:
        print("Using General Pauli Strings")
        num_paulis,PD,fail = pauli_strings.general(H_trot,D,n_qubits)
    
    if fail:
        print("You need D>=T. Not running")
    else:
        psi_QITE=sp.lil_matrix((2**n_qubits,N+1),dtype=complex)
        psi_QITE[:,0]=psi_0.copy()
        
        E=np.zeros(N+1)
        E[0]=np.real((psi_QITE[:,0].getH()@H@psi_QITE[:,0]).trace())
        E_prev = E[0]        
        
        indx = []
        steps = 0
        fallo = False
        while steps<N: 

            # recomputing U if energy increased
            if not fallo:
                psi_prev = psi_QITE[:,steps]
                t = dt #times at which we probe the energies, dt is hyperparam of line search
                print("Computing U at step",steps)
                An,UN = compute_fuse_U(n_qubits,H_trot,num_paulis,PD,psi_prev,dt)
                indx.append(steps)
                psi_test = UN(t)@psi_prev
                E_test = np.real((psi_test.conj().T@H@psi_test).trace())
            
            # test if energy increased and if not evolve with the same U for more time 
            while (E_test<E_prev or fallo) and steps<N:
                
                #increase one step
                steps += 1
                t += dt #hyperparam
                psi_QITE[:,steps]=psi_test
                E[steps]=E_test
                E_prev = E_test
                
                if expm:
                    psi_test = sp.linalg.expm_mulitply(-1j*An*t,psi_prev)
                else:
                    psi_test = UN(t)@psi_prev

                #test energies
                E_test = np.real((psi_test.conj().T@H@psi_test).trace())
                if E_test<E_prev:
                    fallo = False     

            if indx[-1] == steps:
                #energy increased even after recalculation
                fallo = True
                if failstop:
                    print("Energy doubly increased, stopping criteria activated at step",steps+1)
                    return E,psi_QITE,indx

        return E,psi_QITE,indx

