import numpy as np
import scipy
import cmath 
import scipy.sparse as sp
import PauliStrings as pauli_strings

def QITE(n_qubits,H,H_trot,D,psi_0,N,dt,psi_true,vervose=True,sparse=True,stopping_criteria=True):

    #checking whick method to obtain pauli strings is used
    if np.isreal(H.data).all() and np.isreal(psi_0.data).all():
        print("Using Real Pauli Strings") 
        num_paulis,PD,fail = pauli_strings.real(H_trot,D,n_qubits,sparse)
    else:
        print("Using General Pauli Strings")
        num_paulis,PD,fail = pauli_strings.general(H_trot,D,n_qubits,sparse)

    #chosing which routine of QITE will be used
    if fail:
        print("You need D>=T. Not running")
    else:
        if sparse:
            print("Sparse Routine")
            return QITE_sparse(n_qubits,H,H_trot,num_paulis,PD,psi_0,N,dt,psi_true,vervose,stopping_criteria)
        else:
            print("Dense Routine, it will be slower and use more memory")
            return QITE_dense(n_qubits,H,H_trot,num_paulis,PD,psi_0,N,dt,psi_true,vervose,stopping_criteria)

def QITE_sparse(n_qubits,H,H_trot,num_paulis,PD,psi_0,N,dt,psi_true,vervose=True,stopping_criteria=True):
    psi_out=sp.lil_matrix((2**n_qubits,N+1),dtype=complex)
    psi_out[:,0]=psi_0.copy()
    psi_QITE=psi_0.copy()
    a=np.zeros((N,H_trot.Nk,num_paulis),dtype=complex)
    E_QITE=np.zeros(N+1)
    F_QITE=np.zeros(N+1)
    varE_QITE=np.zeros(N+1)
    E_QITE[0]=np.real((psi_QITE.getH()@H@psi_QITE).trace())
    F_QITE[0]=(np.abs((psi_QITE.getH()@psi_true).trace()))**2
    varE_QITE[0]=np.real((psi_QITE.getH()@H@H@psi_QITE).trace())-E_QITE[0]**2
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
        E_QITE[i+1] = np.real((psi_QITE.getH()@H@psi_QITE).trace())
        F_QITE[i+1] = (np.abs((psi_QITE.getH()@psi_true).trace()))**2
        varE_QITE[i+1] = np.real((psi_QITE.getH()@H@H@psi_QITE).trace())-E_QITE[i+1]**2
        psi_out[:,i+1]=psi_QITE.copy()
        if vervose:
            print("Step",i+1,"/",N,"with energy",E_QITE[i+1])
        if stopping_criteria and E_QITE[i+1]>E_QITE[i]:
            print("Energy increased, stopping criteria activated at step",i+1)
            return E_QITE[:i+1],varE_QITE[:i+1],F_QITE[:i+1],psi_out[:,:i+1],a[:i]
    return E_QITE,varE_QITE,F_QITE,psi_out,a

def QITE_dense(n_qubits,H,H_trot,num_paulis,PD,psi_0,N,dt,psi_true,vervose=True,stopping_criteria=True):
    psi_out=np.zeros((2**n_qubits,N+1),dtype=complex)
    psi_out[:,0]=psi_0.copy()
    psi_QITE=psi_0
    a=np.zeros((N,n_qubits,num_paulis),dtype=complex)
    E_QITE=np.zeros(N+1)
    F_QITE=np.zeros(N+1)
    varE_QITE=np.zeros(N+1)
    E_QITE[0]=np.real((psi_QITE.T.conj()@H@psi_QITE))
    F_QITE[0]=(np.abs(psi_QITE.T.conj()@psi_true))**2
    varE_QITE[0]=np.real((psi_QITE.T.conj()@H@H@psi_QITE))-E_QITE[0]**2

    for i in range(0,N):
        #ara en aquest pas de temps anem a calcular els coeffs per a cada qubit
        for l in range(H_trot.Nk):
            #obtencio matrius S
            P=np.zeros((num_paulis,2**n_qubits,2**n_qubits),dtype=complex)
            for j in range(num_paulis):
                P[j,:,:]=PD[l][j]
            X=np.matmul(P,psi_QITE)
            S=X@X.conj().T
            HB=H_trot.Hk[l]
            #obtencio coefficients b
            aux=np.real((psi_QITE.conj().T@scipy.linalg.expm(-2*HB*dt)@psi_QITE))
            c=cmath.sqrt(aux)
            expH=scipy.linalg.expm(-HB*dt)
            auxO=np.matmul(np.matmul(expH,P)-np.matmul(P,expH),psi_QITE)
            b = -1j*np.matmul(auxO,psi_QITE.conj())/c/dt
            #obtencio coefficients a
            invS_ex=np.linalg.pinv(S+S.transpose())
            a[i,l]=np.real(invS_ex@b).flatten()
            
            #ara evolucionem el qbit amb els nous coefficients
            operator=np.zeros((2**n_qubits,2**n_qubits),dtype=complex)
            for j in range(num_paulis):
                operator+=a[i,l,j]*P[j,:,:]
            psi_QITE = scipy.linalg.expm(-1j*operator*dt)@psi_QITE
            
        #valor esperat energia
        E_QITE[i+1] = np.real((psi_QITE.conj().T@H@psi_QITE))
        F_QITE[i+1]=(np.abs(psi_QITE.T.conj()@psi_true))**2
        varE_QITE[i+1]=np.real((psi_QITE.T.conj()@H@H@psi_QITE))-E_QITE[i+1]**2
        psi_out[:,i+1]=psi_QITE.copy()
        if vervose:
            print("Step",i+1,"/",N,"with energy",E_QITE[i+1])
        if stopping_criteria and E_QITE[i+1]>E_QITE[i]:
            print("Energy increased, stopping criteria activated at step",i+1)
            return E_QITE[:i+1],varE_QITE[:i+1],F_QITE[:i+1],psi_out[:,:i+1],a[:i]
    return E_QITE,varE_QITE,F_QITE,psi_out,a


def ITE(H,psi_0,tmax,Nt):
    if sp.issparse(H) and sp.issparse(psi_0):
        return ITE_sparse(H,psi_0,tmax,Nt)
    else:
        return ITE_dense(H,psi_0,tmax,Nt)

def ITE_dense(H,psi_0,tmax,Nt):
    t_ITE=np.linspace(0,tmax,num=Nt)
    EITE=np.zeros(len(t_ITE))
    for i in range(len(t_ITE)):
        phi=scipy.linalg.expm(-H*t_ITE[i])@psi_0
        psiITE=phi/scipy.linalg.norm(phi)
        EITE[i]=np.real((psiITE.T.conj()@H@psiITE))
    return t_ITE,EITE

def ITE_sparse(H,psi_0,tmax,Nt):
    t_ITE=np.linspace(0,tmax,num=Nt)
    EITE=np.zeros(len(t_ITE))
    for i in range(len(t_ITE)):
        phi=sp.linalg.expm_multiply(-H*t_ITE[i],psi_0)
        psiITE=phi/sp.linalg.norm(phi)
        EITE[i]=np.real((psiITE.T.conj()@H@psiITE).trace())
    return t_ITE,EITE


def compute_U(n_qubits,H_trot,num_paulis,PD,psi_QITE,dt):
    UN=np.eye(2**n_qubits)
    a=np.zeros((H_trot.Nk,num_paulis),dtype=complex)
    for l in range(n_qubits):
        X=sp.lil_matrix((2**n_qubits,num_paulis),dtype=complex)
        b=np.zeros((num_paulis),dtype=complex)
        aux=np.real((psi_QITE.getH()@sp.linalg.expm(-2*H_trot.Hk[l]*dt)@psi_QITE).trace())
        c=cmath.sqrt(aux)
        expHdt=sp.linalg.expm(-H_trot.Hk[l]*dt)
        for j in range(num_paulis):
            b[j] = -1j*(psi_QITE.getH()@(expHdt@PD[l][j]-PD[l][j]@expHdt)@psi_QITE).trace()/c/dt
            X[:,j]=PD[l][j]@psi_QITE
        S=(X.getH()@X).todense()
        
        #least square solution of equation (S+S^T)*a = b (step 4 in algorithm)
        a[l]=(scipy.linalg.lstsq(S+S.T,b,lapack_driver='gelsd'))[0]
        
        #construction of the evolution operator (steps 5 and 6 in the algorithm)
        operator=sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
        for j in range(num_paulis):
            operator+=a[l,j]*PD[l][j]
        psi_QITE = sp.linalg.expm(-1j*operator*dt)@psi_QITE
        UN=sp.linalg.expm(-1j*operator*dt)@UN
    return UN

def compute_fuse_U(n_qubits,H_trot,num_paulis,PD,psi_QITE,dt,return_V_instead=False):
    ops = []
    a=np.zeros((H_trot.Nk,num_paulis),dtype=complex)
    A_sum = sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
    for l in range(n_qubits):
        X=sp.lil_matrix((2**n_qubits,num_paulis),dtype=complex)
        b=np.zeros((num_paulis),dtype=complex)
        aux=np.real((psi_QITE.getH()@sp.linalg.expm(-2*H_trot.Hk[l]*dt)@psi_QITE).trace())
        c=cmath.sqrt(aux)
        expHdt=sp.linalg.expm(-H_trot.Hk[l]*dt)
        for j in range(num_paulis):
            b[j] = -1j*(psi_QITE.getH()@(expHdt@PD[l][j]-PD[l][j]@expHdt)@psi_QITE).trace()/c/dt
            X[:,j]=PD[l][j]@psi_QITE
        S=(X.getH()@X).todense()
        
        #least square solution of equation (S+S^T)*a = b (step 4 in algorithm)
        a[l]=(scipy.linalg.lstsq(S+S.T,b,lapack_driver='gelsd'))[0]
        
        #construction of the evolution operator (steps 5 and 6 in the algorithm)
        operator=sp.csc_matrix((2**n_qubits,2**n_qubits),dtype=complex)
        for j in range(num_paulis):
            operator+=a[l,j]*PD[l][j]
        psi_QITE = sp.linalg.expm(-1j*operator*dt)@psi_QITE
        ops.append(operator)
        A_sum += operator
        
    if return_V_instead:
        def V(t):
            V_t = sp.eye(2**n_qubits)
            for op in ops:
                V_t = sp.linalg.expm(-1j*op*t)@V_t
            return V_t
        return V
    else:
        #return lambda t: sp.linalg.expm(-1j*A_sum*t)
        return A_sum
        #return scipy.sparse.linalg.expm_multiply(-1j*A_sum*deltaTau,psi_prev)


def adaptive_QITE(n_qubits,H,H_trot,D,psi_0,N,dt,always_recalculate=False,stopping_criteria=True):
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
        psi=psi_QITE[:,0]
        UN=compute_U(n_qubits,H_trot,num_paulis,PD,psi,dt)
        
        indx=[]
        fallo=0
        for i in range(0,N):
            #ara en aquest pas de temps anem a calcular els coeffs per a cada qubit
            psi_QITE[:,i+1]=UN@psi_QITE[:,i]
            E_test = np.real((psi_QITE[:,i+1].conj().T@H@psi_QITE[:,i+1]).trace())
            
            if E_test>E[i]:
                if fallo==0:
                    print("Computing U at step",i)
                    psi=psi_QITE[:,i]
                    #print(type(psi))
                    UN=compute_U(n_qubits,H_trot,num_paulis,PD,psi,dt)
                    #print(type(UN))
                    indx.append(i)
                psi_QITE[:,i+1]=UN@psi_QITE[:,i]
            E[i+1] = np.real((psi_QITE[:,i+1].conj().T@H@psi_QITE[:,i+1]).trace())
            if E[i+1] > E[i]:
                if stopping_criteria:
                    print("Energy doubly increased, stopping criteria activated at step",i+1)
                    return E[:i+1],psi_QITE[:,:i+1],indx
                if not always_recalculate:
                    fallo=1
                print("Fallo")
        return E,psi_QITE,indx
        
def adaptive_QITE_fuse(n_qubits,H,H_trot,D,psi_0,N,dt,psi_true,use_V=False,stopping_criteria=True):
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
        psi_QITE=[]
        psi_QITE.append(psi_0.copy())
        E=[]
        F_FUSE=[]
        varE_FUSE=[]
        E.append(np.real((psi_QITE[0].getH()@H@psi_QITE[0]).trace()))
        E_prev = E[0]
        F_FUSE.append(((np.abs((psi_QITE[0].getH()@psi_true).trace()))**2).item())
        varE_FUSE.append(np.real((psi_QITE[0].T.conj()@H@H@psi_QITE[0]).trace())-E[0]**2)
        
        indx = []
        steps = 0
        fallo = False
        while steps<N: 

            # recomputing U if energy increased
            if not fallo:
                psi_prev = psi_QITE[-1]
                t = dt #hyperparam 
                #print(type(psi_prev))
                UN = compute_fuse_U(n_qubits,H_trot,num_paulis,PD,psi_prev,dt,return_V_instead=use_V)
                #print(type(UN(t)))
                print(f'Computing {"V" if use_V else "U"} at step',steps)
                indx.append(steps)
                #psi_test = UN(t)@psi_prev
                psi_test = scipy.sparse.linalg.expm_multiply(-1j*UN*dt,psi_prev)
                E_test = np.real((psi_test.conj().T@H@psi_test).trace())
            

            # test if energy increased and if not evolve with the same U for more time
            t_diff=0
            while (E_test<E_prev or fallo) and steps<N:
                psi_QITE.append(psi_test)
                E.append(E_test)
                F_FUSE.append(((np.abs((psi_test.conj().T@psi_true).trace()))**2).item())
                varE_FUSE.append(np.real((psi_test.conj().T@H@H@psi_test).trace())-E_test**2)
                E_prev = E_test
                steps += 1
                t_diff += 1
                
                
                t += 0.1 #hyperparam
                #psi_test = UN(t)@psi_prev
                psi_test = scipy.sparse.linalg.expm_multiply(-1j*UN*(t_diff+1)*dt,psi_prev)
                E_test = np.real((psi_test.conj().T@H@psi_test).trace())

                if E_test<E_prev:
                    fallo = False     

            if indx[-1] == steps:
                #energy increased even after recalculation
                fallo = True
                if stopping_criteria:
                    print("Energy doubly increased, stopping criteria activated at step",steps+1)
                    return np.array(E),np.array(varE_FUSE),np.array(F_FUSE),sp.hstack(psi_QITE),indx

        return np.array(E),np.array(varE_FUSE),np.array(F_FUSE),sp.hstack(psi_QITE),indx



def adaptive_QITE_D(n_qubits,H,H_trot,maxD,psi_0,N,dt):
    
    psi_QITE=sp.lil_matrix((2**n_qubits,N+1),dtype=complex)
    psi_QITE[:,0]=psi_0.copy()
    E=np.zeros(N+1)
    E[0]=np.real((psi_QITE[:,0].getH()@H@psi_QITE[:,0]).trace())
    psi=psi_QITE[:,0]
    indx=[]
    
    D=H_trot.T
    num_paulis,PD,fail = pauli_strings.real(H_trot,D,n_qubits)
    UN=compute_U(n_qubits,H_trot,num_paulis,PD,psi,dt)
   
    fallo=0
    for i in range(0,N):
        #ara en aquest pas de temps anem a calcular els coeffs per a cada qubit
        psi_QITE[:,i+1]=UN@psi_QITE[:,i]
        E_test = np.real((psi_QITE[:,i+1].conj().T@H@psi_QITE[:,i+1]).trace())
        
        if E_test>E[i]:
            if fallo==0:
                print("Computing U at step",i)
                psi=psi_QITE[:,i]
                UN=compute_U(n_qubits,H_trot,num_paulis,PD,psi,dt)
                indx.append(i)
            psi_QITE[:,i+1]=UN@psi_QITE[:,i]
        E[i+1] = np.real((psi_QITE[:,i+1].conj().T@H@psi_QITE[:,i+1]).trace())
        if E[i+1] > E[i]:
            if D==maxD:
                fallo=1
                print("Fallo a temps",i)
            if fallo==0:
                D+=1
                print("Incrementat D:",D)
                num_paulis,PD,fail = pauli_strings.real(H_trot,D,n_qubits)
                UN=compute_U(n_qubits,H_trot,num_paulis,PD,psi,dt)
                psi_QITE[:,i+1]=UN@psi_QITE[:,i]
                E[i+1] = np.real((psi_QITE[:,i+1].conj().T@H@psi_QITE[:,i+1]).trace())
    return E,psi_QITE,indx
