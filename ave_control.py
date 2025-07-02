import numpy as np
from scipy.linalg import schur
import numpy as np 
import scipy as sp
from scipy import sparse
import scipy.integrate
import scipy.linalg as la
from numpy.linalg import eig
from numpy import matmul as mm
from scipy.linalg import expm as expm
from numpy import transpose as tp
from packaging import version


def gramian(A_norm, T, system=None):
    # System Size
    n_nodes = A_norm.shape[0]
    B = np.eye(n_nodes)

    u, v = eig(A_norm)
    BB = mm(B, np.transpose(B))

    # If time horizon is infinite, can only compute the Gramian when stable
    if T == np.inf:
        # check system
        if system == 'continuous':
            # If stable: solve using Lyapunov equation
            if np.max(np.real(u)) < 0:
                return la.solve_continuous_lyapunov(A_norm, -BB)
            else:
                print("cannot compute infinite-time Gramian for an unstable system!")
                return np.nan
        elif system == 'discrete':
            # If stable: solve using Lyapunov equation
            if np.max(np.abs(u)) < 1:
                return la.solve_discrete_lyapunov(A_norm, BB)
            else:
                print("cannot compute infinite-time Gramian for an unstable system!")
                return np.nan
    # If time horizon is finite, perform numerical integration
    else:
        # check system
        if system == 'continuous':
            # Number of integration steps
            STEP = 0.2
            t = np.arange(0, (T+STEP/2), STEP)
            # Collect exponential difference 
            dE = sp.linalg.expm(A_norm * STEP) # how system evolves
            # print('A', dE)
            # Collect state transition matrix (Accumulated the transitions from the initial)
            dEa = np.zeros((n_nodes, n_nodes, len(t)))
            dEa[:, :, 0] = np.eye(n_nodes) # When t = 0, e to the power of At is I 
            # Collect Gramian difference
            dG = np.zeros((n_nodes, n_nodes, len(t)))
            dG[:, :, 0] = mm(B, B.T)
            for i in np.arange(1, len(t)):
                dEa[:, :, i] = mm(dEa[:, :, i-1], dE)
                # print(f'State Transition matrix of step {i}: \n{dEa[:, :, i]}')
                dEab = mm(dEa[:, :, i], B)
                # print(f'Control Influence matrix of step {i}: \n{dEab}')
                dG[:, :, i] = mm(dEab, dEab.T)

            # Integrate
            if sp.__version__ < '1.6.0':
                G = sp.integrate.simpson(dG, t, STEP, 2)
            else:
                G = sp.integrate.simpson(dG, t, STEP, 2)

            return G
        elif system == 'discrete':
            Ap = np.eye(n_nodes)
            Wc = np.eye(n_nodes)
            for i in range(T):
                Ap = mm(Ap, A_norm)
                Wc = Wc + mm(Ap, tp(Ap))

            return Wc
        
def ave_control(A_norm, system=None):
    if system is None:
        raise Exception("Time system not specified. "
                        "Please nominate whether you are normalizing A for a continuous-time or a discrete-time system "
                        "(see matrix_normalization help).")
    elif system != 'continuous' and system != 'discrete':
        raise Exception("Incorrect system specification. "
                        "Please specify either 'system=discrete' or 'system=continuous'.")
    elif system == 'discrete':
        T, U = schur(A_norm, 'real')  # Schur stability
        midMat = np.multiply(U, U).transpose()
        v = np.diag(T)[np.newaxis, :].transpose()
        N = A_norm.shape[0]
        P = np.diag(1 - np.matmul(v, v.transpose()))
        P = np.tile(P.reshape([N, 1]), (1, N))
        ac = sum(np.divide(midMat, P))

        return ac
    elif system == 'continuous':
        G = gramian(A_norm, T=1, system=system)
        ac = G.diagonal()

        return ac