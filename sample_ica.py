import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from manopt.ica.off_diag_oblique import ica_hybrid_oblique
from manopt.ica.off_diag_stiefel import ica_hybrid_stiefel

from sklearn.decomposition import FastICA

def get_signal():
    np.random.seed(0)
    n_samples = 50
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.2 * np.random.normal(size=S.shape)  # Add noise

    S  = S - np.mean(S,axis=0)
    S /= S.std(axis=0)  # Standardize data
    # Mix data
    A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations
    return A,X,S

def show_resul(X,S,S_1,S_2):
    plt.figure()

    models = [X, S, S_1,S_2]
    names = [
        "Observations (mixed signal)",
        "True Sources",
        "ICA recovered signals (Riemannian Optimization)",
        "ICA recovered signals (FastICA)",
    ]

    
    colors = ["red", "steelblue", "orange"]

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(4, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.tight_layout()
    plt.savefig(".\ica_result.png")
    plt.close()

def main():
    A,X,S = get_signal()

    
    ica = FastICA(n_components=3)
    S_1 = ica.fit_transform(X)

    BetaTypes = ["DaiYuan","PolakRibiere", "Hybrid1", "Hybrid2"]
    for beta_type in BetaTypes:
        res = ica_hybrid_oblique(X,beta_type=beta_type)
        print("---------")
        print(beta_type)
        print_result(S,S_1,res.point)
        print("---------")
        print(beta_type)
        res = ica_hybrid_stiefel(X,beta_type=beta_type)
        print_result(S,S_1,res.point)

    show_resul(X,S,res.point,res.point)

def print_result(S,S_1,S_2):
    JJ1 = 0
    JJ2 = 0
    for i in range(S.shape[1]):
        J1 = []
        J2 = []
        for j in range(S.shape[1]):
            j1 = np.min([np.linalg.norm(S[:,i]-S_1[:,j]),np.linalg.norm(S[:,i]+S_1[:,j])])
            j2 = np.min([np.linalg.norm(S[:,i]-S_2[:,j]),np.linalg.norm(S[:,i]+S_2[:,j])])
            J1.append(j1)
            J2.append(j2)
        #print(np.min(J1))
        #print(np.min(J2))
        print((np.min(J2)-np.min(J1))/np.min(J2)*100)

if __name__ == "__main__":
    main()
