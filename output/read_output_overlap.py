import pickle
import numpy as np
import matplotlib.pyplot as plt


def make_dico(file):
    depickler = pickle.Unpickler(file)
    dico_res = depickler.load()
    return dico_res


def read_results_overlap(dico):
    plt.figure(dpi=200)
    plt.ylabel("Coefficients of B_tensor")
    plt.xlabel('Optimization Steps')
    plt.title("Optimization of the fidelity O")
    for key in dico.keys():
        plt.plot(range(len(dico[key])), dico[key], markersize=3, label=f"{key}")
    plt.legend()
    plt.show()


def read_results_fidelity(dico):
    plt.figure(dpi=200)
    plt.ylabel("Fidelity O")
    plt.xlabel('Optimization Steps')
    plt.title("Optimization of the fidelity O")
    w0l, w1l, w2l = [list_w for list_w in list(dico.values())][:]
    res = [w1/np.sqrt(w0*w2) for (w0,w1,w2) in zip(w0l,w1l,w2l)]
    plt.plot(range(len(list(dico.values())[0])), res)
    plt.show()


if __name__ == '__main__':
    file = open("output/output_overlap", "rb")
    dico = make_dico(file=file)
    read_results_overlap(dico=dico)
    read_results_fidelity(dico=dico)
