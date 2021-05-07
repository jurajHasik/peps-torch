import pickle
import matplotlib.pyplot as plt

def make_list(file):
    depickler = pickle.Unpickler(file)
    result = depickler.load()
    list_energy = []
    for optim_step, res in enumerate(result):
        list_energy.append(res)
    return list_energy


def read_results(list_energy):
    plt.figure(dpi=200)
    plt.ylabel("Energy")
    plt.xlabel('Imaginary Time Steps')
    plt.title("Imaginary time evolution of the bond energy")
    plt.plot(range(len(list_energy)), list_energy, markersize=3)
    plt.show()

if __name__ == '__main__':
    file = open("output/output_energy", "rb")
    list_energy = make_list(file=file)
    read_results(list_energy=list_energy)