import matplotlib.pyplot as plt
import numpy as np

path = '/mnt/D484FDE484FDC94E/Documents/StageTN/TensorNetwork/tn-gl/'

# open a data file and create a list of the output
# Function that uses an iterator to get each value in the file
def parse_lines(open_file, sep, conversions):
    for line in open_file:
        # zip concatenates iterables (here a tuple of float types and values)
        # Then conv(item) works like float('value')
        yield [conv(item) for conv, item in
               zip(conversions, line.strip('\n').split(sep))]


# The plotter creates a dictionary with lists of values in the file
def dicter(filename):
    with open(filename, "r") as fp:
        len_line = len(fp.readline().split())
        fp.seek(0)  # Reset the cursor because of readline().split()
        dresult = {'coef_6': [], 'coef_13': []}
        for fields in parse_lines(fp, ' ',
                                  tuple(float for i in range(len_line))):
            dresult['coef_6'].append(fields[0])
            dresult['coef_13'].append(fields[1])
    return dresult

def plot_output(filename):
    plt.figure(dpi=200)
    dico = dicter(filename)
    plt.ylabel(r"optimized coefficients")
    plt.xlabel(r"$\tau$")
    plt.title(r"Coefficients optimized vs $\tau$")
    plt.plot(np.logspace(-3, 0, 10), dico['coef_6'], 'b', markersize=3,
                 label=r"$\lambda_0$")
    plt.plot(np.logspace(-3, 0, 10), dico['coef_13'], 'r', markersize=3,
                 label=r"$\lambda_1$")
    plt.plot(np.logspace(-3, 0, 10), 
             [val2/val1 for val1,val2 in zip(dico['coef_6'], dico['coef_13'])],
             label=r"$\frac{\lambda_1}{\lambda_0}$", color='green')
    plt.plot(np.logspace(-3, 0, 10), 
             [(2*val)**(1/2) for val in np.logspace(-3, 0, 10)],
             linestyle='dashed', color='black', label=r'$\sqrt{2*\tau}$')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    plot_output(path+'output/output_test.txt')