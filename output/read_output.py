import pickle
import re
import matplotlib.pyplot as plt
import config as cfg

############################ Initialization ##################################
# Get parser from config
parser = cfg.get_args_parser()
parser.add_argument("--obs", type=str, default='energy', help="Observable to compute")
parser.add_argument("--res", type=str, default="output/obs/output_res", help="Observable to compute")
parser.add_argument("--coeff", type=str, default="output/obs/output_coeff_bin", help="Observable to compute")
args, unknown_args = parser.parse_known_args()

################################ Energy ######################################
def read_txt(file):
    liste_beta = []
    liste_val = []
    with open(file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            liste_beta.append(float(re.split(' ', line)[0]))
            liste_val.append(float(re.split('[ \n]', line)[1])) 
    return liste_beta, liste_val
    

def read_results(list_energy):
    res_b, res_val = read_txt("output/obs/res_didier.txt")
    res2_b, res2_val = read_txt("output/obs/res_QMC.txt")
    beta_list = [i*(1/8) for i in range(len(list_energy))]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set(xlabel=r"$\beta$", ylabel=r"$\langle H \rangle_T /site$")
    fig.suptitle("Imaginary time evolution of the bond energy")
    ax1.plot(beta_list, list_energy, 'bo-', markersize=3, label='Results')
    ax1.plot(beta_list[:15], [-(3/8)*val for val in beta_list[:15]], 
             linestyle='dashed', markersize=3, label=r'$\mathcal{O}(\tau)$')
    ax1.plot(res_b, res_val, markersize=3, label='Expected')
    ax1.plot(res2_b, res2_val, linestyle='dashed', markersize=3, label='QMC')
    ax2.set(xlabel=r"$\beta$", ylabel=r"$\langle H \rangle_T /site + (3/8) \times \tau$")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.plot(beta_list[:12], 
             [val1 + (3/8)*val2 for val1,val2 in zip(list_energy[:12],beta_list[:12])],
             'bo-', markersize=3, label='Results')
    ax2.plot(beta_list[:6], [-(3/32)*val**2 for val in beta_list[:6]],
             linestyle='dashed', markersize=3, label=r'$\mathcal{O}(\tau^2)$')
    ax2.plot(res_b[:13], 
             [val1 + (3/8)*val2 for val1,val2 in zip(res_val[:13],res_b[:13])],
             label='Expected')
    ax2.plot(res2_b[:30], 
             [val1 + (3/8)*val2 for val1,val2 in zip(res2_val[:30],res2_b[:30])],
             label='Expected')
    ax1.legend()
    ax2.legend()
    plt.show()

def plot_j1(file, j1, tau):
    res_b, res_val = read_txt("output/obs/res_didier.txt")
    res2_b, res2_val = read_txt("output/obs/res_QMC.txt")
    res_x, res_y = read_txt(file)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set(xlabel=r"$\beta$", ylabel=r"$\langle H \rangle_T /site$")
    fig.suptitle("Imaginary time evolution of the bond energy")
    ax1.plot(res_x, res_y, 'bo-', markersize=3, label='Results')
    ax1.plot(res_x[:15], [-(3/8)*val for val in res_x[:15]], 
             linestyle='dashed', markersize=3, label=r'$\mathcal{O}(\tau)$')
    ax1.plot(res_b, res_val, markersize=3, label='Expected')
    ax1.plot(res2_b, res2_val, linestyle='dashed', markersize=3, label='QMC')
    ax2.set(xlabel=r"$\beta$", ylabel=r"$\langle H \rangle_T /site + (3/8) \times \tau$")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.plot(res_x[:12], 
             [val1 + (3/8)*val2 for val1,val2 in zip(res_y[:12],res_x[:12])],
             'bo-', markersize=3, label='Results')
    ax2.plot(res_x[:6], [-(3/32)*val**2 for val in res_x[:6]],
             linestyle='dashed', markersize=3, label=r'$\mathcal{O}(\tau^2)$')
    ax2.plot(res_b[:13], 
             [val1 + (3/8)*val2 for val1,val2 in zip(res_val[:13],res_b[:13])],
             label='Expected')
    ax2.plot(res2_b[:30], 
             [val1 + (3/8)*val2 for val1,val2 in zip(res2_val[:30],res2_b[:30])],
             label='QMC')
    ax1.legend()
    ax2.legend()
    plt.show()
    
def plot_j2(file, j1, j2, tau):
    res_x, res_y = read_txt(file)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Imaginary time evolution of the bond energy")
    ax1.set(xlabel=r"$\beta$", ylabel=r"$\langle H \rangle_T /site$")
    ax1.plot(res_x, res_y,
             'bo-', markersize=3, label=f"$j_1$ = {j1}\n$j_2$ = {j2}\n"+r"$\tau$"+f" = {tau}")
    ax1.plot(res_x[:15], [-(3/8)*(j1**2+j2**2)*val for val in res_x[:15]],
             linestyle='dashed', markersize=3, label=r"$\mathcal{O}(\tau)$")
    ax2.set(xlabel=r"$\beta$", ylabel=r"$\langle H \rangle_T /site + \frac{3}{8}(j_1^2+j_2^2)\times \beta$")       
    ax2.plot(res_x[:15], [val1 + (3/8)*(j1**2+j2**2)*val2 for val1,val2 in zip(res_y,res_x)][:15],
             'bo-', markersize=3)
    ax2.plot(res_x[:15], [-(3/32)*(j1**3-6*j1**2*j2+j2**3)*val**2 for val in res_x][:15],
             linestyle='dashed', markersize=3, label=r"$\mathcal{O}(\tau^2)$")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax1.legend(); ax2.legend()
    plt.show()
        
    
############################## Coefficients ###################################
def make_dico(file):
    depickler = pickle.Unpickler(file)
    result = depickler.load()
    dico = {}
    for i in range(len(result[0])):
        dico[f"coef_{i}"]=[]
    for optim_step, res in enumerate(result):
        for i, coef in enumerate(res):
            dico[f"coef_{i}"].append(coef)
    return dico

def read_coeff(dico):
    plt.figure(dpi=200)
    plt.ylabel("Coefficients of the C4v tensor")
    plt.xlabel('Imaginary Time Steps')
    plt.title("Imaginary time evolution of symmetric C4v tensor")
    for key in dico.keys():
        plt.plot([i*(1/8) for i in range(len(dico[key]))], dico[key], markersize=3, label=f"{key}")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    file_res = open(args.res, "rb")
    file_coeff = open(args.coeff, "rb")
    depickler = pickle.Unpickler(file_res)
    results = depickler.load()
    read_coeff(make_dico(file_coeff))
    read_results(results[args.obs])