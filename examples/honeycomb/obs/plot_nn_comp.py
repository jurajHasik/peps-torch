import numpy as np
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre

def plot_corrf(dist, corrf, lw=0, markersize=0, ax=None, label=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(dist, corrf, markersize=markersize, markerfacecolor='none', lw=lw, markeredgewidth=1.5, label=label, **kwargs)

    return ax

def plot_nn_corr_nB_2dirn(site, file_nB_nB, file_nB_nA, ax=None):
    plt.style.use("science")
    colors = ['#009E73', '#D55E00']
    dist = 40

    markers = ['x', 'o']
    for j, dirn in enumerate([(1, 0), (0, 1)]):
        # filename = f"./FCI_3x3_N3/optchi_{optchi:d}/chi_{chi:d}/nB_nB_corrf_site_{i:d}_dirn_({dirn[0]},{dirn[1]}).npy"
        # filename = f"./FCI_3x3_N3/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nB_corrf_site_{i:d}_dirn_({dirn[0]},{dirn[1]}).npy"
        # filename = f"./CI_honeycomb_1x1/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nB_corrf_site_{i:d}_dirn_({dirn[0]},{dirn[1]}).npy"
        filename = file_nB_nB + f"_site_{site:d}_dirn_({dirn[0]},{dirn[1]}).npy"
        with open(filename, "rb") as f:
            dist1 = np.load(f)
            nB_nB_corrf = np.load(f)
        # ax = plot_corrf(dist1, nB_nB_corrf, markersize=5, marker='o', ax=ax, color=colors[j], label=f'{dirn}, BB')
        dist1 = dist1.astype(dtype=np.float64)
        dist1 = np.sqrt(3)*np.arange(1, len(dist1)+1)
        ax = plot_corrf(dist1, nB_nB_corrf, markersize=7.5, marker=markers[j], ax=ax, color=colors[j])

        # filename = f"./FCI_3x3_N3/optchi_{optchi:d}/chi_{chi:d}/nB_nA_corrf_site_{i:d}_dirn_({dirn[0]},{dirn[1]}).npy"
        # filename = f"./FCI_3x3_N3/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nA_corrf_site_{i:d}_dirn_({dirn[0]},{dirn[1]}).npy"
        # filename = f"./CI_honeycomb_1x1/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nA_corrf_site_{i:d}_dirn_({dirn[0]},{dirn[1]}).npy"
        filename = file_nB_nA + f"_site_{site:d}_dirn_({dirn[0]},{dirn[1]}).npy"

        with open(filename, "rb") as f:
            dist2 = np.load(f)
            nB_nA_corrf = np.load(f)

        # shift = +0.5 if dirn == (1, 0) else -0.5
        # ax = plot_corrf(dist2+shift, nB_nA_corrf, ax=ax, markersize=6, marker='*', color=colors[j], label=f'{dirn}, BA')
        dist2 = dist2.astype(dtype=np.float64)
        for i in range(len(dist2)):
            dist2[i] = np.sqrt((2*i+1)**2*3/4 + 0.25)
        # ax = plot_corrf(dist2+shift, nB_nA_corrf, ax=ax, markersize=8, marker='*', color=colors[j])
        ax = plot_corrf(dist2, nB_nA_corrf, ax=ax, markersize=7.5, marker=markers[j], color=colors[j])

        corrf = np.empty(nB_nB_corrf.size + nB_nA_corrf.size, dtype=nB_nB_corrf.dtype)
        dist = np.empty(dist1.size + dist2.size, dtype=np.float64)

        # if dirn == (1, 0):
        #     corrf[::2] = nB_nA_corrf
        #     corrf[1::2] = nB_nB_corrf
        #     # dist[::2] = dist2+shift
        #     dist[::2] = dist2
        #     dist[1::2] = dist1
        # else:
        #     dist1 = np.sqrt(3)*np.arange(1, len(dist1)+1)
        #     for i in range(len(dist2)):
        #         dist2[i] = np.sqrt((2*i+1)**2*3/4 + 0.25)
        #     corrf[::2] = nB_nA_corrf
        #     corrf[1::2] = nB_nB_corrf
        #     # dist[::2] = dist2+shift
        #     dist[::2] = dist2
        #     dist[1::2] = dist1
        # ax = plot_corrf(dist, corrf, lw=0, markersize=0, ax=ax, color=colors[j], zorder=0)

def compute_poly(rs, cn):
    g = 1 - np.exp(-rs**2/2)

    factor = np.exp(-rs**2/2)*rs**2
    for n, c in enumerate(cn, start=1):
        g += c*(-1)**n*factor*assoc_laguerre(rs**2, n-1, 2)/np.sqrt(np.pi*(n+1)*n)
    return g

def nn_Laughlin_1_3(rs):
    cn = [2.64496,  1.00274, -0.06065, -0.41040, -0.39510,\
        -0.26016, -0.12206, -0.02167,  0.03658,  0.06148,\
        0.06439,  0.05506,  0.04083,  0.02574,  0.01264,\
        0.00280, -0.00414, -0.00825, -0.01011, -0.01028]
    g = compute_poly(rs, cn)
    return g

def nn_IQHE(rs):
    cn = [0]
    g = compute_poly(rs, cn)
    return g


plt.style.use("science")
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5, 2))
fig.subplots_adjust(wspace=0.2)

l = 1.63
rs = np.linspace(0, l*10, 200)
g = nn_IQHE(rs)
axes[0].plot(rs/l, g, color='k', label=r"$\nu=1$", lw=1.5, ls='dashed')

rs = np.linspace(0, l*10, 200)
g = nn_Laughlin_1_3(rs)
axes[1].plot(rs/l, g, color='k', label=r"$\nu=1/3$", lw=1.5, ls='dashed')

optchi=108
chi=324
CI_nB_nB = f"./CI_honeycomb_1x1/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nB_corrf"
CI_nB_nA = f"./CI_honeycomb_1x1/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nA_corrf"
plot_nn_corr_nB_2dirn(0, CI_nB_nB, CI_nB_nA, ax=axes[0])
axes[0].set_xlabel(r"$|\boldsymbol{x}|/a$")
axes[0].set_ylabel(r"$g(\boldsymbol{x})$")
axes[0].text(0.3, 0.15, r"CI: $D=6, \chi=$"+f"{chi:d}", transform=axes[0].transAxes, ha="left", va="top", fontsize=10, fontweight="extra bold")

D=9
optchi=108
chi=468
FCI_nB_nB = f"./FCI_3x3_N3/D{D:d}/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nB_corrf"
FCI_nB_nA = f"./FCI_3x3_N3/D{D:d}/optchi_{optchi:d}/chi_{chi:d}/normalized_nB_nA_corrf"
plot_nn_corr_nB_2dirn(0, FCI_nB_nB, FCI_nB_nA, ax=axes[1])
axes[1].set_xlabel(r"$|\boldsymbol{x}|/a$")

axes[1].text(0.3, 0.15, r"FCI: $D=9, \chi=$"+f"{chi:d}", transform=axes[1].transAxes, ha="left", va="top", fontsize=10, fontweight="extra bold")


axes[0].legend(fontsize=10)
axes[1].legend(fontsize=10)

for ax in axes:
    ax.set_xlim([-0.5, 10])
    ax.set_ylim([-0.05, 1.15])
# plt.show()
plt.savefig(f"../figs/corrf/nn/normalized_nB_n_corrf_site_0_dirns.pdf")