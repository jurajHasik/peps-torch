import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import assoc_laguerre
from scipy.optimize import curve_fit


def plot_corrf(dist, corrf, lw=0, markersize=0, ax=None, label=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(dist, corrf, markersize=markersize, lw=lw, markeredgewidth=1.5, label=label, **kwargs)

    return ax

def plot_ccp_corr(D, opt_chi, chis, site, dirn=(1, 0), ax=None, cmap='viridis', vmin=None, vmax=None, **kwargs):
    # cmap = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
    # if vmin is None: vmin = np.min(chis)
    # if vmax is None: vmax = np.max(chis)
    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    for chi in chis:
        # color = cmap(norm(chi))
        if dirn == (1, 0):
            filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(1,0).npy"
            # filename = f"./FCI_3x3_N3/D4/optchi_64/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(1,0).npy"
        else:
            filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(0,1).npy"
            # filename = f"./FCI_3x3_N3/D4/optchi_64/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(0,1).npy"
        with open(filename, "rb") as f:
            dist1 = np.load(f)
            cA_cpA_corrf = np.load(f)

        ax = plot_corrf(dist1*np.sqrt(3), np.abs(cA_cpA_corrf), markersize=5, marker='o', lw=1.5, markerfacecolor='none', label=f'$\chi={chi:d}$', ax=ax, **kwargs)
        # ax = plot_corrf(dist1, np.abs(cA_cpA_corrf), markersize=0, marker='o', color='black', label=r'AA', lw=1, zorder=0, ax=ax)

    return ax

def collect_c_cp_bulk(D, opt_chi, chis, num_points, site, dirn=(1, 0)):
    x, y = [], []
    for i, chi in enumerate(chis):
        if dirn == (1, 0):
            filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(1,0).npy"
        else:
            filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(0,1).npy"
        with open(filename, "rb") as f:
            dist1 = np.load(f)
            cA_cpA_corrf = np.load(f)
            x.append(dist1[:num_points[i]]*np.sqrt(3))
            y.append(np.abs(cA_cpA_corrf)[:num_points[i]])

    return np.concatenate(x), np.concatenate(y)


def linear_fit(x, y, bounds_tau_positive=True, p0=None, maxfev=20000):
    """
    Returns
    -------
    result : dict
        Contains:
          - popt: best-fit params
          - perr: 1σ parameter errors
          - pcov: covariance matrix
          - model_func: callable f(x, *params)
          - x_fit, y_fit, sigma: data actually used in the fit (possibly aggregated)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Choose model
    def f(x, A, tau):
        return np.log(A) - x / tau
    param_names = ("A", "tau")
    npar = 2

    x_fit, y_fit = x, y

    if p0 is None:
        # sort by x for tail estimate
        idx = np.argsort(x_fit)
        xs = x_fit[idx]
        ys = y_fit[idx]
        span = xs.max() - xs.min()
        tau0 = span / 3 if span > 0 else 1.0
        A0 = float(ys.max())
        p0 = (A0, tau0)

    # Bounds
    if bounds_tau_positive:
        lower = [-np.inf] * npar
        upper = [ np.inf] * npar
        # tau is always the 2nd parameter
        lower[1] = 0.0
        bounds = (lower, upper)
    else:
        bounds = (-np.inf, np.inf)

    popt, pcov = curve_fit(
        f, x_fit, y_fit,
        p0=p0,
        bounds=bounds,
        maxfev=maxfev
    )
    perr = np.sqrt(np.diag(pcov))

    return {
        "popt": popt,
        "perr": perr,
        "pcov": pcov,
        "param_names": param_names,
        "model_func": f,
        "x_fit": x_fit,
        "y_fit": y_fit,
    }



plt.style.use("science")
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(5, 2))
fig.subplots_adjust(wspace=0.1)

ax = axes[0]
site=0
chi=98

# for D, opt_chi, chi in [(4, 64, 320), (6, 72, 432), (7, 98, 441)]:
# for D, opt_chi, chi in [(7, 98, 441), (8, 112, 448), (9, 108, 468)]:
#     filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(1,0).npy"
#     with open(filename, "rb") as f:
#         dist1 = np.load(f)
#         cA_cpA_corrf = np.load(f)
#     ax = plot_corrf(dist1*np.sqrt(3), np.abs(cA_cpA_corrf), markersize=5, marker='o', lw=1.5, markerfacecolor='none', \
#         label=r'$D=$' + f'{D:d}'+'$, \chi=$' + f'{chi:d}', ax=ax,)
#     #  color='#555555'

colors = ['#009E73', '#D55E00']
for D, opt_chi, chi in [(9, 108, 468)]:
    filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(1,0).npy"
    with open(filename, "rb") as f:
        dist1 = np.load(f)
        cA_cpA_corrf = np.load(f)
    ax = plot_corrf(dist1*np.sqrt(3), np.abs(cA_cpA_corrf), markersize=5, marker='o', lw=1.5, markerfacecolor='none', \
        label=r'$\boldsymbol{a_1}$', ax=ax, color=colors[0])

    filename = f"./FCI_3x3_N3/D{D}/optchi_{opt_chi}/chi_{chi:d}/cA_cpA_corrf_site_{site:d}_dirn_(0,1).npy"
    with open(filename, "rb") as f:
        dist1 = np.load(f)
        cA_cpA_corrf = np.load(f)
    ax = plot_corrf(dist1*np.sqrt(3), np.abs(cA_cpA_corrf), markersize=5, marker='x', lw=1.5, markerfacecolor='none', \
        label=r'$\boldsymbol{a_2}$', ax=ax,color=colors[1])

ax.text(0.09, 0.15, r"(a) $D=9, \chi=468$", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="extra bold")

uniq = {}
for h, l in zip(*ax.get_legend_handles_labels()):
    if l not in uniq:
        uniq[l] = h
labels = list(uniq.keys())
handles = list(uniq.values())

order = np.argsort(labels)
ax.legend([handles[i] for i in order], [labels[i] for i in order], ncols=1, handlelength=1.4, fontsize=10, columnspacing=0.5, handletextpad=0.3, loc=3, bbox_to_anchor=(0.6, 0.6))


ax.set_ylim((5e-13, 2e-1))
ax.set_xlim((-1, 120))

ax.set_ylabel(r"$\langle \hat{c}_{\text{A},0} \hat{c}^\dagger_{\text{A},\boldsymbol{r}}\rangle$")
ax.set_xlabel(r"$|\boldsymbol{r}|/a$")

ax = axes[1]
D, opt_chi = 9, 108
# chis = [49, 98, 147, 196, 245, 441]
chis = [72, 108, 180, 252, 324, 468]
# chis = [180, 468]
# chis = [96, 128, 160, 192, 224]
plot_ccp_corr(D, opt_chi, chis=chis, site=0, dirn=(0, 1), ax=ax)

num_points = [4]*6
x, y = collect_c_cp_bulk(D, opt_chi, chis, num_points, site, dirn=(1, 0))
print(x, y)
p0 = (0.2, 0.6)
res = linear_fit(x, np.log(y), p0=p0)
print(dict(zip(res["param_names"], res["popt"])))
print("1σ:", dict(zip(res["param_names"], res["perr"])))

xs = np.linspace(1.5, 30, 100)
A, tau = res["popt"]
ax.plot(xs, A*np.exp(-xs/tau), ls='--', color='black')

ax.set_yscale('log')
ax.set_xlabel(r"$|\boldsymbol{r}|/a$")
ax.text(0.2, 0.15, r"(b)", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="extra bold")
ax.text(0.35, 0.55, r"$\xi_{\mathrm{bulk}}/a=$"+f'{tau:.2f}', transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="extra bold")

from matplotlib.lines import Line2D
handles, labels = ax.get_legend_handles_labels()
seen = set()
h, l = [], []
for hd, lb in zip(handles, labels):
    if lb not in seen:
        seen.add(lb)
        h.append(hd)
        l.append(lb)

# --- build 2-column layout: col1=[first], col2=[rest] ---
spacer = Line2D([], [], linestyle='None', marker=None, alpha=0)  # invisible
# h2 = [h[0], h[1]] + h[2:]
# l2 = [l[0], l[1]] + l[2:]
h2 = h
l2 = l

ax.legend(
    h2, l2,
    ncol=2, handletextpad=0.3, fontsize=9.5, handlelength=1.4,
    labelspacing=0.4, columnspacing=0.7, frameon=False, loc=3, bbox_to_anchor=(0.05, 0.55)
)

# ax.legend(ncols=2, fontsize=10, columnspacing=0.5, handletextpad=0.3, loc=3, bbox_to_anchor=(0.1, 0.55))
ax.set_xlim((-1, 120))
ax.set_ylim((1e-11, 1.8e-1))

plt.savefig("../figs/corrf/c_cp_corr.pdf")