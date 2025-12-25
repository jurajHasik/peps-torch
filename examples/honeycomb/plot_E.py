from pathlib import Path
import re
from typing import Dict, Optional
import ast

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import matplotlib.cm as cm
from matplotlib.lines import Line2D

_energy_re = re.compile(r"loss:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_chi_re = re.compile(r"_chi_(\d+)")

def _last_energy(text: str) -> Optional[float]:
    last = None
    for m in _energy_re.finditer(text):
        last = m
    return float(last.group(1)) if last else None

def energies_by_chi(log_dir: str, lower_bound=0) -> Dict[int, float]:
    """
    Scan *.log files and return {chi: energy}.
    If multiple logs share the same chi, use the most recently modified file.
    Files without '_chi_<int>' in the name are ignored.
    """
    best = {}  # chi -> (mtime, energy)
    for fp in Path(log_dir).glob("*.log"):
        m = _chi_re.search(fp.name)
        if not m:
            continue
        chi = int(m.group(1))
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        energy = _last_energy(text)
        if energy is None:
            continue
        mtime = fp.stat().st_mtime
        if chi > lower_bound:
            # print(log_dir, chi)
            if (chi not in best) or (mtime > best[chi][0]):
                best[chi] = (mtime, energy)
    return {chi: energy*10 for chi, (_, energy) in best.items()} # t1=0.1



import ast
from pathlib import Path
from typing import Dict

def extract_densities(log_path: str) -> Dict[str, float]:
    """
    Read a log file and return the LAST densities dict printed like:
      INFO:__main__:{...}

    Works for lines such as:
      INFO:__main__:{"nA_Site(0, 0)": 0.1, "nB_Site(0, 0)": 0.2, ...}
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")

    last = None
    for line in text.splitlines():
        if "INFO:__main__:" not in line:
            continue
        i = line.find("{")
        j = line.rfind("}")
        if i == -1 or j == -1 or j <= i:
            continue
        blob = line[i:j+1]
        try:
            d = ast.literal_eval(blob)  # parses the {...} literal safely
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict) and all(isinstance(k, str) for k in d.keys()):
            last = d

    if last is None:
        raise ValueError(f"No densities dict found in {log_path}")

    return {k: float(v) for k, v in last.items()}


def plot_e_vs_invchi(chi_to_E, marker='o', ax=None, **kwargs):
    """
    Plot energy E versus 1/chi from a dict {chi: E}.
    Returns the matplotlib Axes.
    """
    # keep only positive chi and sort by chi
    items = sorted((int(k), float(v)) for k, v in chi_to_E.items() if int(k) > 0)
    if not items:
        raise ValueError("No valid chi>0 entries to plot.")

    chis = [k for k, _ in items]
    inv_chi = [1.0 / k for k in chis]
    Es = [v for _, v in items]

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(inv_chi, Es, lw=0, marker=marker, **kwargs)

    return ax

def plot_E0_vs_D(D_to_E0, marker: str = "o", ax=None, **kwargs):
    items = []
    for D, v in D_to_E0.items():
        D = int(D)
        E0, sE0 = float(v[0]), float(v[1])
        items.append((D, E0, sE0))

    if not items:
        raise ValueError("No valid D>0 entries to plot.")

    Darr = np.array([t[0] for t in items], dtype=float)
    y = np.array([t[1] for t in items], dtype=float)
    s = np.array([t[2] for t in items], dtype=float)

    if ax is None:
        _, ax = plt.subplots()

    for i in range(len(Darr)):
        ax.errorbar(Darr[i], y[i], yerr=s[i], lw=0, marker=marker, label=r"$D=$" + f"{int(Darr[i])}", **kwargs)

    return ax

def polyfit_e_vs_invchi(
    chi_to_E: Dict[int, float],
    deg: int = 1,
    ax=None,
    line_kw: Optional[dict] = None,
    show_e0_errorbar: bool = True,
    markersize=9, markeredgewidth=2.5,
):
    """
    Fit E versus 1/chi with a polynomial of degree `deg`.
    Optionally plot the fitted curve on `ax`, and (by default) an error bar for E0 at x=0.

    Returns (p, E0, sigma_E0, R2):
      - p: np.poly1d (evaluate at x = 1/chi)
      - E0: extrapolated E at chi->infinity (x=0)
      - sigma_E0: standard error of E0 from the coefficient covariance
      - R2: coefficient of determination on the data points
    """
    items = sorted((int(k), float(v)) for k, v in chi_to_E.items() if int(k) > 0)
    if len(items) < deg + 1:
        raise ValueError(f"Need at least {deg+1} points for a degree-{deg} fit.")
    chis = np.array([k for k, _ in items], dtype=float)
    x = 1.0 / chis
    y = np.array([v for _, v in items], dtype=float)

    # polynomial fit with covariance
    coeffs, cov = np.polyfit(x, y, deg=deg, cov=True)
    p = np.poly1d(coeffs)

    # quality (R^2)
    yhat = p(x)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # E0 and its standard error from covariance (intercept is last coefficient)
    E0 = float(p(0.0))
    sigma_E0 = float(np.sqrt(cov[-1, -1])) if cov.size else float("nan")

    if ax is not None:
        # fit curve
        xs = np.linspace(0.0, float(x.max()), 200)
        _kw = dict(lw=1.5, ls='--')
        if line_kw:
            _kw.update(line_kw)
        ax.plot(xs, p(xs), **_kw)

        # intercept marker + error bar at x=0
        col = _kw.get("color", None)
        if show_e0_errorbar:
            ax.errorbar(
                [0.0], [E0], yerr=[sigma_E0],
                fmt='x', ms=markersize, capsize=3, elinewidth=1.2,
                color=col, markeredgewidth=2.5
            )
        else:
            ax.plot([0.0], [E0], marker='x', ms=markersize, color=col, markeredgewidth=2.5)

    return p, E0, sigma_E0, R2


def ns_variance(ns, ave=1/6):
    tot = 0
    for n in ns.values():
        tot += (n - ave)**2

    return np.sqrt(tot/len(ns))/ave

def ns_max_dev(ns, ave=1/6):
    max_val = 0
    for n in ns.values():
        max_val = max(max_val, abs((n - ave)))

    return max_val/ave

def plot_densities(dens, fig=None, ax=None, colorbar=True):
    # rows increase along \tilde a2, columns along \tilde a1
    pattern = np.array([
        [0, 1, 2],  # row 0: A1 A2 A3
        [1, 2, 0],  # row 1: A2 A3 A1
        [2, 0, 1],  # row 2: A3 A1 A2
    ], dtype=int)
    label_from_idx = {0: "A1", 1: "A2", 2: "A3"}

    def density_for_cell(r, c, sublat):
        """Return nA or nB for blocked cell at integer (r,c) using explicit pattern."""
        idx = pattern[r % 3, c % 3]  # 0→A1, 1→A2, 2→A3
        return dens[f"n{sublat}_Site(0, {idx})"]

    # ======= geometry: honeycomb primitives and user's tilded basis =======
    alpha = 0.4
    t1 = np.array([0.5, -np.sqrt(3.0) / 2.0])*np.sqrt(3)*alpha
    t2 = np.array([0.5, +np.sqrt(3.0) / 2.0])*np.sqrt(3)*alpha


    # vertical split inside blocked cell for A and B (purely visual)
    delta = 0.5*alpha
    dA = np.array([0.0, -delta])
    dB = np.array([0.0,  delta])

    # ======= tiling size =======
    NX, NY = 4, 4  # cells along \tilde a1 (x) and \tilde a2 (y)

    # ======= build coordinates and colors =======
    X, Y, C = [], [], []     # all points (A and B), circles only
    cell_R = []              # for A1/A2/A3 labels at each Bravais origin

    for r in range(NY):
        for c in range(NX):
            R = c * t1 + r * t2
            cell_R.append((R[0], R[1], pattern[r % 3, c % 3]))

            nA = density_for_cell(r, c, "A")
            nB = density_for_cell(r, c, "B")

            RA = R + dA
            RB = R + dB

            X.extend([RA[0], RB[0]])
            Y.extend([RA[1], RB[1]])
            C.extend([nA, nB])

    X = np.array(X); Y = np.array(Y); C = np.array(C)

    # ======= plot =======
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.2, 6.6))

    lw=4
    for r in range(NY):
        for c in range(NX):
            R = c * t1 + r * t2

            RA = R + dA
            RB = R + dB
            ax.plot((RA[0], RB[0]), (RA[1], RB[1]), lw=lw, color='grey', zorder=0)
            ax.plot((RA[0], RB[0]-t2[0]), (RA[1], RB[1]-t2[1]), lw=lw, color='grey', zorder=0)
            ax.plot((RA[0], RB[0]+t1[0]), (RA[1], RB[1]+t1[1]), lw=lw, color='grey', zorder=0)

            ax.plot((RA[0]-t1[0], RB[0]), (RA[1]-t1[1], RB[1]), lw=lw, color='grey', zorder=0)
            ax.plot((RA[0]+t2[0], RB[0]), (RA[1]+t2[1], RB[1]), lw=lw, color='grey', zorder=0)

    sc = ax.scatter(X, Y, c=C, s=300, marker='o', vmin=0.16, vmax=0.18, edgecolors='black', linewidths=0.0, cmap=cm.plasma)
    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.04, ticks=[0.16, 0.17, 0.18])
        cbar.ax.tick_params(labelsize=30)
    # cbar.set_label("$n$", size=30)

    # place \tilde a1, \tilde a2 arrows near lower-left
    ax.set_xlim((-0.2, 2.3))
    ax.set_ylim((-1.2, 1.2))
    ax.set_xticks([]); ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_aspect('equal')
    return sc


from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # add this import

plt.style.use('science')


# --- main figure (E0 vs D) ---
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
fig.subplots_adjust(hspace=0, wspace=0.22)
axes[0].tick_params(axis="both", labelsize=13)
axes[1].tick_params(axis="both", labelsize=13)
axes[1].set_xticks([5, 6, 7, 8, 9])
axes[0].set_yticks([-0.86, -0.855, -0.85, -0.845])

ax_main = axes[0]
lower_bounds = {5:75, 6:72, 7:98, 8:112, 9:144}
degs = [2, 2, 2, 2, 2, 2]

E0_by_D = {}  # D -> (E0, sE0)

chis = [425, 432, 441, 448, 468]
Ds = [5, 6, 7, 8, 9]
ns_var = []
ns_data = []
for i, D in enumerate(Ds):
    d = energies_by_chi(f"FCI_data/states/D{D:d}", lower_bound=lower_bounds[D])
    logfile=f"FCI_data/states/D{D:d}/t1_0.1_3x3_N3_D_{D}_chi_{chis[i]}_fullrank_cuda.log"
    print(logfile)
    ns = extract_densities(logfile)
    ns_data.append(ns)
    # ns_var.append(ns_variance(ns))
    ns_var.append(ns_max_dev(ns))

    # plot chi-scaling in the inset
    plot_e_vs_invchi(d, ax=ax_main, label=rf"$D={D:d}$", markersize=8)
    col = ax_main.lines[-1].get_color()

    p, E0, sE0, R2 = polyfit_e_vs_invchi(
        d, deg=degs[i], ax=ax_main,
        line_kw={"color": col},
        show_e0_errorbar=True,
        markersize=8, markeredgewidth=2.5
    )
    E0_by_D[D] = (E0, sE0)
    print(D, E0, sE0)

# inset cosmetics
ax_main.set_xlabel(r"$\chi^{-1}$", fontsize=14)
ax_main.set_ylabel(r"$e_0/t_1$", fontsize=14)
ax_main.tick_params(axis="both", labelsize=12)

# main plot: E0 vs D (with error bars)
# ---- broken-axis inset (two stacked inset axes) ----
# overall inset box in *ax_main axes-fraction* coords: [x0, y0, w, h]
box = [0.205, 0.46, 0.75, 0.45]   # tune these
gap = 0.02                       # vertical gap between the two inset axes
h_each = (box[3] - gap) / 2

ax_in_bot = ax_main.inset_axes([box[0], box[1],               box[2], h_each],
                               transform=ax_main.transAxes)
ax_in_top = ax_main.inset_axes([box[0], box[1] + h_each + gap, box[2], h_each],
                               transform=ax_main.transAxes, sharex=ax_in_bot)

# plot the same inset content on both
plot_E0_vs_D(E0_by_D, marker="x", ax=ax_in_top, markersize=8, markeredgewidth=2.5)
plot_E0_vs_D(E0_by_D, marker="x", ax=ax_in_bot, markersize=8, markeredgewidth=2.5)

# choose y-windows (example: break around ~ -0.858)
ax_in_top.set_ylim(-0.8572, -0.8420)
ax_in_bot.set_ylim(-0.860, -0.8585)



# cosmetics for broken axis
ax_in_top.spines["bottom"].set_visible(False)
ax_in_bot.spines["top"].set_visible(False)
ax_in_top.tick_params(labelbottom=False)
ax_in_bot.set_xlabel(r"$D$", fontsize=12)
ax_in_bot.set_xticks([5,6,7,8,9])

# top inset: bottom ticks only
from matplotlib.ticker import NullLocator
ax_in_top.tick_params(axis="x", which='both', bottom=False, top=True, labelbottom=False, labeltop=False)

# bottom inset: top ticks only
ax_in_bot.tick_params(axis="x", which='both', bottom=True, top=False, labelbottom=True, labeltop=False)

# draw the diagonal "break marks"
d = 0.02
kw = dict(color="k", clip_on=False, linewidth=1)
ax_in_top.plot((-d, +d), (-d, +d), transform=ax_in_top.transAxes, **kw)
ax_in_top.plot((1-d, 1+d), (-d, +d), transform=ax_in_top.transAxes, **kw)
ax_in_bot.plot((-d, +d), (1-d, 1+d), transform=ax_in_bot.transAxes, **kw)
ax_in_bot.plot((1-d, 1+d), (1-d, 1+d), transform=ax_in_bot.transAxes, **kw)

ax_in_bot.xaxis.set_ticks_position("bottom")
ax_in_bot.set_xlabel(r"$D$", fontsize=12)
ax_in_bot.set_xticks([5,6,7,8,9])

handles, labels = ax_main.get_legend_handles_labels()

# choose which 2 go to the left column, and which 3 go to the right column
left_h, left_l   = handles[:2], labels[:2]
right_h, right_l = handles[2:], labels[2:]

dummy = Line2D([], [], linestyle="none", marker=None, alpha=0)

handles2 = left_h + [dummy] + right_h
labels2  = left_l + [""]     + right_l

leg = fig.legend(handles2, labels2, ncols=2, frameon=False,
                 handletextpad=0.1, columnspacing=0.4,
                 loc="lower left",
                 bbox_to_anchor=(0.32, 0.65),
                 bbox_transform=ax_main.transAxes)

# Hide the dummy row completely (handle + text)
for txt, h in zip(leg.get_texts(), leg.legend_handles):
    if txt.get_text() == "":
        txt.set_visible(False)
        h.set_visible(False)

leg.set_zorder(1000)
ax = axes[1]
ax.plot(Ds, ns_var, lw=0, marker='x', color='red', markersize=8, markeredgewidth=2.5, label=r"$\sqrt{\text{Var}( \langle \hat{n}_{\alpha, \boldsymbol{r}} \rangle)}/\bar{n}$")
ax.set_xlabel(r"$D$", fontsize=14)
ax.legend(fontsize=14, frameon=False, bbox_to_anchor=(0.19, 0.8), loc=3, handletextpad=0.05, markerfirst=False)
# ax.set_ylabel(r"$\sqrt{\text{Var}( n_i )}/\bar{n}$", fontsize=14)

plt.savefig("E_scaling.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(4, 4))
sc = plot_densities(ns_data[0], ax=ax, fig=fig, colorbar=False)
plt.savefig("density_D_5.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(4, 4))
sc = plot_densities(ns_data[-1], ax=ax, fig=fig)
plt.savefig("density_D_9.pdf", bbox_inches="tight")

# plt.tight_layout()