import os, logging

os.environ["OMP_NUM_THREADS"] = "16"      # For OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "16" # For OpenBLAS
os.environ["MKL_NUM_THREADS"] = "16"      # For Intel MKL


import torch

import context
import config as cfg
import yastn.yastn as yastn
from yastn.yastn.sym import sym_U1
from yastn.yastn.tensor import eigh, exp, ncon, tensordot
from yastn.yastn.tn.fpeps import fkron, EnvNTU, EnvBP, evolution_step_, accumulated_truncation_error
from yastn.yastn.tn.fpeps.gates import Gate_local, decompose_nn_gate, decompose_nnn_gate, Gates
from tqdm import tqdm

from models.fermion.tv_model import *

def compute_gates(state, model, db=0.01):
    n_A = model.sf.n(spin="u")  # parity-even operator, no swap gate needed
    n_B = model.sf.n(spin="d")
    c_A = model.sf.c(spin="u")
    cp_A = model.sf.cp(spin="u")
    c_B = model.sf.c(spin="d")
    cp_B = model.sf.cp(spin="d")
    I = model.sf.I()

    local, nn, nnn = [], [], []
    for site in state.sites():
        # onsite
        onsite_op = (
            0 * I
            + model.V1 * (n_A @ n_B)
            - model.mu * (n_A + n_B)
            - model.t1 * (cp_A @ c_B + cp_B @ c_A)
            + model.m * (n_A - n_B)
        )
        D, S = eigh(onsite_op, axes = (0, 1))
        D = exp(D, step=-db/2)
        G_local = ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))

        local.append(Gate_local(G_local, site))

    for site in state.sites():
        # horizontal bonds
        h_bond = Bond(site, state.nn_site(site, "r"))

        # important
        Hh = 0*fkron(I, I, sites=(0, 1))
        Hh = Hh + model.V1*fkron(n_B, n_A, sites=(0, 1)) + model.V2*fkron(n_B, n_B, sites=(0, 1)) + model.V2*fkron(n_A, n_A, sites=(0, 1))
        Hh = Hh - model.t1*fkron(cp_A, c_B, sites=(1, 0)) - model.t1*fkron(cp_B, c_A, sites=(0, 1))
        Hh = Hh -model.t2*np.exp(1j*model.phi)*fkron(cp_A, c_A, sites=(1, 0)) \
            -model.t2*np.exp(-1j*model.phi)*fkron(cp_A, c_A, sites=(0, 1)) \
            -model.t2*np.exp(1j*model.phi)*fkron(cp_B, c_B, sites=(0, 1)) \
            -model.t2*np.exp(-1j*model.phi)*fkron(cp_B, c_B, sites=(1, 0)) \

        Hh = Hh.fuse_legs(axes=((0, 1), (2, 3)))
        D, S = eigh(Hh, axes=(0, 1))
        D = exp(D, step=-db/2)
        G_h = ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))
        G_h = G_h.unfuse_legs(axes=(0, 1))
        nn.append(decompose_nn_gate(G_h, bond=h_bond))

        # vertical bonds
        v_bond = Bond(site, state.nn_site(site, "b"))

        # important
        Hv = 0*fkron(I, I, sites=(0, 1))
        Hv = Hv + model.V1*fkron(n_A, n_B, sites=(0, 1)) + model.V2*fkron(n_B, n_B, sites=(0, 1)) + model.V2*fkron(n_A, n_A, sites=(0, 1))
        Hv = Hv - model.t1*fkron(cp_A, c_B, sites=(0, 1)) - model.t1*fkron(cp_B, c_A, sites=(1, 0))
        Hv = Hv - model.t2*np.exp(1j*model.phi)*fkron(cp_A, c_A, sites=(1, 0)) \
            - model.t2*np.exp(-1j*model.phi)*fkron(cp_A, c_A, sites=(0, 1)) \
            - model.t2*np.exp(1j*model.phi)*fkron(cp_B, c_B, sites=(0, 1)) \
            - model.t2*np.exp(-1j*model.phi)*fkron(cp_B, c_B, sites=(1, 0)) \

        Hv = Hv.fuse_legs(axes=((0, 1), (2, 3)))
        D, S = eigh(Hv, axes=(0, 1))
        D = exp(D, step=-db/2)
        G_v = ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))
        G_v = G_v.unfuse_legs(axes=(0, 1))
        nn.append(decompose_nn_gate(G_v, bond=v_bond))

        if model.V2 != 0 or model.V3 != 0 or model.t2 != 0 or model.t3 != 0:
            s0 = site
            s1, s2, s3 = state.nn_site(s0, "b"), state.nn_site(s0, "r"), state.nn_site(s0, (1, 1))
            # diagonal bonds
            H_diag = 0*fkron(I, I, sites=(0, 1))
            H_diag = H_diag + model.V2*fkron(n_A, n_A, sites=(0, 1)) + model.V2*fkron(n_B, n_B, sites=(0, 1)) \
                    + model.V3*fkron(n_A, n_B, sites=(0, 1)) + model.V3*fkron(n_B, n_A, sites=(0, 1))
            H_diag = H_diag - model.t2*np.exp(1j*model.phi)*fkron(cp_A, c_A, sites=(0, 1)) \
                    - model.t2*np.exp(-1j*model.phi)*fkron(cp_A, c_A, sites=(1, 0)) \
                    - model.t2*np.exp(1j*model.phi)*fkron(cp_B, c_B, sites=(1, 0)) \
                    - model.t2*np.exp(-1j*model.phi)*fkron(cp_B, c_B, sites=(0, 1)) \
                    - model.t3*fkron(cp_A, c_B, sites=(1, 0)) \
                    - model.t3*fkron(cp_B, c_A, sites=(0, 1)) \
                    - model.t3*fkron(cp_B, c_A, sites=(1, 0)) \
                    - model.t3*fkron(cp_A, c_B, sites=(0, 1))

            H_diag = H_diag.fuse_legs(axes=((0, 1), (2, 3)))
            D, S = eigh(H_diag, axes=(0, 1))
            D = exp(D, step=-db/4) # extra factor of 2 due to two choices of the corners
            G_diag = ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))
            G_diag = G_diag.unfuse_legs(axes=(0, 1))

            # insert an identity operator between two legs
            G_tmp = decompose_nn_gate(G_diag)
            G0 = tensordot(G_tmp.G0, I, axes=((), ())) # p0 p0' c p1 p1'
            G0 = G0.swap_gate(axes=(2, 3)) # c x p1
            G_diag = tensordot(G0, G_tmp.G1, axes=(2, 2)) # p0 p0' p1 p1' p2 p2'
            G_diag = G_diag.transpose(axes=(0, 2, 4, 1, 3, 5))

            # dirn = "br", corner = "tr"
            nnn.append(decompose_nnn_gate(G_diag, s0, s2, s3))
            # dirn = "br", corner = "tr"
            nnn.append(decompose_nnn_gate(G_diag, s0, s1, s3))

            # anti-diagonal bonds
            s1, s2 = state.nn_site(s0, "b"), state.nn_site(s0, "r")
            H_anti_diag = 0*fkron(I, I, sites=(0, 1))
            H_anti_diag = H_anti_diag + model.V3*fkron(n_B, n_A, sites=(0, 1)) \
                        - model.t3*fkron(cp_A, c_B, sites=(1, 0)) \
                        - model.t3*fkron(cp_B, c_A, sites=(0, 1))
            H_anti_diag = H_anti_diag.fuse_legs(axes=((0, 1), (2, 3)))
            D, S = eigh(H_anti_diag, axes=(0, 1))
            D = exp(D, step=-db/4) # extra factor of 2 due to two choices of the corners
            G_anti_diag = ncon((S, D, S), ([-1, 1], [1, 2], [-3, 2]), conjs=(0, 0, 1))

            # dirn = "tr", corner = "tl"
            G_anti_diag1 = fkron(I, G_anti_diag) # p0 [p1 p2] p0' [p1' p2']
            G_anti_diag1 = G_anti_diag1.unfuse_legs(axes=(1, 3))
            nnn.append(decompose_nnn_gate(G_anti_diag1, s0, s1, s2))
            # dirn = "tr", corner = "br"
            G_anti_diag2 = fkron(G_anti_diag, I) # [p1 p2] p3 [p1' p2'] p3'
            G_anti_diag2 = G_anti_diag2.unfuse_legs(axes=(0, 2))
            nnn.append(decompose_nnn_gate(G_anti_diag1, s1, s2, s3))

    return Gates(nn=nn, nnn=nnn, local=local)
parser = cfg.get_args_parser()
parser.add_argument(
    "--V1", type=float, default=0.0, help="Nearest-neighbor interaction"
)
parser.add_argument(
    "--V2", type=float, default=0.0, help="2nd. nearest-neighbor interaction"
)
parser.add_argument(
    "--V3", type=float, default=0.0, help="3rd. nearest-neighbor interaction"
)
parser.add_argument(
    "--t1", type=float, default=1.0, help="Nearest-neighbor hopping amplitude"
)
parser.add_argument(
    "--t2", type=float, default=0.0, help="2nd. nearest-neighbor hopping amplitude"
)
parser.add_argument(
    "--t3", type=float, default=0.0, help="2nd. nearest-neighbor hopping amplitude"
)
parser.add_argument(
    "--phi", type=float, default=0.0, help="phase of the 2nd. nearest-neighbor hopping"
)
parser.add_argument("--mu", type=float, default=0.0, help="chemical potential")
parser.add_argument("--m", type=float, default=0.0, help="Semenoff mass")

def BP_conv_check_diff(env_prev, env):
    """
    Check convergence of the BP environment.
    """
    if env_prev is None or env is None:
        return torch.inf
    diff = 0.0
    for site in env_prev.sites():
        for d in ['t', 'l', 'b', 'r']:
            try:
                diff += (getattr(env_prev[site], d) - getattr(env[site], d)).norm(p='inf').item()
            except YastnError:
                return torch.inf
    return diff

if __name__ == "__main__":
    args = parser.parse_args()  # process command line arguments
    args.phi = 0.5 * np.pi
    cfg.configure(args)

    log = logging.getLogger(__name__)
    yastn_config = yastn.make_config(
        backend='torch',
        sym=sym_U1,
        fermionic=True,
        default_device='cpu',
        default_dtype='complex128',
        tensordot_policy="no_fusion",
    )
    torch.manual_seed(args.seed)
    model = tV_model(yastn_config, V1=args.V1, V2=args.V2, V3=args.V3, t1=args.t1, t2=args.t2, t3=args.t3, phi=args.phi, mu=args.mu, m=args.m)
    if cfg.main_args.instate is None:
        bond_dims = {1:1, 0:1, -1:1}
        state = random_2x2_state_U1(bond_dims=bond_dims, config=yastn_config)
    else:
        state = load_PepsAD(yastn_config, cfg.main_args.instate)
        log.log(logging.INFO, "loaded " + cfg.main_args.instate)
        print("loaded ", cfg.main_args.instate)
    outputstatefile= cfg.main_args.out_prefix+"_state.json"

    D = cfg.main_args.bond_dim
    opts_svd = {'D_total': D, 'tol': 1e-12}
    infoss = []

    # BP-update
    betas, dbs, env_types = [10, 5, 1], [1e-2, 1e-3, 1e-4], ['BP', 'BP', 'BP']
    weights_tol = 1e-8

    losses = []
    for beta, db, env_type in zip(betas, dbs, env_types):
        steps = round(beta / db)
        db = beta / steps
        gates = compute_gates(state, model, db=db)
        env_prev = None

        env = EnvBP(state, which=env_type)
        out = env.iterate_(max_sweeps=400, iterator_step=None, diff_tol=1e-8)
        # log.log(logging.INFO, f" Start, loss: {model.energy_per_site(state, env).item()}")
        for step in tqdm(range(1, steps+1)):
            infos = evolution_step_(env, gates, symmetrize=True, opts_svd=opts_svd, initialization='EAT_SVD')
            infoss.append(infos)

            if step % 10 == 0:
                for site in state.sites():
                    state.parameters[site] = state[site]/state[site].norm(p='inf')
                state.sync_()
                out = env.iterate_(max_sweeps=400, iterator_step=None, diff_tol=1e-8)
                diff = BP_conv_check_diff(env_prev, env)
                log.log(logging.INFO, f"{env_type}_db={db} step: {step:d}, weights diff: {diff}")
                state.write_to_file(outputstatefile, normalize=True)
                if diff < weights_tol:
                    log.log(logging.INFO, f"{env_type}_db={db} step: {step:d}, converged")
                    break
                env_prev = env.copy()
                # losses.append(model.energy_per_site(state, env).item())
                # log.log(logging.INFO, f"{env_type}_db={db} step: {step:d}, loss: {losses[-1]}")

        # Evaluate energy with CTM-environment
        chi = cfg.main_args.chi
        env_opts_svd = {
            "D_total": chi,
            "tol": 1e-10,
            "eps_multiplet": 1e-10,
            "truncate_multiplets": True,
        }

        env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(chi,))
        with torch.no_grad():
            ctm_env = EnvCTM(state, init='eye', leg=env_leg)
            for out in ctm_env.ctmrg_(opts_svd=env_opts_svd, max_sweeps=500, svd_policy='fullrank', iterator_step=1, use_qr=False, corner_tol=1e-9):
                print(out)
        ctm_loss = model.energy_per_site(state, ctm_env).item()
        log.log(logging.INFO, f"CTM_ENV, loss: {ctm_loss}")
        state.write_to_file(outputstatefile, normalize=True)
        if len(losses) >= 2 and losses[-1] - losses[-2] > 1e-4:
            break

        Delta = accumulated_truncation_error(infoss)
        log.log(logging.INFO, f"{env_type}_db={db} step: {step:d}, Delta: {Delta}")




