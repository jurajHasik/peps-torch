import numpy as np
import torch
import yast.yast as yast
import groups.su3_abelian
import examples.abelian.settings_U1xU1_torch as settings_U1xU1
import config as cfg
cfg.global_args.dtype= "complex128"
cfg.global_args.torch_dtype= torch.complex128
settings_U1xU1.default_dtype= cfg.global_args.dtype
cfg.main_args.chi= 17

SU3U1U1_3= groups.su3_abelian.SU3_DEFINING_U1xU1(settings_U1xU1)

import models.abelian.su3_kagome as su3_kagome
import models.su3_kagome as su3_kagome_dense

cfg.print_config()
p_phi=0.5
p_theta=0.0
model= su3_kagome.KAGOME_SU3_U1xU1(settings_U1xU1,
 j=np.cos(np.pi*p_phi),\
 k=np.sin(np.pi*p_phi) * np.cos(np.pi*p_theta),\
 h=np.sin(np.pi*p_phi) * np.sin(np.pi*p_theta))
# m_dense= su3_kagome_dense.KAGOME_SU3(j=1.,k=0.,h=0.)

from ipeps.ipess_kagome_abelian import *

# path_to_file="test-input/abelian/IPESS_FM_D1_1x1_abelian-U1xU1_state.json"
path_to_file="test-input/abelian/IPESS_AKLT_D3_1x1_abelian-U1xU1_T3T8_state.json"
state_ferro= read_ipess_kagome_generic(path_to_file, settings_U1xU1)

from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg

@torch.no_grad()
def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
    if not history:
        history=[]
    e_down_1x1= model.energy_down_t_1x1subsystem(state,env)
    e_down, e_up = model.energy_triangles_2x2subsystem(state, env)
    print(e_down_1x1)
    print(e_down, e_up)
    e_curr= (e_down+e_up)/3
    history.append(e_curr)
    
    obs_values, obs_labels= model.eval_obs(state,env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

    obs_values, obs_labels= model.eval_obs_2x2subsystem(state,env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{e_curr}"]+[f"{v}" for v in obs_values]))

    if len(history) >= ctm_args.ctm_max_iter:
#         log.info({"history_length": len(history), "history": history})
        return True, history
    return False, history

ctm_env= ENV_ABELIAN(cfg.main_args.chi, state=state_ferro, init=True)
print(ctm_env)

# 3) compute initial observables
ctm_env, *ctm_log = ctmrg.run(state_ferro, ctm_env, conv_check=ctmrg_conv_energy)

#-2.8078e-17, -2.5390e-17, -1.8210e-17, -1.2053e-17, -5.8351e-18,
         # 0.0000e+00,  0.0000e+00,  2.3768e-20,  3.0197e-20,  1.3879e-17,
         # 2.0122e-17,  6.2500e-02,  6.2500e-02,  6.2500e-02,  6.2500e-02,
         # 6.2500e-02,  6.2500e-02,  6.2500e-02,  6.2500e-02,  6.2500e-02,
         # 6.2500e-02,  6.2500e-02,  6.2500e-02,  6.2500e-02,  6.2500e-02,
         # 6.2500e-02,  6.2500e-02