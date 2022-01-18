from math import sqrt
import itertools
import numpy as np
import torch
import yamps.yast as yast
import config as cfg
import groups.su2_abelian as su2
import ctm.pess_kagome_abelian.rdm_kagome as rdm_kagome
from yamps.yast.tensor._output import to_number, to_dense, to_numpy, to_raw_tensor, to_nonsymmetric
from ctm.generic import transferops

def _cast_to_real(t, check=True, imag_eps=1.0e-8):
    if t.is_complex():
        #print(abs(t.imag)/abs(t.real))
        #print(t)
        assert (abs(t.imag)/abs(t.real) < imag_eps) or (abs(t.imag)< imag_eps),"unexpected imaginary part "+str(t.imag)
        return t.real
    return t

class KAGOME_U1():
    def __init__(self, settings, j1=1., JD=0, j1sq=0., j2=0., j2sq=0., jtrip=0., jperm=0., h=0, global_args=cfg.global_args):
        r"""
        H = J_1 \sum_{<ij>} S_i.S_j + J_{1sq} \sum_{<ij>} (S_i.S_j)^2
            + J_2 \sum_{<<ij>>} S_i.S_j + J_{2sq} \sum_{<<ij>>} (S_i.S_j)^2
            - J_{trip} \sum_t (S_{t_1} \times S_{t_2}).S_{t_3}
            + J_{perm} \sum_t P_t + J*_{perm} \sum_t P^{-1}_t
        """


        self.engine= settings
        self.dtype=settings.default_dtype
        self.device='cpu' if not hasattr(settings, 'device') else settings.device
        self.phys_dim = 2
        self.j1 = j1
        self.JD = JD
        self.j1sq = j1sq
        self.j2 = j2
        self.j2sq = j2sq
        self.jtrip = jtrip
        self.jperm = jperm

        irrep = su2.SU2_U1(self.engine, 2)

        #import pdb; pdb.set_trace()

        Id1= irrep.I()
        #self.Id3_t= torch.eye(self.phys_dim**3, dtype=self.dtype, device=self.device)

        id2= yast.tensordot(irrep.I(),irrep.I(),([],[])).transpose(axes=(0,2,1,3))
        self.Id2=id2
        self.Id3_t= yast.tensordot(id2,irrep.I(),([],[])).transpose(axes=(0,1,4,2,3,5)).fuse_legs(axes=((0,1,2), (3,4,5)))

        

        if abs(JD)==0:
            SS= irrep.SS(zpm=(j1, j1, j1))
        else:
            SS= irrep.SS(zpm=(j1, j1+1j*JD, j1-1j*JD))
            
        self.SSnnId= yast.tensordot(SS,Id1,([],[])).transpose(axes=(0,1,4,2,3,5)) 
        SSnn_t= self.SSnnId + self.SSnnId.transpose(axes=(1,2,0, 4,5,3)) + self.SSnnId.transpose(axes=(2,0,1, 5,3,4))

        mag_field=h*(irrep.SZ())
        mag_field= yast.tensordot(mag_field,irrep.I(),([],[])).transpose(axes=(0,2,1,3))
        mag_field= yast.tensordot(mag_field,irrep.I(),([],[])).transpose(axes=(0,1,4,2,3,5))
        mag_field=mag_field + mag_field.transpose(axes=(1,2,0, 4,5,3)) + mag_field.transpose(axes=(2,0,1, 5,3,4))

        smsp= yast.tensordot(irrep.SM(),irrep.SP(),([],[])).transpose(axes=(0,2,1,3))
        spsm= yast.tensordot(irrep.SP(),irrep.SM(),([],[])).transpose(axes=(0,2,1,3))
        SxSS_t= yast.tensordot(smsp-spsm,irrep.SZ()/(2j),([],[])).transpose(axes=(0,1,4,2,3,5))
        SxSS_t= SxSS_t+ SxSS_t.transpose(axes=(1,2,0, 4,5,3)) + SxSS_t.transpose(axes=(2,0,1, 5,3,4))
        SxSS_t= jtrip*SxSS_t

        self.h_triangle= SSnn_t +mag_field + SxSS_t

        szId2= yast.tensordot(irrep.SZ(),id2,([],[])).transpose(axes=(0,2,3,1,4,5))
        spId2= yast.tensordot(irrep.SP(),id2,([],[])).transpose(axes=(0,2,3,1,4,5))
        smId2= yast.tensordot(irrep.SM(),id2,([],[])).transpose(axes=(0,2,3,1,4,5))
        self.obs_ops= {
            "sz_0": szId2, "sp_0": spId2, "sm_0": smId2,\
            "sz_1": szId2.transpose(axes=(1,2,0, 4,5,3)),\
            "sp_1": spId2.transpose(axes=(1,2,0, 4,5,3)),\
            "sm_1": smId2.transpose(axes=(1,2,0, 4,5,3)),\
            "sz_2": szId2.transpose(axes=(2,0,1, 5,3,4)),\
            "sp_2": spId2.transpose(axes=(2,0,1, 5,3,4)),\
            "sm_2": smId2.transpose(axes=(2,0,1, 5,3,4)),
            }
        self.SS01= self.SSnnId
        self.SS12= self.SSnnId.transpose(axes=(1,2,0, 4,5,3))
        self.SS02= self.SSnnId.transpose(axes=(2,0,1, 5,3,4))



    def energy_triangle_dn(self, state, env, force_cpu=False):
        #print(self.h_triangle)
        e_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, self.h_triangle.fuse_legs(axes=((0,1,2),(3,4,5))), force_cpu=force_cpu).to_number()
        #print(rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, self.Id3_t, force_cpu=force_cpu).to_number())
        return _cast_to_real(e_dn)

    def energy_triangle_up(self, state, env, force_cpu=False):
        rdm_up= rdm_kagome.rdm2x2_up_triangle_open((0, 0), state, env, force_cpu=force_cpu)
        e_up=yast.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),self.h_triangle.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0])).to_number()
        # rdm=rdm_kagome.rdm2x2_kagome(coord, state, ctm_env_init, sites_to_keep_00=(), sites_to_keep_10=('B'), sites_to_keep_01=('A'), sites_to_keep_11=('C'))
        # print(rdm.fuse_legs(axes=((0,1,2),(3,4,5))))
        # print(yast.tensordot(rdm.fuse_legs(axes=((0,1,2),(3,4,5))),H,([0,1],[1,0])).to_number())
        return _cast_to_real(e_up)

    def energy_triangle_dn_NoCheck(self, state, env, force_cpu=False):
        #print(self.h_triangle)
        e_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, self.h_triangle.fuse_legs(axes=((0,1,2),(3,4,5))), force_cpu=force_cpu).to_number()
        return e_dn

    def energy_triangle_up_NoCheck(self, state, env, force_cpu=False):
        rdm_up= rdm_kagome.rdm2x2_up_triangle_open((0, 0), state, env, force_cpu=force_cpu)
        e_up=yast.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),self.h_triangle.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0])).to_number()
        return e_up
        


    #define operators for correlation functions
    def sz_0_op(self):
        op=self.obs_ops["sz_0"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sz_1_op(self):
        op=self.obs_ops["sz_1"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sz_2_op(self):
        op=self.obs_ops["sz_2"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sp_0_op(self):
        op=self.obs_ops["sp_0"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sp_1_op(self):
        op=self.obs_ops["sp_1"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sp_2_op(self):
        op=self.obs_ops["sp_2"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sm_0_op(self):
        op=self.obs_ops["sm_0"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sm_1_op(self):
        op=self.obs_ops["sm_1"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sm_2_op(self):
        op=self.obs_ops["sm_2"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op

    def SS01_op(self):
        op=self.SS01.fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def SS12_op(self):
        op=self.SS12.fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def SS02_op(self):
        op=self.SS02.fuse_legs(axes=((0,1,2),(3,4,5)))
        return op

   



    # Observables
    def eval_obs(self,state,env,force_cpu=True,cast_real=True, disp_corre_len=False):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]
        """
        obs= {"e_t_dn": 0, "e_t_up": 0, "m_0": 0, "m_1": 0, "m_2": 0}
        with torch.no_grad():
            e_t_dn= self.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
            e_t_up= self.energy_triangle_up_NoCheck(state, env, force_cpu=force_cpu)
            obs["e_t_dn"]= e_t_dn
            obs["e_t_up"]= e_t_up

            for label in self.obs_ops.keys():
                op= self.obs_ops[label]
                obs_val=rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, op, force_cpu=force_cpu).to_number()
                obs[f"{label}"]= obs_val #_cast_to_real(obs_val)

            for i in range(3):
                #obs[f"m_{i}"]= sqrt(_cast_to_real(obs[f"sz_{i}"]*obs[f"sz_{i}"]+ obs[f"sp_{i}"]*obs[f"sm_{i}"]))
                obs[f"m_{i}"]= ((obs[f"sz_{i}"]*obs[f"sz_{i}"]+ obs[f"sp_{i}"]*obs[f"sm_{i}"]))
 
            # nn S.S pattern
            
            SS_dn_01= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, self.SS01, force_cpu=force_cpu).to_number()
            SS_dn_12= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, self.SS12, force_cpu=force_cpu).to_number()
            SS_dn_02= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, self.SS02, force_cpu=force_cpu).to_number()
            
            rdm_up= rdm_kagome.rdm2x2_up_triangle_open((0, 0), state, env, force_cpu=force_cpu)
            #bb=yast.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),self.Id3_t,([0,1],[1,0])).to_number()
            #print(bb)
            SS_up_01= yast.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),self.SS01.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0])).to_number()
            SS_up_12= yast.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),self.SS12.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0])).to_number()
            SS_up_02= yast.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),self.SS02.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0])).to_number()

            obs.update({"SS_dn_01": SS_dn_01, "SS_dn_12": SS_dn_12, "SS_dn_02": SS_dn_02,\
                "SS_up_01": SS_up_01, "SS_up_12": SS_up_12, "SS_up_02": SS_up_02 })
            if disp_corre_len:

                #convert to dense env and compute transfer operator spectrum
                env= env.to_dense(state)
                state= state.to_dense()
                
                
                #Correlation length
                Ns=3
                coord=(0,0)
                direction=(1,0)
                Lx= transferops.get_Top_spec(Ns, coord, direction, state, env)
                direction=(0,1)
                Ly= transferops.get_Top_spec(Ns, coord, direction, state, env)
                lambdax_0=torch.abs(Lx[0,0]+1j*Lx[0,1])
                lambdax_1=torch.abs(Lx[1,0]+1j*Lx[1,1])
                lambdax_2=torch.abs(Lx[2,0]+1j*Lx[2,1])
                lambday_0=torch.abs(Ly[0,0]+1j*Ly[0,1])
                lambday_1=torch.abs(Ly[1,0]+1j*Ly[1,1])
                lambday_2=torch.abs(Ly[2,0]+1j*Ly[2,1])
                correlen_x=-1/(torch.log(lambdax_1/lambdax_0))
                correlen_y=-1/(torch.log(lambday_1/lambday_0))
                obs.update({"lambdax_0": lambdax_0, "lambdax_1": lambdax_1, "lambdax_2": lambdax_2,\
                    "lambday_0": lambday_0, "lambday_1": lambday_1, "lambday_2": lambday_2, "correlen_x": correlen_x, "correlen_y": correlen_y})
                

        # prepare list with labels and values
        return list(obs.values()), list(obs.keys())
