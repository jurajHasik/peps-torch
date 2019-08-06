import torch
import su2
# from env import ENV
import ipeps
from c4v import *
from ctm.generic import rdm
from ctm.one_site_c4v import rdm_c4v
import config as cfg
from math import sqrt
import itertools

class ISING():
    def __init__(self, hx=0.0, q=0.0, global_args=cfg.global_args):
        r"""
        :param hx: transverse field
        :param q: plaquette interaction 
        :param global_args: global configuration
        :type hx: float
        :type q: float
        :type global_args: GLOBALARGS

        Build Ising Hamiltonian in transverse field with plaquette interaction

        .. math:: H = - \sum_{<i,j>} h2_{<i,j>} + q\sum_{p} h4_p - h_x\sum_i h1_i

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), the second sum runs over 
        all plaquettes `p`, and the last sum runs over all sites::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h2_{ij} = 4S^z_i S^z_j` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        * :math:`h4_p  = 16S^z_i S^z_j S^z_k S^z_l` where `i,j,k,l` labels the sites of a plaquette::
          
            p= i---j
               |   |
               k---l 

          and the indices of `h4` correspond to :math:`s_is_js_ks_l;s'_is'_js'_ks'_l`
        
        * :math:`h1_i  = 2S^x_i`
        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.hx=hx
        self.q=q
        
        self.h2, self.h4, self.h1, self.hp = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device) 
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        SzSz = 4*torch.einsum('ij,ab->iajb',s2.SZ(),s2.SZ())
        SzSzIdId= torch.einsum('ijab,klcd->ijklabcd',SzSz,id2)
        SzSzSzSz= torch.einsum('ijab,klcd->ijklabcd',SzSz,SzSz)
        Sx = s2.SP()+s2.SM()
        SxIdIdId= torch.einsum('ia,jb,kc,ld->ijklabcd',Sx,s2.I(),s2.I(),s2.I())

        hp = -SzSzIdId -SzSzIdId.permute(0,2,1,3,4,6,5,7) -self.q*SzSzSzSz -self.hx*SxIdIdId

        return SzSz, SzSzSzSz, Sx, hp

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= 2*s2.SZ()
        obs_ops["sp"]= 2*s2.SP()
        obs_ops["sm"]= 2*s2.SM()
        return obs_ops

    def energy_1x1(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        For 1-site invariant iPEPS it's enough to construct a single reduced
        density matrix of a 2x2 plaquette. Afterwards, the energy per site `e` is 
        computed by evaluating individual terms in the Hamiltonian through
        :math:`\langle \mathcal{O} \rangle = Tr(\rho_{2x2} \mathcal{O})`
        
        .. math:: 

            e = -(\langle h2_{<\bf{0},\bf{x}>} \rangle + \langle h2_{<\bf{0},\bf{y}>} \rangle)
            + q\langle h4_{\bf{0}} \rangle - h_x \langle h4_{\bf{0}} \rangle

        """
        rdm2x2= rdm.rdm2x2((0,0),state,env)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp) 
        # eSx= torch.einsum('ijklajkl,ia',rdm2x2,self.h1)
        # eSzSz= torch.einsum('ijklabkl,ijab',rdm2x2,self.h2) + \
        #     torch.einsum('ijklajcl,ikac',rdm2x2,self.h2)
        # eSzSzSzSz= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.h4)
        # energy_per_site = -eSzSz - self.hx*eSx + self.q*eSzSzSzSz
        return energy_per_site 

    # assuming reduced density matrix of 2x2 cluster with indexing of DOFs
    # as follows rdm2x2=rdm2x2(s0,s1,s2,s3;s0',s1',s2',s3')
    #
    # s0,s1
    # s2,s3
    #                
    #                          A3--1B   B3  1A
    #                          2 \/ 2   2 \/ 2
    #                A B       0 /\ 0   0 /\ 0
    # Ex.1 unit cell B A terms B3--1A & A3  1B
    #
    #                          A3--1B   B3--1A
    #                          2 \/ 2   2 \/ 2
    #                A B       0 /\ 0   0 /\ 0
    # Ex.2 unit cell A B terms A3--1B & B3--1A
    def energy_2x2_2site(self,state,env):
        pass

    def eval_obs(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. :math:`\langle 2S^z \rangle,\ \langle 2S^x \rangle` for each site in the unit cell

        """
        obs= dict()
        with torch.no_grad():
            for coord,site in state.sites.items():
                rdm1x1= rdm.rdm1x1(coord,state,env)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"sx{coord}"]= 0.5*(obs[f"sp{coord}"] + obs[f"sm{coord}"])
            
            for coord,site in state.sites.items():
                rdm2x1= rdm.rdm2x1(coord,state,env)
                rdm1x2= rdm.rdm1x2(coord,state,env)
                rdm2x2= rdm.rdm2x2(coord,state,env)
                obs[f"SzSz2x1{coord}"]= torch.einsum('ijab,ijab',rdm2x1,self.h2)
                obs[f"SzSz1x2{coord}"]= torch.einsum('ijab,ijab',rdm1x2,self.h2)
                obs[f"SzSzSzSz{coord}"]= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.h4)

        # prepare list with labels and values
        obs_labels= [f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), ["sz","sx"]))]
        obs_labels+= [f"SzSz2x1{coord}" for coord in state.sites.keys()]
        obs_labels+= [f"SzSz1x2{coord}" for coord in state.sites.keys()]
        obs_labels+= [f"SzSzSzSz{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

class ISING_C4V():
    def __init__(self, hx=0.0, q=0.0, global_args=cfg.global_args):
        r"""
        :param hx: transverse field
        :param q: plaquette interaction 
        :param global_args: global configuration
        :type hx: float
        :type q: float
        :type global_args: GLOBALARGS

        Build Ising Hamiltonian in transverse field with plaquette interaction

        .. math:: H = - \sum_{<i,j>} h2_{<i,j>} + q\sum_{p} h4_p - h_x\sum_i h1_i

        on the square lattice. Where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), the second sum runs over 
        all plaquettes `p`, and the last sum runs over all sites::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h2_{ij} = 4S^z_i S^z_j` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        * :math:`h4_p  = 16S^z_i S^z_j S^z_k S^z_l` where `i,j,k,l` labels the sites of a plaquette::
          
            p= i---j
               |   |
               k---l 

          and the indices of `h4` correspond to :math:`s_is_js_ks_l;s'_is'_js'_ks'_l`
        
        * :math:`h1_i  = 2S^x_i`
        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.hx=hx
        self.q=q
        
        self.h2, self.h4, self.h1, self.hp = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device) 
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        SzSz = 4*torch.einsum('ij,ab->iajb',s2.SZ(),s2.SZ())
        SzSzIdId= torch.einsum('ijab,klcd->ijklabcd',SzSz,id2)
        SzSzSzSz= torch.einsum('ijab,klcd->ijklabcd',SzSz,SzSz)
        Sx = s2.SP()+s2.SM()
        SxIdIdId= torch.einsum('ia,jb,kc,ld->ijklabcd',Sx,s2.I(),s2.I(),s2.I())

        hp = -SzSzIdId -SzSzIdId.permute(0,2,1,3,4,6,5,7) -self.q*SzSzSzSz -self.hx*SxIdIdId

        return SzSz, SzSzSzSz, Sx, hp

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= 2*s2.SZ()
        obs_ops["sp"]= 2*s2.SP()
        obs_ops["sm"]= 2*s2.SM()
        return obs_ops

    def energy_1x1(self,state,env_c4v):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env_c4v: ENV_C4V
        :return: energy per site
        :rtype: float

        For 1-site invariant c4v iPEPS it's enough to construct a single reduced
        density matrix of a 2x2 plaquette. Afterwards, the energy per site `e` is 
        computed by evaluating individual terms in the Hamiltonian through
        :math:`\langle \mathcal{O} \rangle = Tr(\rho_{2x2} \mathcal{O})`
        
        .. math:: 

            e = -(\langle h2_{<\bf{0},\bf{x}>} \rangle + \langle h2_{<\bf{0},\bf{y}>} \rangle)
            + q\langle h4_{\bf{0}} \rangle - h_x \langle h4_{\bf{0}} \rangle

        """
        rdm2x2= rdm_c4v.rdm2x2(state,env_c4v)
        energy_per_site= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.hp)
        
        # From individual contributions 
        # rdm2x1= rdm_c4v.rdm2x1(state,env_c4v)
        # eSx= torch.einsum('ijaj,ia',rdm2x1,self.h1)
        # eSzSz= torch.einsum('ijab,ijab',rdm2x1,self.h2)
        # eSzSzSzSz= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.h4)
        # energy_per_site = -2*eSzSz - self.hx*eSx + self.q*eSzSzSzSz
        return energy_per_site

    def eval_obs(self,state,env_c4v):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env_c4v: ENV_C4V
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. :math:`\langle 2S^z \rangle,\ \langle 2S^x \rangle` for each site in the unit cell

        TODO 2site observable SzSz
        """
        obs= dict()
        with torch.no_grad():
            # symmetrize on-site tensor
            symm_sites= {(0,0): make_c4v_symm(state.sites[(0,0)])}
            symm_sites[(0,0)]= symm_sites[(0,0)]/torch.max(torch.abs(symm_sites[(0,0)]))
            symm_state= ipeps.IPEPS(symm_sites)

            rdm1x1= rdm_c4v.rdm1x1(symm_state,env_c4v)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs["sx"]= 0.5*(obs["sp"] + obs["sm"])
            
            #rdm2x1= rdm.rdm2x1(coord,state,env)
            #rdm1x2= rdm.rdm1x2(coord,state,env)
            rdm2x2= rdm_c4v.rdm2x2(symm_state,env_c4v)
            #obs[f"SzSz2x1{coord}"]= torch.einsum('ijab,ijab',rdm2x1,self.h2)
            #obs[f"SzSz1x2{coord}"]= torch.einsum('ijab,ijab',rdm1x2,self.h2)
            obs["SzSzSzSz"]= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.h4)

        # prepare list with labels and values
        obs_labels= [lc for lc in ["sz","sx"]]
        #obs_labels+= [f"SzSz2x1{coord}" for coord in state.sites.keys()]
        #obs_labels+= [f"SzSz1x2{coord}" for coord in state.sites.keys()]
        obs_labels+= ["SzSzSzSz"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels