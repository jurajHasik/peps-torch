import torch
import groups.su2 as su2
from ctm.generic import rdm
from ctm.one_site_c4v import rdm_c4v
from ctm.one_site_c4v import corrf_c4v
import config as cfg
from math import sqrt
import itertools

def _cast_to_real(t):
    return t.real if t.is_complex() else t

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
        self.dtype=global_args.torch_dtype
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
        energy_per_site= _cast_to_real(energy_per_site)

        return energy_per_site 

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
                SzSz2x1= torch.einsum('ijab,ijab',rdm2x1,self.h2)
                SzSz1x2= torch.einsum('ijab,ijab',rdm1x2,self.h2)
                SzSzSzSz= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.h4)
                obs[f"SzSz2x1{coord}"]= _cast_to_real(SzSz2x1)
                obs[f"SzSz1x2{coord}"]= _cast_to_real(SzSz1x2)
                obs[f"SzSzSzSz{coord}"]= _cast_to_real(SzSzSzSz)

        # prepare list with labels and values
        obs_labels= [f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), ["sz","sx"]))]
        obs_labels+= [f"SzSz2x1{coord}" for coord in state.sites.keys()]
        obs_labels+= [f"SzSz1x2{coord}" for coord in state.sites.keys()]
        obs_labels+= [f"SzSzSzSz{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

class ISING_C4V():
    def __init__(self, hx=0.0, q=0, global_args=cfg.global_args):
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
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=2
        self.hx=hx
        self.q=q
        
        self.h2, self.hp, self.szszszsz, self.szsz, self.sx = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device) 
        id2= torch.eye(4,dtype=self.dtype,device=self.device)
        id2= id2.view(2,2,2,2).contiguous()
        SzSz = 4*torch.einsum('ij,ab->iajb',s2.SZ(),s2.SZ())
        SzSzIdId= torch.einsum('ijab,klcd->ijklabcd',SzSz,id2)
        SzSzSzSz= torch.einsum('ijab,klcd->ijklabcd',SzSz,SzSz)
        Sx = s2.SP()+s2.SM()
        SxId= torch.einsum('ij,ab->iajb', Sx, s2.I())
        SxIdIdId= torch.einsum('ia,jb,kc,ld->ijklabcd',Sx,s2.I(),s2.I(),s2.I())

        h2= -SzSz - 0.5*self.hx*SxId
        hp= -SzSzIdId -SzSzIdId.permute(0,2,1,3,4,6,5,7) -self.q*SzSzSzSz -self.hx*SxIdIdId

        return h2, hp, SzSzSzSz, SzSz, Sx

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= 2*s2.SZ()
        obs_ops["sp"]= 2*s2.SP()
        obs_ops["sm"]= 2*s2.SM()
        return obs_ops

    def energy_1x1_nn(self,state,env_c4v):
        r"""
        :param state: wavefunction
        :param env_c4v: CTM c4v symmetric environment
        :type state: IPEPS
        :type env_c4v: ENV_C4V
        :return: energy per site
        :rtype: float

        For 1-site invariant c4v iPEPS with no 4-site term present in Hamiltonian it is enough 
        to construct a single reduced density matrix of a 2x1 nearest-neighbour sites. 
        Afterwards, the energy per site `e` is computed by evaluating individual terms 
        in the Hamiltonian through :math:`\langle \mathcal{O} \rangle = Tr(\rho_{2x1} \mathcal{O})`
        
        .. math:: 

            e = -\langle h2_{<\bf{0},\bf{x}>} \rangle - h_x \langle h1_{\bf{0}} \rangle
        """
        assert self.q==0, "Non-zero value of 4-site term coupling"

        rdm2x1= rdm_c4v.rdm2x1_sl(state, env_c4v)
        eSx= torch.einsum('ijaj,ia',rdm2x1,self.sx)
        eSzSz= torch.einsum('ijab,ijab',rdm2x1,self.szsz)
        energy_per_site = -2*eSzSz - self.hx*eSx
        return energy_per_site

    def energy_1x1_plaqette(self,state,env_c4v):
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
        # eSx= torch.einsum('ijklajkl,ia',rdm2x2,self.sx)
        # eSzSz= torch.einsum('ijklabkl,ijab',rdm2x1,self.szsz)
        # eSzSzSzSz= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.szszszsz)
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
            rdm1x1= rdm_c4v.rdm1x1(state,env_c4v)
            for label,op in self.obs_ops.items():
                obs[f"{label}"]= torch.trace(rdm1x1@op)
            obs["sx"]= 0.5*(obs["sp"] + obs["sm"])
            
            rdm2x2= rdm_c4v.rdm2x2(state,env_c4v)
            obs["SzSzSzSz"]= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.szszszsz)

        # prepare list with labels and values
        obs_labels= [lc for lc in ["sz","sx"]]
        obs_labels+= ["SzSzSzSz"]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_corrf_SS(self,state,env_c4v,dist):
        Sop_zxy= torch.zeros((3,self.phys_dim,self.phys_dim),dtype=self.dtype,device=self.device)
        Sop_zxy[0,:,:]= self.obs_ops["sz"]
        Sop_zxy[1,:,:]= 0.5*(self.obs_ops["sp"] + self.obs_ops["sm"])    # S^x

        # dummy function, since no sublattice rotation is present
        def get_op(op):
            op_0= op
            def _gen_op(r): return op_0
            return _gen_op

        Sz0szR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[0,:,:], get_op(Sop_zxy[0,:,:]), dist)
        Sx0sxR= corrf_c4v.corrf_1sO1sO(state, env_c4v, Sop_zxy[1,:,:], get_op(Sop_zxy[1,:,:]), dist)
 
        res= dict({"ss": Sz0szR+Sx0sxR, "szsz": Sz0szR, "sxsx": Sx0sxR})
        return res