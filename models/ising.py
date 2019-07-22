import torch
import su2
from env import ENV
import ipeps
import rdm
from args import GLOBALARGS
from math import sqrt
import itertools

class ISING():
    def __init__(self, hx=0.0, q=0.0, global_args=GLOBALARGS()):
        r"""
        
        :param hx: transverse field
        :param q: plaquette interaction 
        :param global_args:
        :type hx: float
        :type q: float
        :type global_args: type description

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

          and indices of `h4` correspond to :math:`s_is_js_ks_l;s'_is'_js'_ks'_l`
        
        * :math:`h1_i  = 2S^x_i`
        """
        self.dtype=global_args.dtype
        self.device=global_args.device
        self.phys_dim=2
        self.hx=hx
        self.q=q
        
        self.h2, self.h4, self.h1 = self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device) 
        SzSz = 4*torch.einsum('ij,ab->iajb',s2.SZ(),s2.SZ())
        SzSzSzSz = 4*torch.einsum('ijab,klcd->ijklabcd',SzSz,SzSz)
        Sx = s2.SP()+s2.SM()
        return SzSz, SzSzSzSz, Sx 

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= 2*s2.SZ()
        obs_ops["sp"]= 2*s2.SP()
        obs_ops["sm"]= 2*s2.SM()
        return obs_ops

    def energy_1x1(self,state,env):
        r"""
        For 1-site invariant iPEPS it's enough to construct a single reduced
        density matrix of a 2x2 plaquette. Afterwards, the energy per site `e` is 
        computed by evaluating individual terms in the Hamiltonian through
        :math:`\langle \mathcal{O} \rangle = Tr(\rho_{2x2} \mathcal{O})`
        
        .. math:: 

            e = -(\langle h2_{<\bf{0},\bf{x}>} \rangle + \langle h2_{<\bf{0},\bf{y}>} \rangle)
            + q\langle h4_{\bf{0}} \rangle - h_x \langle h4_{\bf{0}} \rangle

        """
        rdm2x2= rdm.rdm2x2((0,0),state,env)
        eSx= torch.einsum('ijklajkl,ia',rdm2x2,self.h1)
        eSzSz= torch.einsum('ijklabkl,ijab',rdm2x2,self.h2) + \
            torch.einsum('ijklajcl,ikac',rdm2x2,self.h2)
        eSzSzSzSz= torch.einsum('ijklabcd,ijklabcd',rdm2x2,self.h4)
        energy_per_site = -eSzSz - self.hx*eSx + self.q*eSzSzSzSz
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

    # definition of other observables
    # sp=sx+isy, sm=sx-isy => sx=0.5(sp+sm), sy=-i0.5(sp-sm)
    # m=\sqrt(<sz>^2+<sx>^2+<sy>^2)=\sqrt(<sz>^2+0.25(<sp>+<sm>)^2-0.25(<sp>-<sm>)^2)
    #  =\sqrt(<sz>^2+0.5<sp><sm>)
    #
    # expect "list" of (observable label, value) pairs
    # TODO optimize/unify ?
    def eval_obs(self,state,env):
        obs= dict()
        with torch.no_grad():
            for coord,site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord,state,env)
                for label,op in self.obs_ops.items():
                    obs[f"{label}{coord}"]= torch.trace(rdm1x1@op)
                obs[f"sx{coord}"]= 0.5*(obs[f"sp{coord}"] + obs[f"sm{coord}"])
        
        # prepare list with labels and values
        obs_labels= [f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), ["sz","sx"]))]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels