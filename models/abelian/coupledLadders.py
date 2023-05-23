from math import sqrt
import itertools
import config as cfg
import yastn.yastn as yastn
from tn_interface_abelian import contract, permute  
import groups.su2_abelian as su2
from ctm.generic_abelian import rdm

def _null_Bz(coord):
    return 0.0

class StaggeredLocalField():
    # staggered field 
    # Given the coordinates (x, y), a minus sign is used when x+y is odd
    def __init__(self, B):
        self.B = float(B)

    def __call__(self, coord):
        x, y = coord
        return self.B * (-1) ** ((x+y)%2)

class COUPLEDLADDERS_NOSYM():
    def __init__(self, settings, alpha=0.0, Bz_val=0.0, global_args=cfg.global_args):
        r"""
        :param alpha: nearest-neighbour interaction
        :param Bz_val: staggered magnetic field
        :param global_args: global configuration
        :type alpha: float
        :type Bz: float
        :type global_args: GLOBALARGS

        Build Hamiltonian of spin-1/2 coupled ladders

        .. math:: H = \sum_{i=(x,y)} h2_{i,i+\vec{x}} + \sum_{i=(x,2y)} h2_{i,i+\vec{y}}
                   + \alpha \sum_{i=(x,2y+1)} h2_{i,i+\vec{y}} + (-1)^{x+y} Bz h1_{i}

        on the square lattice. The spin-1/2 ladders are coupled with strength :math:`\alpha`::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._a__a__a__a_...
            ..._|__|__|__|_...
            ..._a__a__a__a_...   
            ..._|__|__|__|_...   
                :  :  :  :      (a = \alpha) 

        where

        * :math:`h2_{ij} = \mathbf{S}_i.\mathbf{S}_j` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`

        * :math:`h1_{i} = \mathbf{S}^z_i` with indices of h1 corresponding to :math:`s_i ;s'_i`
        """
        assert settings.sym.NSYM==0, "No abelian symmetry is assumed"
        self.engine= settings
        self.dtype=settings.default_dtype
        self.device='cpu' if not hasattr(settings, 'device') else settings.device
        self.phys_dim=2
        self.alpha=alpha
        self.Bz_val=Bz_val
        self.Bz=StaggeredLocalField(self.Bz_val)

        self.h1 = self.get_h1()
        self.h2 = self.get_h2()
        self.obs_ops = self.get_obs_ops()

    def get_h1(self):
        irrep = su2.SU2_NOSYM(self.engine, self.phys_dim)
        I1, SP, SM, Sz = irrep.I(), irrep.SP(), irrep.SM(), irrep.SZ()
        SzId= contract(Sz,I1,([],[]))
        SzId= permute(SzId, (0,2,1,3))
        return SzId

    def get_h2(self):
        irrep = su2.SU2_NOSYM(self.engine, self.phys_dim)
        SS = irrep.SS()
        return SS

    def get_obs_ops(self):
        obs_ops = dict()
        irrep = su2.SU2_NOSYM(self.engine, self.phys_dim)
        obs_ops["sz"]= irrep.SZ()
        obs_ops["sp"]= irrep.SP()
        obs_ops["sm"]= irrep.SM()
        return obs_ops

    def energy_2x1_1x2(self,state,env,**kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
        :return: energy per site
        :rtype: float
        
        We assume iPEPS with 2x2 unit cell containing four tensors A, B, C, and D with
        simple PBC tiling::

            A B A B
            C D C D
            A B A B
            C D C D

        Taking the reduced density matrix :math:`\rho_{2x1}` (:math:`\rho_{1x2}`) 
        of 2x1 (1x2) cluster given by :py:func:`rdm.rdm2x1` (:py:func:`rdm.rdm1x2`) 
        with indexing of sites as follows :math:`s_0,s_1;s'_0,s'_1` for both types
        of density matrices::

            rdm2x1   rdm1x2

            s0--s1   s0
                     |
                     s1

        The primed indices represent "bra": :math:`\rho_{2x1} = \sum_{s_0 s_1;s'_0 s'_1}
        | s_0 s_1 \rangle \langle s'_0 s'_1|` where the signature of primed indices is +1.
        Without assuming any symmetry on the indices of individual tensors a following
        set of terms has to be evaluated in order to compute energy-per-site::

               0       0       0
            1--A--3 1--B--3 1--A--3
               2       2       2
               0       0       0
            1--C--3 1--D--3 1--C--3
               2       2       2             A--3 1--B,      A  B  C  D
               0       0                     B--3 1--A,      2  2  2  2
            1--A--3 1--B--3                  C--3 1--D,      0  0  0  0
               2       2             , terms D--3 1--C, and  C, D, A, B
        """
        energy=yastn.zeros(self.engine)
        #
        # (-1)0--|rho|--2(+1) (-1)0--|S.S|--2(+1)
        # (-1)1--|   |--3(+1) (-1)1--|   |--3(+1)
        _ci= ([0,1,2,3],[2,3,0,1])
        _tmp_t= yastn.ones(config=state.engine, s=(-1, -1, 1, 1),
                  t=((-1, 1), (-1, 1), (-1, 1), (-1, 1)),
                  D=((1, 1), (1, 1), (1, 1), (1, 1)))
        _lss_dense={i: l for i,l in enumerate(_tmp_t.get_legs())}
        for coord,site in state.sites.items():
            rdm2x1= rdm.rdm2x1(coord,state,env).to_nonsymmetric(legs=_lss_dense,reverse=True)
            rdm1x2= rdm.rdm1x2(coord,state,env).to_nonsymmetric(legs=_lss_dense,reverse=True)
            ss= contract(rdm2x1, self.h2, _ci)
            energy += ss
            if coord[1] % 2 == 0:
                ss = contract(rdm1x2,self.h2,_ci)
            else:
                ss = contract(rdm1x2,self.alpha * self.h2,_ci)
            energy += ss

            # local field enegy
            sz = contract(rdm1x2, self.Bz(coord) * self.h1, _ci)
            energy += sz

        # return energy-per-site
        energy_per_site=energy/len(state.sites.items())
        energy_per_site=rdm._cast_to_real(energy_per_site,**kwargs)

        return energy_per_site

    def eval_obs(self,state,env,**kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. average magnetization over the unit cell,
            2. magnetization for each site in the unit cell
            3. :math:`\langle S^z \rangle,\ \langle S^+ \rangle,\ \langle S^- \rangle` 
               for each site in the unit cell
            4. :math:`\mathbf{S}_i.\mathbf{S}_j` for all non-equivalent nearest neighbour
               bonds

        where the on-site magnetization is defined as
        
        .. math::
            m = \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }
        """
        obs= dict({"avg_m": 0.})
        _ci= ([0,1],[1,0])
        _tmp_t= yastn.ones(config=state.engine, s=(-1, 1),
                  t=((-1, 1), (-1, 1)),
                  D=((1, 1), (1, 1)))
        _lss_dense={i: l for i,l in enumerate(_tmp_t.get_legs())}
        for coord,site in state.sites.items():
            rdm1x1 = rdm.rdm1x1(coord,state,env).to_nonsymmetric(legs=_lss_dense,reverse=True)
            for label,op in self.obs_ops.items():
                obs[f"{label}{coord}"]= contract(rdm1x1, op, _ci).to_number()
            obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
            obs["avg_m"] += obs[f"m{coord}"]
        obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())
    
        _ci= ([0,1,2,3],[2,3,0,1])
        _tmp_t= yastn.ones(config=state.engine, s=(-1, -1, 1, 1),
                  t=((-1, 1), (-1, 1), (-1, 1), (-1, 1)),
                  D=((1, 1), (1, 1), (1, 1), (1, 1)))
        _lss_dense={i: l for i,l in enumerate(_tmp_t.get_legs())}
        for coord,site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord,state,env).to_nonsymmetric(legs=_lss_dense,reverse=True)
            rdm1x2 = rdm.rdm1x2(coord,state,env).to_nonsymmetric(legs=_lss_dense,reverse=True)
            SS2x1= contract(rdm2x1,self.h2,_ci)
            SS1x2= contract(rdm1x2,self.h2,_ci)
            obs[f"SS2x1{coord}"]= rdm._cast_to_real(SS2x1,**kwargs).to_number()
            obs[f"SS1x2{coord}"]= rdm._cast_to_real(SS1x2,**kwargs).to_number()

        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

class COUPLEDLADDERS_U1():
    def __init__(self, settings, alpha=0.0, Bz_val=0.0, global_args=cfg.global_args):
        r"""
        :param settings: YAST configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
        :param alpha: nearest-neighbour interaction
        :param Bz_val: transverse field
        :type Bz_val: float
        :param global_args: global configuration
        :type alpha: float
        :type global_args: GLOBALARGS

        Build Hamiltonian of spin-1/2 coupled ladders

        .. math:: H = \sum_{i=(x,y)} h2_{i,i+\vec{x}} + \sum_{i=(x,2y)} h2_{i,i+\vec{y}}
                   + \alpha \sum_{i=(x,2y+1)} h2_{i,i+\vec{y}} + (-1)^{x+y} B_z h1_{i}

        on a square lattice. The spin-1/2 ladders are coupled with strength :math:`\alpha`::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._a__a__a__a_...
            ..._|__|__|__|_...
            ..._a__a__a__a_...   
            ..._|__|__|__|_...   
                :  :  :  :      (a = \alpha) 

        where

            * :math:`h2_{ij} = \mathbf{S}_i.\mathbf{S}_j` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
            * :math:`h1_{i} = \mathbf{S}^z_i` with indices of h1 corresponding to :math:`s_i ;s'_i`
        """
        assert settings.sym.NSYM==1, "U(1) abelian symmetry is assumed"
        self.engine= settings
        self.dtype=settings.default_dtype
        self.device='cpu' if not hasattr(settings, 'device') else settings.device
        self.phys_dim=2
        self.alpha=alpha
        self.Bz_val=Bz_val
        self.Bz=StaggeredLocalField(self.Bz_val)

        self.h1 = self.get_h1()
        self.h2 = self.get_h2()
        self.obs_ops = self.get_obs_ops()

    def get_h1(self):
        irrep = su2.SU2_U1(self.engine, self.phys_dim)
        I1, SP, SM, Sz = irrep.I(), irrep.SP(), irrep.SM(), irrep.SZ()
        SzId= contract(Sz,I1,([],[]))
        SzId= permute(SzId, (0,2,1,3))
        return SzId

    def get_h2(self):
        irrep = su2.SU2_U1(self.engine, self.phys_dim)
        SS = irrep.SS()
        return SS

    def get_obs_ops(self):
        obs_ops = dict()
        irrep = su2.SU2_U1(self.engine, self.phys_dim)
        obs_ops["sz"]= irrep.SZ()
        obs_ops["sp"]= irrep.SP()
        obs_ops["sm"]= irrep.SM()
        return obs_ops

    def energy_2x1_1x2(self,state,env,**kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
        :return: energy per site
        :rtype: float
        
        We assume iPEPS with 2x2 unit cell containing four tensors A, B, C, and D with
        simple PBC tiling::

            A B A B
            C D C D
            A B A B
            C D C D

        Taking the reduced density matrix :math:`\rho_{2x1}` (:math:`\rho_{1x2}`) 
        of 2x1 (1x2) cluster given by :py:func:`rdm.rdm2x1` (:py:func:`rdm.rdm1x2`) 
        with indexing of sites as follows :math:`s_0,s_1;s'_0,s'_1` for both types
        of density matrices::

            rdm2x1   rdm1x2

            s0--s1   s0
                     |
                     s1

        The :math:`\rho_{2x1} = \sum_{s_0 s_1;s'_0 s'_1}
        | s_0 s_1 \rangle \langle s'_0 s'_1|` where the signature of primed indices (:math:`|bra\rangle`)
        is +1. Without assuming any symmetry on the indices of individual tensors a following
        set of terms has to be evaluated in order to compute energy-per-site::

               0       0       0
            1--A--3 1--B--3 1--A--3
               2       2       2
               0       0       0
            1--C--3 1--D--3 1--C--3
               2       2       2             A--3 1--B,      A  B  C  D
               0       0                     B--3 1--A,      2  2  2  2
            1--A--3 1--B--3                  C--3 1--D,      0  0  0  0
               2       2             , terms D--3 1--C, and  C, D, A, B
        """
        energy=yastn.zeros(self.engine)
        #
        # (-1)0--|rho|--2(+1) (-1)0--|S.S|--2(+1)
        # (-1)1--|   |--3(+1) (-1)1--|   |--3(+1)
        _ci= ([0,1,2,3],[2,3,0,1])
        for coord,site in state.sites.items():
            rdm2x1= rdm.rdm2x1(coord,state,env)
            rdm1x2= rdm.rdm1x2(coord,state,env)
            ss= contract(rdm2x1, self.h2,_ci)
            energy += ss
            if coord[1] % 2 == 0:
                ss = contract(rdm1x2,self.h2,_ci)
            else:   
                ss = contract(rdm1x2,self.alpha * self.h2,_ci)
            energy += ss

            # local field enegy
            sz = contract(rdm1x2, self.Bz(coord) * self.h1, _ci)
            energy += sz

        # return energy-per-site
        energy_per_site=energy/len(state.sites.items())
        energy_per_site=rdm._cast_to_real(energy_per_site,**kwargs)

        return energy_per_site

    def energy_2x1_1x2_H(self,state,env,**kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN

        Analogous to :meth:`energy_2x1_1x2`, with ladders being weakly coupled 
        in horizontal direction::

            y\x
                   _:_a_:__:_a_:__:
                ..._|_a_|__|_a_|__|...
                ..._|_a_|__|_a_|__|...
                ..._|_a_|__|_a_|__|...
                ..._|_a_|__|_a_|__|...   
                ..._|_a_|__|_a_|__|...   
                    :   :  :   :  : (a = \alpha)
        """
        energy=yastn.zeros(self.engine)
        #
        # (-1)0--|rho|--2(+1) (-1)0--|S.S|--2(+1)
        # (-1)1--|   |--3(+1) (-1)1--|   |--3(+1)
        _ci= ([0,1,2,3],[2,3,0,1])
        for coord,site in state.sites.items():
            rdm2x1= rdm.rdm2x1(coord,state,env)
            rdm1x2= rdm.rdm1x2(coord,state,env)
            ss= contract(rdm1x2, self.h2,_ci)
            energy += ss
            if coord[0] % 2 == 0:
                ss = contract(rdm2x1,self.h2,_ci)
            else:   
                ss = contract(rdm2x1,self.alpha * self.h2,_ci)
            energy += ss

            # local field energy
            sz = contract(rdm1x2, self.Bz(coord) * self.h1, _ci)
            energy += sz

        # return energy-per-site
        energy_per_site=energy/len(state.sites.items())
        energy_per_site=rdm._cast_to_real(energy_per_site,**kwargs)

        return energy_per_site

    def eval_obs(self,state,env,**kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_ABELIAN
        :type env: ENV_ABELIAN
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. average magnetization over the unit cell,
            2. magnetization for each site in the unit cell
            3. :math:`\langle S^z \rangle,\ \langle S^+ \rangle,\ \langle S^- \rangle` 
               for each site in the unit cell
            4. :math:`\mathbf{S}_i.\mathbf{S}_j` for all non-equivalent nearest neighbour
               bonds

        where the on-site magnetization is defined as
        
        .. math::
            m = \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }
        """
        obs= dict({"avg_m": 0.})
        _ci= ([0,1],[1,0])
        for coord,site in state.sites.items():
            rdm1x1 = rdm.rdm1x1(coord,state,env)
            for label,op in self.obs_ops.items():
                obs[f"{label}{coord}"]= contract(rdm1x1, op, _ci).to_number()
            obs[f"m{coord}"]= sqrt(abs(obs[f"sz{coord}"]**2 + obs[f"sp{coord}"]*obs[f"sm{coord}"]))
            obs["avg_m"] += obs[f"m{coord}"]
        obs["avg_m"]= obs["avg_m"]/len(state.sites.keys())
    
        _ci= ([0,1,2,3],[2,3,0,1])
        for coord,site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord,state,env)
            rdm1x2 = rdm.rdm1x2(coord,state,env)
            SS2x1= contract(rdm2x1,self.h2,_ci).to_number()
            SS1x2= contract(rdm1x2,self.h2,_ci).to_number()
            obs[f"SS2x1{coord}"]=rdm._cast_to_real(SS2x1,**kwargs)
            obs[f"SS1x2{coord}"]=rdm._cast_to_real(SS1x2,**kwargs)

        # prepare list with labels and values
        obs_labels=["avg_m"]+[f"m{coord}" for coord in state.sites.keys()]\
            +[f"{lc[1]}{lc[0]}" for lc in list(itertools.product(state.sites.keys(), self.obs_ops.keys()))]
        obs_labels += [f"SS2x1{coord}" for coord in state.sites.keys()]
        obs_labels += [f"SS1x2{coord}" for coord in state.sites.keys()]
        obs_values=[obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def _gen_gate_Sz(self,t):
        gate_Sz= self.h1
        D, U= yastn.linalg.eigh(gate_Sz, axes=([0,1],[2,3]))
        D= D.exp(t)
        gate_Sz = U.tensordot(D, ([2],[0]))
        gate_Sz = gate_Sz.tensordot(U, ([2,2]), conj=(0,1))
        return gate_Sz

    def _gen_gate_SS(self,t):
        gate_SS= self.h2
        D, U= yastn.linalg.eigh(gate_SS, axes=([0,1],[2,3]))
        D= D.exp(t)
        gate_SS= U.tensordot(D, ([2],[0]))
        gate_SS= gate_SS.tensordot(U, ([2,2]), conj=(0,1))
        return gate_SS

    def _gen_gate_SS_hz(self, t, alpha, hz_stag):
        gate_SS_Sz= alpha*self.h2 + hz_stag*(self.h1 - self.h1.transpose((1,0,3,2)) )
        D, U= yastn.linalg.eigh(gate_SS_Sz, axes=([0,1],[2,3]))
        D= D.exp(t)
        gate_SS= U.tensordot(D, ([2],[0]))
        gate_SS= gate_SS.tensordot(U, ([2,2]), conj=(0,1))
        return gate_SS

    def gen_gate_seq_2S(self,t):
        r"""
        :param t: imaginary time step
        :type t: float
        :return: gate sequence
        :rtype: list[tuple(tuple(tuple(int,int),tuple(int,int),tuple(int,int)), yastn.Tensor)]
        
        Generate a 2-site gate sequence :math:`exp(-t \vec{S}.\vec{S})` for imaginary-time optimization.
        Each element of sequence has two parts: First, the placement of the gate encoded by (x,y) 
        coords of the two sites and the vector from 1st to 2nd site: (x_1,y_1), 
        (x_2-x_1, y_2-y_1), (x_2,y_2). Second, the 2-site gate Tensor.

        The gate sequance generated::

                    g[0]         g[2]
            g[4]--(0,0)--g[5]--(1,0)--[g[4]]
                    g[1]         g[3]
            g[6]--(0,1)--g[7]--(1,1)--[g[6]]
                   [g[0]]       [g[2]]

        The g[0] and g[2] are the "weak" links, with :math:`\alpha \vec{S}.\vec{S}` interaction, 
        coupling the ladders. If ``self.Bz`` is non-zero, on-site gates with transverse field 
        are added to the sequence.
        """
        gate_SS_1= self._gen_gate_SS(-t)
        gate_SS_alpha= self._gen_gate_SS(-t*self.alpha)

        # two spin gates 
        gate_seq=[
            (((0,0),(1,0),(1,0)), gate_SS_1),
            (((1,0),(1,0),(0,0)), gate_SS_1),
            (((0,1),(1,0),(1,1)), gate_SS_1),
            (((1,1),(1,0),(0,1)), gate_SS_1),
            (((0,0),(0,1),(0,1)), gate_SS_1),
            (((1,0),(0,1),(1,1)), gate_SS_1),
            (((0,1),(0,1),(0,0)), gate_SS_alpha),
            (((1,1),(0,1),(1,0)), gate_SS_alpha) 
        ]

        # single spin gates 
        # Note: it would be better to join the single spin and the two spin gates
        if self.Bz != _null_Bz:
            for x_1 in range(2):
                for y_1 in range(2):
                    dx, dy = (1, 0)
                    x_2 = (x_1 + dx) % 2
                    y_2 = (y_1 + dy) % 2
                    gate_Sz_Bz= self._gen_gate_Sz(-t*self.Bz((x_1, y_2)))
                    gate_seq+=[(((x_1,y_1),(dx,dy),(x_2,y_2)), gate_Sz_Bz)]
            
        return gate_seq

    def gen_gate_seq_2S_2ndOrder(self,t):
        r"""
        :param t: imaginary time step
        :type t: float
        :return: gate sequence
        :rtype: list[tuple(tuple(tuple(int,int),tuple(int,int),tuple(int,int)), yastn.Tensor)]
        
        Second-order Trotter gate sequence. This sequence can be generated from the result of 
        :meth:`gen_gate_seq_2S` by applying the gates in both direct and reverse order. 
        """
        gate_SS_1= self._gen_gate_SS(-t)
        gate_SS_2= self._gen_gate_SS(-2*t)
        gate_SS_alpha= self._gen_gate_SS(-t*self.alpha)
        gate_SS_2alpha= self._gen_gate_SS(-2*t*self.alpha)

        # single spin gates 
        # Note: it would be better to join the single spin gates and the two spin gates
        gate_seq= []
        if self.Bz != _null_Bz:
            for x_1 in range(2):
                for y_1 in range(2):
                    dx, dy = (1, 0)
                    x_2 = (x_1 + dx) % 2
                    y_2 = (y_1 + dy) % 2
                    gate_Sz_Bz= self._gen_gate_Sz(-t*self.Bz((x_1, y_2)))
                    gate_seq+=[(((x_1,y_1),(dx,dy),(x_2,y_2)), gate_Sz_Bz)]

        # two spin gates 
        gate_seq+=[
            (((0,0),(1,0),(1,0)), gate_SS_1),
            (((1,0),(1,0),(0,0)), gate_SS_1),
            (((0,1),(1,0),(1,1)), gate_SS_1),
            (((1,1),(1,0),(0,1)), gate_SS_1),
            (((0,0),(0,1),(0,1)), gate_SS_1),
            (((1,0),(0,1),(1,1)), gate_SS_1),
            (((0,1),(0,1),(0,0)), gate_SS_alpha),
            (((1,1),(0,1),(1,0)), gate_SS_2alpha)
        ]

        # repeat the sequence in inverse order
        for i in range( len(gate_seq)-2 ,-1,-1): gate_seq.append(gate_seq[i])
        return gate_seq

    def gen_gate_seq_2S_SS_hz(self,t):
        r"""
        :param t: imaginary time step
        :type t: float
        :return: gate sequence
        :rtype: list[tuple(tuple(tuple(int,int),tuple(int,int),tuple(int,int)), yastn.Tensor)]
        
        Generate a 2-site gate sequence :math:`exp(-t (\vec{S}_i.\vec{S}_j + \sum_{r=i,j}(-1)^{x_r+y_r} B_z S^z_r))` 
        for imaginary-time optimization.
        Each element of sequence has two parts: First, the placement of the gate encoded by (x,y) 
        coords of the two sites and the vector from 1st to 2nd site: (x_1,y_1), 
        (x_2-x_1, y_2-y_1), (x_2,y_2). Second, the 2-site gate Tensor.

        The gate sequance generated::

                    g[0]         g[2]
            g[4]--(0,0)--g[5]--(1,0)--[g[4]]
                    g[1]         g[3]
            g[6]--(0,1)--g[7]--(1,1)--[g[6]]
                   [g[0]]       [g[2]]

        The g[0] and g[2] are the "weak" links, with :math:`\alpha \vec{S}.\vec{S}` interaction, 
        coupling the ladders.
        """
        # two spin gates
        # on-site term is applied 4 times on each site, hence its coupling is rescaled
        # accordingly
        gate_seq=[
            (((0,0),(1,0),(1,0)), self._gen_gate_SS_hz(-t, 1, self.Bz((0,0))/4 ) ),
            (((1,0),(1,0),(0,0)), self._gen_gate_SS_hz(-t, 1, self.Bz((1,0))/4 ) ),
            (((0,1),(1,0),(1,1)), self._gen_gate_SS_hz(-t, 1, self.Bz((0,1))/4 ) ),
            (((1,1),(1,0),(0,1)), self._gen_gate_SS_hz(-t, 1, self.Bz((1,1))/4 ) ),
            (((0,0),(0,1),(0,1)), self._gen_gate_SS_hz(-t, 1, self.Bz((0,0))/4 ) ),
            (((1,0),(0,1),(1,1)), self._gen_gate_SS_hz(-t, 1, self.Bz((1,0))/4 ) ),
            (((0,1),(0,1),(0,0)), self._gen_gate_SS_hz(-t, self.alpha, self.Bz((0,1))/4 ) ),
            (((1,1),(0,1),(1,0)), self._gen_gate_SS_hz(-t, self.alpha, self.Bz((1,1))/4 ) ) 
        ]

    def gen_gate_seq_2S_SS_hz_2ndOrder(self,t):
        r"""
        :param t: imaginary time step
        :type t: float
        :return: gate sequence
        :rtype: list[tuple(tuple(tuple(int,int),tuple(int,int),tuple(int,int)), yastn.Tensor)]
        
        Second-order Trotter gate sequence. This sequence can be generated from the result of 
        :meth:`gen_gate_seq_2S_SS_hz` by applying the gates in both direct and reverse order. 
        """
        # two spin gates
        # on-site term is applied 4 times on each site, hence its coupling is rescaled
        # accordingly
        gate_seq=[
            (((0,0),(1,0),(1,0)), self._gen_gate_SS_hz(-t, 1, self.Bz((0,0))/4 ) ),
            (((1,0),(1,0),(0,0)), self._gen_gate_SS_hz(-t, 1, self.Bz((1,0))/4 ) ),
            (((0,1),(1,0),(1,1)), self._gen_gate_SS_hz(-t, 1, self.Bz((0,1))/4 ) ),
            (((1,1),(1,0),(0,1)), self._gen_gate_SS_hz(-t, 1, self.Bz((1,1))/4 ) ),
            (((0,0),(0,1),(0,1)), self._gen_gate_SS_hz(-t, 1, self.Bz((0,0))/4 ) ),
            (((1,0),(0,1),(1,1)), self._gen_gate_SS_hz(-t, 1, self.Bz((1,0))/4 ) ),
            (((0,1),(0,1),(0,0)), self._gen_gate_SS_hz(-t, self.alpha, self.Bz((0,1))/4 ) ),
            (((1,1),(0,1),(1,0)), self._gen_gate_SS_hz(-2*t, self.alpha, self.Bz((1,1))/4 ) )
        ]

        # repeat the sequence in inverse order
        for i in range( len(gate_seq)-2 ,-1,-1): gate_seq.append(gate_seq[i])
        return gate_seq

    def gen_gate_seq_2S_H(self,t):
        r"""
        :param t: imaginary time step
        :type t: float
        :return: gate sequence
        :rtype: list[tuple(tuple(tuple(int,int),tuple(int,int),tuple(int,int)), Tensor)]
        
        Analogous to :meth:`gen_gate_seq_2S`, with ladders being weakly coupled 
        in horizontal direction.

        The g[5] and g[7] are the "weak" links, with :math:`\alpha\vec{S}.\vec{S}` interaction
        coupling the ladders.
        """
        gate_SS_1= self._gen_gate_SS(-t)
        gate_SS_alpha= self._gen_gate_SS(-t*self.alpha)

        gate_seq=[
            (((0,0),(0,1),(0,1)), gate_SS_1),
            (((1,0),(0,1),(1,1)), gate_SS_1),
            (((0,1),(0,1),(0,0)), gate_SS_1),
            (((1,1),(0,1),(1,0)), gate_SS_1),
            (((0,0),(1,0),(1,0)), gate_SS_1),
            (((0,1),(1,0),(1,1)), gate_SS_1),
            (((1,0),(1,0),(0,0)), gate_SS_alpha),
            (((1,1),(1,0),(0,1)), gate_SS_alpha)
        ]

        # single spin gates 
        # Note: it would be better to join the single spin gates and the two spin gates
        if self.Bz != _null_Bz:
            for x_1 in range(2):
                for y_1 in range(2):
                    dx, dy = (1, 0)
                    x_2 = (x_1 + dx) % 2
                    y_2 = (y_1 + dy) % 2
                    gate_Sz_Bz= self._gen_gate_Sz(-t*self.Bz((x_1, y_2)))
                    gate_seq+=[(((x_1,y_1),(dx,dy),(x_2,y_2)), gate_Sz_Bz)]


        return gate_seq

    def gen_gate_seq_2S_2ndOrder_H(self,t):
        r"""
        :param t: imaginary time step
        :type t: float
        :return: gate sequence
        :rtype: list[tuple(tuple(tuple(int,int),tuple(int,int),tuple(int,int)), yastn.Tensor)]
        
        Second-order Trotter gate sequence. This sequence can be generated from the result of 
        :meth:`gen_gate_seq_2S_H` by applying the gates in both direct and reverse order. 
        """
        gate_SS_1= self._gen_gate_SS(-t)
        gate_SS_2= self._gen_gate_SS(-2*t)
        gate_SS_alpha= self._gen_gate_SS(-t*self.alpha)
        gate_SS_2alpha= self._gen_gate_SS(-2*t*self.alpha)

        gate_seq=[
            (((0,0),(0,1),(0,1)), gate_SS_1),
            (((1,0),(0,1),(1,1)), gate_SS_1),
            (((0,1),(0,1),(0,0)), gate_SS_1),
            (((1,1),(0,1),(1,0)), gate_SS_1),
            (((0,0),(1,0),(1,0)), gate_SS_1),
            (((0,1),(1,0),(1,1)), gate_SS_1),
            (((1,0),(1,0),(0,0)), gate_SS_alpha)            
        ]

        # single spin gates 
        # Note: it would be better to join the single spin gates and the two spin gates
        if self.Bz != _null_Bz:
            for x_1 in range(2):
                for y_1 in range(2):
                    dx, dy = (1, 0)
                    x_2 = (x_1 + dx) % 2
                    y_2 = (y_1 + dy) % 2
                    gate_Sz_Bz= self._gen_gate_Sz(-t*self.Bz((x_1, y_2)))
                    gate_seq+=[(((x_1,y_1),(dx,dy),(x_2,y_2)), gate_Sz_Bz)]


        # last gate 
        gate_seq.append(  (((1,1),(1,0),(0,1)), gate_SS_2alpha) ) 
        
        # repeat the sequence in inverse order
        for i in range(6,-1,-1): gate_seq.append(gate_seq[i])
        return gate_seq