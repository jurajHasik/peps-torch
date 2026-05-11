import torch
# import config as cfg
from ctm.generic import rdm
from ctm.generic import transferops
import numpy as np
from linalg.ncon_torch import ncon_torch as ncon


####################################################################################################

def _cast_to_real(t):
    return t.real if t.is_complex() else t


####################################################################################################

class TORICCODE_1x1:
    def __init__(self,
                 jv=1.0,
                 jp=1.0,
                 hx=0.1,
                 hz=0.1,
                 global_args=None,
                 ):
        r"""
        The Hamiltonian of the square latice toric code model in a field is defined as
        H=-jv*∑_{v} A_v - jp*∑_{p} B_p - hx*∑_e X_e - hz*∑_e Z_e

        A_v=\prod_{e\in v}X_e is the vertex term,
        B_p=\prod_{e\in p}Z_e  is the plaquette term,
        X_e is the transverse field term
        Z_e is the longtitudnal term.

        By combine two physical degrees of freedom into one, we can define 1x1 unit cell iPEPS on the primal lattice. The tensor looks like:
               u
               |
               |     s2
               |      |
       l ------------------- r
               |vertex
               |---- s1
               |
               d
       where u, l, d,r are virtual legs and s1,s2 are physical legs.

        :type jv: float
        :type jp: float
        :type hx: float
        :type hz: float
        :type global_args: GLOBALARGS

        """

        self.jv = jv
        self.jp = jp
        self.hx = hx
        self.hz = hz


        self.dtype = global_args.dtype
        self.device = global_args.device
        print('device=',global_args.device)

        # Z4 Pauli matrices
        self.phys_dim = 4
        X= torch.tensor([[0, 1], [1, 0]],dtype= torch.float64, device=self.device)
        Z= torch.tensor([[1, 0], [0, -1]], dtype=torch.float64, device=self.device)
        Id= torch.tensor([[1, 0], [0, 1]], dtype=torch.float64, device=self.device)
        self.X_Id = (torch.einsum('ia,jb->ijab',X, Id)).reshape(4,4)
        self.Id_X = (torch.einsum('ia,jb->ijab', Id, X)).reshape(4,4)
        self.Z_Id = (torch.einsum('ia,jb->ijab', Z, Id)).reshape(4,4)
        self.Id_Z = (torch.einsum('ia,jb->ijab', Id, Z)).reshape(4,4)
        self.X_X = (torch.einsum('ia,jb->ijab', X, X)).reshape(4,4)
        self.Z_Z = (torch.einsum('ia,jb->ijab', Z, Z)).reshape(4,4)
        self.Id_Id = (torch.einsum('ia,jb->ijab', Id, Id)).reshape(4,4)

#       order of physical DOF:
        # s0,s1
        # s2,s3

               ## Tensor prducts for Hamiltonian terms
        self.Av = torch.einsum('ia,jb,kc->ijkabc',
                          self.X_Id,
                          self.Id_X,
                          self.X_X
                          ).contiguous()


        self.Bp = torch.einsum('ia,jb,kc->ijkabc',
                          self.Z_Z,
                          self.Z_Id,
                          self.Id_Z,
                          ).contiguous()


        # Hamiltonian terms
        self.ham2x2_jv = -jv * self.Av
        self.ham2x2_jp = -jp * self.Bp
        self.ham1x1_hx = -hx * (self.Id_X+self.X_Id)
        self.ham1x1_hz = -hz * (self.Id_Z+self.Z_Id)


    def compute_rdm_terms(self, state, env):
        """
        Compute all required RDMs for energy and observables.
        """
        rdm2x2_v = rdm.rdm2x2((0, 0), state, env,open_sites=[1,2,3])
        rdm2x2_p = rdm.rdm2x2((0, 0), state, env,open_sites=[0,1,2])# Vertex and plaq term
        rdm1x1 = rdm.rdm1x1((0, 0), state, env)  # field term

        return {"rdm2x2_v": rdm2x2_v,
                "rdm2x2_p": rdm2x2_p,
                "rdm1x1": rdm1x1}

    def energy(self, state, env):
        """
        Compute the total energy of the system in the bipartite 2x2 block structure.
        """
        rdms = self.compute_rdm_terms(state, env)
        energy_jv = torch.einsum('ijkabc,ijkabc', rdms["rdm2x2_v"], self.ham2x2_jv)
        energy_jp = torch.einsum('ijkabc,ijkabc', rdms["rdm2x2_p"], self.ham2x2_jp)
        energy_hx = torch.einsum('ij,ij', rdms["rdm1x1"], self.ham1x1_hx)
        energy_hz = torch.einsum('ij,ij', rdms["rdm1x1"], self.ham1x1_hz)

        total_energy = _cast_to_real(energy_jv + energy_jp + energy_hx + energy_hz)/2
        return total_energy.contiguous()

    def eval_obs(self, state, env):
        """
        Evaluate observables in the system without gradient tracking.
        """
        with torch.no_grad():
            rdms = self.compute_rdm_terms(state, env)
            obs = {
                "Av": torch.einsum('ijkabc,ijkabc', rdms["rdm2x2_v"], self.Av),
                "Bp": torch.einsum('ijkabc,ijkabc', rdms["rdm2x2_p"], self.Bp),
                "X": torch.einsum('ij,ij', rdms["rdm1x1"], self.X_Id+ self.Id_X)/2,
                "Z": torch.einsum('ij,ij', rdms["rdm1x1"], self.Z_Id + self.Id_Z)/2
            }

            obs_values = [v.item() for v in list(obs.values())]
            obs_labels = list(obs.keys())
            return obs_values, obs_labels

    def eval_corrlen(self, state, env):
        """
        Compute transfer operator spectrum.
        """
        # Number of eigenvalues to consider
        with torch.no_grad():
            # Number of eigenvalues to consider
            Ns = 3

            # Define coordinates and directions for evaluation
            coord_dir_pairs = [
                {"coord": (0, 0), "direction": (1, 0)},  # x-direction from (0,0)
                {"coord": (0, 0), "direction": (0, 1)},  # y-direction from (0,0)
                # {"coord": (1, 1), "direction": (1, 0)},  # x-direction from (1,1)
                # {"coord": (1, 1), "direction": (0, 1)},  # y-direction from (1,1)
            ]

            # Initialize results
            corrlengths = {}
            eigenvalues = {}

            # Compute the transfer operator spectrum along x and y directions
            for idx, pair in enumerate(coord_dir_pairs):
                coord, direction = pair["coord"], pair["direction"]
                # Get the transfer operator spectrum
                spectrum = transferops.get_Top_spec(Ns, coord, direction, state, env)
                lambdas = [torch.abs(spectrum[i, 0] + 1j * spectrum[i, 1]) for i in range(Ns)]
                # Compute correlation length
                corrlen = torch.tensor([-1 / torch.log(lambdas[1] / lambdas[0]),
                                        -1 / torch.log(lambdas[2] / lambdas[0]),
                                        ])

                # Define keys based on direction and coordinate
                key_suffix = f"_{coord[0]}{coord[1]}_{'x' if direction == (1, 0) else 'y'}"
                corrlengths[f"corrlen{key_suffix}"] = corrlen
                eigenvalues[f"lambdas{key_suffix}"] = lambdas

            # Combine results
            corrlen_obs = {
                "lengths": corrlengths,
                "lambdas": eigenvalues,
            }

        return corrlen_obs

    def FPT_TC(self):
        # Generate fixed point tensor the toric code ground state
        Q = torch.zeros((2, 2, 2),dtype= torch.float64, device=self.device)
        Q[0, 0, 0] = 1
        Q[1, 1, 0] = 1
        Q[1, 0, 1] = 1
        Q[0, 1, 1] = 1

        Delta = torch.zeros((2, 2, 2, 2),dtype= torch.float64, device=self.device)
        Delta[0, 0, 0, 0] = 1
        Delta[1, 1, 1, 1] = 1

        T = ncon([Q,Q,Delta],[[1,-1,-5],[2,-2,-6],[-3,-4,1,2]])
        T = T.reshape(4,2, 2, 2, 2)
        return T

class TORICCODE_2x2:
    def __init__(self,
                 jv=1.0,
                 jp=1.0,
                 hx=0.1,
                 hz=0.1,
                 global_args=None,
                 ):
        r"""
        The Hamiltonian of the square latice toric code model in a field is defined as
        H=-jv*∑_{v} A_v - jp*∑_{p} B_p - hx*∑_e X_e - hz*∑_e Z_e

        A_v=\prod_{e\in v}X_e is the vertex term,
        B_p=\prod_{e\in p}Z_e  is the plaquette term,
        X_e is the transverse field term
        Z_e is the longtitudnal term.

        :type jv: float
        :type jp: float
        :type hx: float
        :type hz: float
        :type global_args: GLOBALARGS


       The iPEPS is defined on the medial lattice with 2x2 unit cell, which is a bipartite structure, see details in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.115143
       or https://arxiv.org/abs/1912.00908
       We use non-symmetric CTMRG to comtract the iPEPS.
        """

        self.jv = jv
        self.jp = jp
        self.hx = hx
        self.hz = hz


        self.dtype = global_args.dtype
        self.device = global_args.device
        print('device=',global_args.device)

        # Z4 Pauli matrices
        self.phys_dim = 2
        self.X= torch.tensor([[0, 1], [1, 0]],dtype= torch.float64, device=self.device)
        self.Z= torch.tensor([[1, 0], [0, -1]], dtype=torch.float64, device=self.device)
        self.Id= torch.tensor([[1, 0], [0, 1]], dtype=torch.float64, device=self.device)


#       order of physical DOF:
        # s0,s1
        # s2,s3

               ## Tensor prducts for Hamiltonian terms
        self.Av = torch.einsum('ia,jb,kc,ld->ijklabcd',
                          self.X,
                          self.X,
                          self.X,
                          self.X
                          ).contiguous()


        self.Bp = torch.einsum('ia,jb,kc,ld->ijklabcd',
                          self.Z,
                          self.Z,
                          self.Z,
                          self.Z,
                          ).contiguous()


        # Hamiltonian terms
        self.ham2x2_jv = -jv * self.Av
        self.ham2x2_jp = -jp * self.Bp
        self.ham1x1_hx = -hx * self.X
        self.ham1x1_hz = -hz * self.Z


    def compute_rdm_terms(self, state, env):
        """
        Compute all required RDMs for energy and observables.
        """
        rdm2x2_v = (rdm.rdm2x2((0, 0), state, env)+rdm.rdm2x2((1, 1), state, env))/2  # Vertex and plaq term
        rdm2x2_p = (rdm.rdm2x2((0, 1), state, env)+rdm.rdm2x2((1, 0), state, env))/2
        rdm1x1 = (rdm.rdm1x1((0, 0), state, env)+rdm.rdm1x1((0, 1), state, env)+rdm.rdm1x1((1, 0), state, env)+rdm.rdm1x1((1, 1), state, env))/2

        return {"rdm2x2_v": rdm2x2_v,"rdm2x2_p": rdm2x2_p,"rdm1x1":rdm1x1}

    def energy(self, state, env):
        """
        Compute the total energy of the system in the bipartite 2x2 block structure.
        """
        rdms = self.compute_rdm_terms(state, env)
        energy_jv = torch.einsum('ijklabcd,ijklabcd', rdms["rdm2x2_v"], self.ham2x2_jv)
        energy_jp = torch.einsum('ijklabcd,ijklabcd', rdms["rdm2x2_p"], self.ham2x2_jp)
        energy_hx = torch.einsum('ij,ij', rdms["rdm1x1"], self.ham1x1_hx)
        energy_hz = torch.einsum('ij,ij', rdms["rdm1x1"], self.ham1x1_hz)

        total_energy = _cast_to_real(energy_jv + energy_jp + energy_hx + energy_hz)/2
        return total_energy.contiguous()

    def eval_obs(self, state, env):
        """
        Evaluate observables in the system without gradient tracking.
        """
        with torch.no_grad():
            rdms = self.compute_rdm_terms(state, env)
            obs = {
                "Av": torch.einsum('ijklabcd,ijklabcd', rdms["rdm2x2_v"], self.Av),
                "Bp": torch.einsum('ijklabcd,ijklabcd', rdms["rdm2x2_p"], self.Bp),
                "X": torch.einsum('ij,ij', rdms["rdm1x1"], self.X)/2,
                "Z": torch.einsum('ij,ij', rdms["rdm1x1"], self.Z)/2
            }

            obs_values = [v.item() for v in list(obs.values())]
            obs_labels = list(obs.keys())
            return obs_values, obs_labels

    def eval_corrlen(self, state, env):
        """
        Compute transfer operator spectrum.
        """
        # Number of eigenvalues to consider
        with torch.no_grad():
            # Number of eigenvalues to consider
            Ns = 3

            # Define coordinates and directions for evaluation
            coord_dir_pairs = [
                {"coord": (0, 0), "direction": (1, 0)},  # x-direction from (0,0)
                {"coord": (0, 0), "direction": (0, 1)},  # y-direction from (0,0)
                # {"coord": (1, 1), "direction": (1, 0)},  # x-direction from (1,1)
                # {"coord": (1, 1), "direction": (0, 1)},  # y-direction from (1,1)
            ]

            # Initialize results
            corrlengths = {}
            eigenvalues = {}

            # Compute the transfer operator spectrum along x and y directions
            for idx, pair in enumerate(coord_dir_pairs):
                coord, direction = pair["coord"], pair["direction"]
                # Get the transfer operator spectrum
                spectrum = transferops.get_Top_spec(Ns, coord, direction, state, env)
                lambdas = [torch.abs(spectrum[i, 0] + 1j * spectrum[i, 1]) for i in range(Ns)]
                # Compute correlation length
                corrlen = torch.tensor([-1 / torch.log(lambdas[1] / lambdas[0]),
                                        -1 / torch.log(lambdas[2] / lambdas[0]),
                                        ])

                # Define keys based on direction and coordinate
                key_suffix = f"_{coord[0]}{coord[1]}_{'x' if direction == (1, 0) else 'y'}"
                corrlengths[f"corrlen{key_suffix}"] = corrlen
                eigenvalues[f"lambdas{key_suffix}"] = lambdas

            # Combine results
            corrlen_obs = {
                "lengths": corrlengths,
                "lambdas": eigenvalues,
            }

        return corrlen_obs
    def FPT_TC(self):
        # Generate fixed point tensor the toric code ground state
        Q = torch.zeros((2, 2, 2),dtype= torch.float64, device=self.device)
        Q[0, 0, 0] = 1
        Q[1, 1, 0] = 1
        Q[1, 0, 1] = 1
        Q[0, 1, 1] = 1

        Delta = torch.zeros((2, 2, 2),dtype= torch.float64, device=self.device)
        Delta[0, 0, 0] = 1
        Delta[1, 1, 1] = 1

        TA = ncon([Q, Delta, Q], [[1, -3, -4], [1, 2, -1], [2, -2, -5]])
        TB = ncon([Q, Delta, Q], [[1, -3, -2], [1, 2, -1], [2, -4, -5]])
        # TA = ncon([Delta, Q,Delta], [[1, -3, -4], [1, 2, -1], [2, -2, -5]])
        # TB = ncon([Delta, Q,Delta], [[1, -3, -2], [1, 2, -1], [2, -4, -5]])

        return TA,TB