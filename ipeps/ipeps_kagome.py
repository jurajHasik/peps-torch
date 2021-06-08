import torch
from collections import OrderedDict
import json
import math
import config as cfg
import ipeps.ipeps as ipeps
from ipeps.tensor_io import *

class IPEPS_KAGOME(ipeps.IPEPS):
    def __init__(self, triangle_up, bond_site, triangle_down=None, 
                 SYM_UP_DOWN=True, 
                 peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param sym_tensors: list of selected symmetric tensors
        :param coeffs: map from elementary unit cell to vector of coefficients
        :param vertexToSite: function mapping arbitrary vertex of a square lattice
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sym_tensors: list[tuple(dict(str,str), torch.tensor)]
        :type coeffs: dict[tuple(int,int) : torch.tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Member ``sites`` is a dictionary of non-equivalent on-site tensors
        indexed by tuple of coordinates (x,y) within the elementary unit cell.
        The index-position convetion for on-site tensors is defined as follows::

               u s
               |/
            l--A--r  <=> A[s,u,l,d,r]
               |
               d

            where s denotes physical index, and u,l,d,r label four principal directions
            up, left, down, right in anti-clockwise order starting from up

        Member ``vertexToSite`` is a mapping function from vertex on a square lattice
        passed in as tuple(x,y) to a corresponding tuple(x,y) within elementary unit cell.

        On-site tensor of an IPEPS object ``wfc`` at vertex (x,y) is conveniently accessed
        through the member function ``site``, which internally uses ``vertexToSite`` mapping::

            coord= (0,0)
            A_00= wfc.site(coord)

        By combining the appropriate ``vertexToSite`` mapping function with elementary unit
        cell specified through ``sites`` various tilings of a square lattice can be achieved::

            # Example 1: 1-site translational iPEPS

            sites={(0,0): A}
            def vertexToSite(coord):
                return (0,0)
            wfc= IPEPS(sites,vertexToSite)

            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  A A A A
            # -1   A  A A A A
            #  0   A  A A A A
            #  1   A  A A A A

            # Example 2: 2-site bipartite iPEPS

            sites={(0,0): A, (1,0): B}
            def vertexToSite(coord):
                x = (coord[0] + abs(coord[0]) * 2) % 2
                y = abs(coord[1])
                return ((x + y) % 2, 0)
            wfc= IPEPS(sites,vertexToSite)

            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  B A B A
            # -1   B  A B A B
            #  0   A  B A B A
            #  1   B  A B A B

            # Example 3: iPEPS with 3x2 unit cell with PBC

            sites={(0,0): A, (1,0): B, (2,0): C, (0,1): D, (1,1): E, (2,1): F}
            wfc= IPEPS(sites,lX=3,lY=2)

            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   B  C A B C
            # -1   E  F D E F
            #  0   B  C A B C
            #  1   E  F D E F

        where in the last example we used default setting for ``vertexToSite``, which
        maps square lattice into elementary unit cell of size `lX` x `lY` assuming
        periodic boundary conditions (PBC) along both X and Y directions.

        TODO we infer the size of the cluster from the keys of sites. Is it OK?
        """
        self.SYM_UP_DOWN= SYM_UP_DOWN
        self.elem_tensors= OrderedDict({'UP_T': triangle_up, 'BOND_S': bond_site})
        if not SYM_UP_DOWN:
            assert isinstance(triangle_down,torch.Tensor),\
                "rank-3 tensor for down triangle must be provided"
            self.elem_tensors['DOWN_T']= triangle_down
        else:
            self.elem_tensors['DOWN_T']= triangle_up

        sites = self.build_onsite_tensors()

        super().__init__(sites, lX=1, lY=1, peps_args=peps_args,
                         global_args=global_args)

    def __str__(self):
        print(f"Symmetric up and down triangle: {self.SYM_UP_DOWN}")
        print(f"lX x lY: {self.lX} x {self.lY}")
        for nid, coord, site in [(t[0], *t[1]) for t in enumerate(self.coeffs.items())]:
            print(f"A{nid} {coord}: {site.size()}")

        # show tiling of a square lattice
        coord_list = list(self.coeffs.keys())
        mx, my = 3 * self.lX, 3 * self.lY
        label_spacing = 1 + int(math.log10(len(self.coeffs.keys())))
        for y in range(-my, my):
            if y == -my:
                print("y\\x ", end="")
                for x in range(-mx, mx):
                    print(str(x) + label_spacing * " " + " ", end="")
                print("")
            print(f"{y:+} ", end="")
            for x in range(-mx, mx):
                print(f"A{coord_list.index(self.vertexToSite((x, y)))} ", end="")
            print("")

        return ""

    def get_parameters(self):
        if self.SYM_UP_DOWN:
            return [self.elem_tensors['UP_T'], self.elem_tensors['BOND_S']]
        else:
            return self.elem_tensors.values()

    def get_checkpoint(self):
        return self.elem_tensors

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file)
        self.elem_tensors= checkpoint["parameters"]
        if self.SYM_UP_DOWN:
            self.elem_tensors['DOWN_T']= self.elem_tensors['UP_T']
        for t in self.elem_tensors.values(): t.requires_grad_(False)
        self.sites = self.build_onsite_tensors()

    def build_onsite_tensors(self):
        # square-lattice tensor with 3 physical indices (d=3)
        a_tensor_temp = torch.einsum('iab,uji,jkl,vkc,wld->uvwabcd', self.elem_tensors['UP_T'],\
            self.elem_tensors['BOND_S'], self.elem_tensors['DOWN_T'], self.elem_tensors['BOND_S'], \
            self.elem_tensors['BOND_S'])
        # reshape to a single d=27 index
        # TODO can we substitute this by a simple reshape ? 
        D = a_tensor_temp.size(3)
        a_tensor = torch.zeros((27, D, D, D, D), dtype=torch.complex128, device=a_tensor_temp.device)
        for si in range(27):
            n1, n2, n3 = fmap_inv(si)
            a_tensor[si, :, :, :, :] = a_tensor_temp[n1, n2, n3, :, :, :, :]
        # END TODO
        sites= {(0, 0): a_tensor}
        return sites

    def add_noise(self, noise):
        for k in self.elem_tensors:
            rand_t= torch.rand( self.elem_tensors[k].size(), dtype=self.dtype, device=self.device)
            self.elem_tensors[k]= self.elem_tensors[k] + noise * (rand_t-1.0)
        if self.SYM_UP_DOWN:
            self.elem_tensors['DOWN_T']= self.elem_tensors['UP_T']
        self.sites = self.build_onsite_tensors()

    def get_aux_bond_dims(self):
        auxd_set= set(self.elem_tensors['UP_T'].size()).union(set(self.elem_tensors['DOWN_T'].size()),\
            set(self.elem_tensors['UP_T'].size()[1:]))
        return auxd_set

    def write_to_file(self, outputfile, aux_seq=None, tol=1.0e-14, normalize=False):
        write_ipeps_kagome(self, outputfile, tol=tol, normalize=normalize)

def extend_bond_dim(state, new_d):
    r"""
    :param state: wavefunction to modify
    :param new_d: new enlarged auxiliary bond dimension
    :type state: IPEPS
    :type new_d: int
    :return: wavefunction with enlarged auxiliary bond dimensions
    :rtype: IPEPS

    Take IPEPS and enlarge all auxiliary bond dimensions of all on-site tensors up to 
    size ``new_d``
    """
    auxd_set= list(state.get_aux_bond_dims())
    assert len(auxd_set)==1, "different auxiliary dimensions for elem tensors"
    assert new_d>=auxd_set[0], "Desired dimension is smaller than current aux dimension"
    ad= auxd_set[0]
    pd= state.elem_tensors['BOND_S'].size(0)
    new_elem_tensors= dict()
    new_elem_tensors['UP_T']= torch.zeros(new_d,new_d,new_d, dtype=state.dtype, device=state.device)
    new_elem_tensors['UP_T'][:ad,:ad,:ad]= state.elem_tensors['UP_T']
    new_elem_tensors['DOWN_T']= torch.zeros(new_d,new_d,new_d, dtype=state.dtype, device=state.device)
    new_elem_tensors['DOWN_T'][:ad,:ad,:ad]= state.elem_tensors['DOWN_T']
    new_elem_tensors['BOND_S']= torch.zeros(pd,new_d,new_d, dtype=state.dtype, device=state.device)
    new_elem_tensors['BOND_S'][:,:ad,:ad]= state.elem_tensors['UP_T']

    new_state= state.__class__(new_elem_tensors['UP_T'], new_elem_tensors['BOND_S'],\
        triangle_down=new_elem_tensors['DOWN_T'],\
        SYM_UP_DOWN=state.SYM_UP_DOWN,\
        peps_args=cfg.peps_args, global_args=cfg.global_args)

    return new_state

def read_ipeps_kagome(jsonfile, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPS in json format
    :param vertexToSite: function mapping arbitrary vertex of a square lattice
                         into a vertex within elementary unit cell
    :param aux_seq: array specifying order of auxiliary indices of on-site tensors stored
                    in `jsonfile`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type vertexToSite: function(tuple(int,int))->tuple(int,int)
    :type aux_seq: list[int]
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS


    A simple PBC ``vertexToSite`` function is used by default

    Parameter ``aux_seq`` defines the expected order of auxiliary indices
    in input file relative to the convention fixed in tn-torch::

         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2

        for alternative order, eg.

         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2]
         3
    """
    dtype = global_args.torch_dtype

    with open(jsonfile) as j:
        raw_state = json.load(j)

        SYM_UP_DOWN= raw_state["SYM_UP_DOWN"] 

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        elem_tensors= OrderedDict()
        assert set(('UP_T', 'BOND_S', 'DOWN_T'))==set(list(raw_state["elem_tensors"].keys())),\
            "missing elementary tensors"
        for key,t in raw_state["elem_tensors"].items():
            elem_tensors[key]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))\
                .to(global_args.device)

        if SYM_UP_DOWN: elem_tensors['DOWN_T']=None

        state = IPEPS_KAGOME(elem_tensors['UP_T'], elem_tensors['BOND_S'], \
            triangle_down=elem_tensors['DOWN_T'], SYM_UP_DOWN=SYM_UP_DOWN, \
            peps_args=peps_args, global_args=global_args)
    return state

def write_ipeps_kagome(state, outputfile, tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param aux_seq: array specifying order in which the auxiliary indices of on-site tensors
                    will be stored in the `outputfile`
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPS
    :type ouputfile: str or Path object
    :type aux_seq: list[int]
    :type tol: float
    :type normalize: bool

    Parameter ``aux_seq`` defines the order of auxiliary indices relative to the convention
    fixed in tn-torch in which the tensor elements are written out::

         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2

        for alternative order, eg.

         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2]
         3

    TODO drop constrain for aux bond dimension to be identical on
    all bond indices

    TODO implement cutoff on elements with magnitude below tol
    """
    json_state = dict({"lX": state.lX, "lY": state.lY, \
        "elem_tensors": {}, "SYM_UP_DOWN": state.SYM_UP_DOWN})

    # write list of considered elementary tensors
    for key, t in state.elem_tensors.items():
        json_state["elem_tensors"][key]= serialize_bare_tensor_legacy(t)

    with open(outputfile, 'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))


def fmap(n1, n2, n3):
    return n3 + 3 * n2 + 9 * n1

def fmap_inv(s):
    n1 = s // 9
    n2 = (s - 9 * n1) // 3
    n3 = s - 9 * n1 - 3 * n2
    return n1, n2, n3
