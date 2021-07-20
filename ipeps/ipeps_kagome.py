import torch
from collections import OrderedDict
import json
import math
import config as cfg
import ipeps.ipeps as ipeps
from ipeps.tensor_io import *

class IPEPS_KAGOME(ipeps.IPEPS):
    def __init__(self, kagome_tensors, vertexToSite=None, lX=None, lY=None, \
        peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param kagome_tensors: list of selected KAGOME-type tensors A, B, C, R1, R2; A', B', C', R1', R2'; ...
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type kagome_tensors: list[tuple(dict(str,str,str), torch.tensor)]
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
        

        where in the last example we used default setting for ``vertexToSite``, which
        maps square lattice into elementary unit cell of size `lX` x `lY` assuming 
        periodic boundary conditions (PBC) along both X and Y directions.

        Kagome sites:
        coord_kagome - ( 0, 0, x=0(A),1(B),2(C),3(RD),4(RU) )

        """
        self.lX = lX
        self.lY = lY
        if vertexToSite is not None:
            self.vertexToSite = vertexToSite
        else:
            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ((x + abs(x)*self.lX) % self.lX, (y + abs(y)*self.lY) % self.lY)
            self.vertexToSite = vertexToSite

        self.kagome_tensors = kagome_tensors
        self.phys_dim = self.get_physical_dim()
        self.bond_dims = self.get_aux_bond_dims()
        sites = self.build_unit_cell_tensors_kagome()

        super().__init__(sites, vertexToSite=vertexToSite, peps_args=peps_args,\
            global_args=global_args)

    def __str__(self):
        print(f"lX x lY: {self.lX} x {self.lY}")
        for nid, coord_kagome, kagome_site in [(t[0], *t[1]) for t in enumerate(self.kagome_tensors.items())]:
            print(f"A{nid} {coord_kagome}: {kagome_site.size()}")

        # show the unit cell
        unit_cell = """-- A^a_kl   B^b_mn --
--  \\       /
      R1_lno
        |
      C^c_op
        |
      R2_pqr
    /       \\
"""
        print(unit_cell)

        return ""

    def get_parameters(self):
        return self.kagome_tensors.values()

    def get_checkpoint(self):
        return self.kagome_tensors

    def load_checkpoint(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.kagome_tensors = checkpoint["parameters"]
        for kagome_t in self.kagome_tensors.values():
            kagome_t.requires_grad_(False)
        self.sites = self.build_unit_cell_tensors_kagome()

    def kagome_vertex_to_vertex(self, coord_kagome):
        x = coord_kagome[0]
        y = coord_kagome[1]
        # z = coord_kagome[2]
        return (x, y)

    def build_unit_cell_tensors_kagome(self):
        sites = dict()
        coords = []
        for coord_kagome in self.kagome_tensors.keys():
            tmp_coord = self.kagome_vertex_to_vertex(coord_kagome)
            if torch.all(torch.tensor(coords != tmp_coord, dtype=torch.bool)):
                coords.append(tmp_coord)
        for coord in coords:
            t_a = self.kagome_tensors[(coord[0], coord[1], 0)]
            t_b = self.kagome_tensors[(coord[0], coord[1], 1)]
            t_c = self.kagome_tensors[(coord[0], coord[1], 2)]
            t_r1 = self.kagome_tensors[(coord[0], coord[1], 3)]
            t_r2 = self.kagome_tensors[(coord[0], coord[1], 4)]
            sites[coord] = torch.einsum('akl,lno,bmn,cop,pqr->abcmrqk', t_a, t_r1, t_b, t_c, t_r2).flatten(start_dim=0, end_dim=2)

        return sites

    def add_noise(self, noise):
        r"""
        :param noise: magnitude of the noise
        :type noise: float

        Take IPEPS and add random uniform noise with magnitude ``noise`` to all on-site tensors
        """
        if self.dtype == torch.float64:
            for coord_kagome in self.kagome_tensors.keys():
                rand_t = torch.rand(self.kagome_tensors[coord_kagome].size(), dtype=self.dtype, device=self.device)
                self.kagome_tensors[coord_kagome] = self.kagome_tensors[coord_kagome] + noise * rand_t
        elif self.dtype == torch.complex128:
            for coord_kagome in self.kagome_tensors.keys():
                rand_t_real = torch.rand(self.kagome_tensors[coord_kagome].size(), dtype=self.dtype, device=self.device)
                rand_t_img = torch.rand(self.kagome_tensors[coord_kagome].size(), dtype=self.dtype, device=self.device)
                self.kagome_tensors[coord_kagome] = self.kagome_tensors[coord_kagome] + noise * rand_t_real
                self.kagome_tensors[coord_kagome] = self.kagome_tensors[coord_kagome] + noise * rand_t_img * 1j
        else:
            raise Exception("Unsuppoted data type. Optional: \"float64\", \"complex128\".")


    def get_physical_dim(self):
        phys_dim = None
        for coord_kagome, t in self.kagome_tensors.items():
            if phys_dim == None:
                if coord_kagome[2] == 0 or coord_kagome[2] == 1 or coord_kagome[2] == 2:
                    phys_dim = t.size()[0]
        return phys_dim

    def get_aux_bond_dims(self):
        bond_dims = None
        coords = []
        for coord_kagome in self.kagome_tensors.keys():
            tmp_coord = self.kagome_vertex_to_vertex(coord_kagome)
            if torch.all(torch.tensor(coords != tmp_coord, dtype=torch.bool)):
                coords.append(tmp_coord)
        for coord in coords:
            t_a = self.kagome_tensors[(coord[0], coord[1], 0)]
            t_b = self.kagome_tensors[(coord[0], coord[1], 1)]
            t_r2 = self.kagome_tensors[(coord[0], coord[1], 4)]
            if bond_dims == None:
                bond_dims = [t_b.size()[1], t_r2.size()[1], t_r2.size()[2], t_a.size()[1]]

        return bond_dims

    def write_to_file(self, outputfile, tol=1.0e-14, normalize=False):
        write_ipeps_kagome(self, outputfile, tol=tol, normalize=normalize)


def extend_bond_dim_kagome(state, new_d):
    coords = []
    new_state = state
    for coord_kagome in state.kagome_tensors.keys():
        tmp_coord = state.kagome_vertex_to_vertex(coord_kagome)
        if torch.all(torch.tensor(coords != tmp_coord, dtype=torch.bool)):
            coords.append(tmp_coord)
    for coord in coords:
        for i in [0, 1, 2, 3, 4]:
            if i < 3:
                t = state.kagome_tensors[(coord[0], coord[1], i)]
                dims = t.size()
                new_t = torch.zeros((dims[0], new_d, new_d), dtype=state.dtype, device=state.device)
                new_t[:, :dims[1], :dims[2]] = t
                new_state.kagome_tensors[(coord[0], coord[1], i)] = new_t
            else:
                t = state.kagome_tensors[(coord[0], coord[1], i)]
                dims = t.size()
                new_t = torch.zeros((new_d, new_d, new_d), dtype=state.dtype, device=state.device)
                new_t[:dims[0], :dims[1], :dims[2]] = t
                new_state.kagome_tensors[(coord[0], coord[1], i)] = new_t
    new_state.build_unit_cell_tensors_kagome()

    return new_state


def read_ipeps_kagome(jsonfile, vertexToSite=None, peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
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
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS
    

    A simple PBC ``vertexToSite`` function is used by default
    """
    # asq = [x+1 for x in aux_seq]
    kagome_tensors = OrderedDict()
    with open(jsonfile) as j:
        raw_state = json.load(j)
        # read the list of considered kagome-type tensors
        for ts in raw_state["map"]:
            coord_kagome = (ts["x"],ts["y"],ts["z"])
            t = None
            for s in raw_state["kagome_sites"]:
                if s["siteId"] == ts["siteId"]:
                    t = s
            if t == None:
                raise Exception("Tensor with siteId: "+ts["sideId"]+" NOT FOUND in \"sites\"")

            if "format" in t.keys():
                if t["format"] == "1D":
                    X = torch.from_numpy(read_bare_json_tensor_np(t))
            else:
                # default
                X = torch.from_numpy(read_bare_json_tensor_np_legacy(t))

            kagome_tensors[coord_kagome] = X.to(device=global_args.device)

        # Unless given, construct a function mapping from
        # any site of square-lattice back to unit-cell
        if vertexToSite == None:
            # check for legacy keys
            lX = 0
            lY = 0
            lX = raw_state["sizeM"] if "sizeM" in raw_state else raw_state["lX"]
            lY = raw_state["sizeN"] if "sizeN" in raw_state else raw_state["lY"]

            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*lX)%lX, (y + abs(y)*lY)%lY )

            state = IPEPS_KAGOME(kagome_tensors=kagome_tensors, vertexToSite=vertexToSite, \
                lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)
        else:
            state = IPEPS_KAGOME(kagome_tensors=kagome_tensors, vertexToSite=vertexToSite, \
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
    :type tol: float
    :type normalize: bool

    TODO implement cutoff on elements with magnitude below tol
    """
    json_state = dict({"lX": state.lX, "lY": state.lY, "kagome_sites": []})

    def vertexToSite(coord):
        x = coord[0]
        y = coord[1]
        return ((x + abs(x) * state.lX) % state.lX, (y + abs(y) * state.lY) % state.lY)

    site_ids = []
    site_map = []
    # The default input/output order
    tensor_names = ['A', 'B', 'C', 'RD', 'RU']
    for coord_kagome, kagome_tensor in state.kagome_tensors.items():
        if normalize:
            kagome_tensor = kagome_tensor / kagome_tensor.abs().max()
        tensor_name = tensor_names[coord_kagome[2]%5]
        nid = coord_kagome[0] + coord_kagome[1] * state.lX
        site_ids.append(f"{tensor_name}{nid}")
        site_map.append(dict({"siteId": site_ids[-1], "x": coord_kagome[0], "y": coord_kagome[1], "z": coord_kagome[2]}))

        json_tensor = serialize_bare_tensor_legacy(kagome_tensor)
        json_tensor["siteId"]=site_ids[-1]
        json_state["kagome_sites"].append(json_tensor)

    json_state["siteIds"] = site_ids
    json_state["map"] = site_map

    with open(outputfile, 'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))

