import torch
from collections import OrderedDict
import json
import math
import config as cfg
import ipeps.ipeps as ipeps
from ipeps.tensor_io import *

class IPEPS_KAGOME(ipeps.IPEPS):
    def __init__(self, sites=None, vertexToSite=None, lX=None, lY=None,\
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
        super().__init__(sites, vertexToSite=vertexToSite, lX=lX, lY=lY,\
            peps_args=peps_args, global_args=global_args)

def read_ipeps_kagome(jsonfile, vertexToSite=None, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    
    tmp_ipeps= ipeps.read_ipeps(jsonfile, vertexToSite=vertexToSite, aux_seq=aux_seq,\
        peps_args=peps_args, global_args=global_args)

    return IPEPS_KAGOME(tmp_ipeps.sites, vertexToSite=vertexToSite, lX=tmp_ipeps.lX,\
        lY=tmp_ipeps.lY, peps_args=peps_args, global_args=global_args)