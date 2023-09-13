from collections import OrderedDict
import json
import math
try:
    import torch
    from ipeps.ipeps import IPEPS
except ImportError as e:
    warnings.warn("torch not available", Warning)
import config as cfg
import ipeps.ipeps_abelian as ipeps_abelian
from ipeps.tensor_io import *

class IPEPS_KAGOME_ABELIAN(ipeps_abelian.IPEPS_ABELIAN):
    def __init__(self, settings, sites=None, vertexToSite=None, lX=None, lY=None,\
            build_open_dl=True, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param settings: YAST configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : yastn.Tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        This class serves as an iPEPS interface for CTMRG algorithm over different
        tensor networks defined over Kagome lattice. Various TNs on Kagome lattice
        are transformed into standard iPEPS on square lattice by contraction over
        some of the bonds in the network. The physical space of on-site tensors 
        of resulting iPEPS is typically tensor product of 3 DoFs of the original
        system::       
                           
               /|   /|   /|          |  |  |
              /_|__/_|__/_|__  =>  --A--A--A-- 
                | /  | /  | /        |  |  |
                |/   |/   |/       --A--A--A--
               /|   /|   /|          |  |  |
              /_|__/_|__/_|__
                | /  | /  | /
                |/   |/   |/ <- down triangle


        where all 3 DoFs on the vertices of down triangles are fused
        into single physical DoF of on-site tensor A. The state on the up-triangles
        is thus spanned by considering 3 sites A of resulting square-lattice iPEPS. 
        """
        super().__init__(settings, sites, vertexToSite=vertexToSite, lX=lX, lY=lY,\
            peps_args=peps_args, global_args=global_args)

def read_ipeps_kagome(jsonfile, settings, vertexToSite=None, peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPEPS_KAGOME_ABELIAN in json format
    :param settings: YAST configuration
    :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                         into a vertex within elementary unit cell
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
    :type vertexToSite: function(tuple(int,int))->tuple(int,int)
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_KAGOME_ABELIAN
    
    Read IPEPS_KAGOME_ABELIAN from file.
    """
    tmp_ipeps= ipeps_abelian.read_ipeps(jsonfile, settings, vertexToSite=vertexToSite,\
        peps_args=peps_args, global_args=global_args)

    return IPEPS_KAGOME_ABELIAN(settings, sites=tmp_ipeps.sites, vertexToSite=vertexToSite,\
        lX=tmp_ipeps.lX,lY=tmp_ipeps.lY, peps_args=peps_args, global_args=global_args)