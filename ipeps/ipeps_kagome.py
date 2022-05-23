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
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : torch.tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Initialization of IPEPS_KAGOME follows :class:`ipeps.ipeps.IPEPS`.
        The physical dimension of on-sites tensors is assumed to be a cube  
        of (integer) physical dimension of single DoF of the underlying Kagome system.
        """
        #TODO we infer the size of the cluster from the keys of sites. Is it OK?
        #TODO validate, that on-site physical dimension is 3rd power of integer ?
        super().__init__(sites, vertexToSite=vertexToSite, lX=lX, lY=lY,\
            peps_args=peps_args, global_args=global_args)

    def get_physical_dim(self):
        r"""
        Return physical dimension of a single DoF of the underlying Kagome system. 
        Assume the first dimension of on-site tensor is obtained as a cube 
        of single DoF physical dimension i.e. 3**3 for spin-1
        """
        phys_dims=[]
        for t in self.sites.values():
            assert abs(int( round(t.size(0)**(1./3.)) )**3 - t.size(0)) < 1.0e-8,\
                "Physical dimension of Kagome iPEPS is not a cube of integer"
            dof1_pd= int(round(t.size(0)**(1./3.)))
            if not dof1_pd in phys_dims: phys_dims.append(dof1_pd) 
        assert len(phys_dims)==1, "Kagome sites with different physical DoFs"
        return phys_dims[0]

    def extend_bond_dim(self, new_d, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param new_d: new enlarged auxiliary bond dimension
        :type new_d: int
        :return: wavefunction with enlarged auxiliary bond dimensions
        :rtype: IPEPS_KAGOME

        Take IPEPS_KAGOME and enlarge all auxiliary bond dimensions of all on-site tensors up to 
        size ``new_d``
        """
        new_sites = dict()
        for coord,site in self.sites.items():
            dims = site.size()
            size_check = [new_d >= d for d in dims[1:]]
            if False in size_check:
                raise ValueError("Desired dimension is smaller than following aux dimensions: "+str(size_check))

            new_site = torch.zeros((dims[0],new_d,new_d,new_d,new_d), dtype=self.dtype, device=self.device)
            new_site[:,:dims[1],:dims[2],:dims[3],:dims[4]] = site
            new_sites[coord] = new_site
        
        new_state= self.__class__(new_sites, vertexToSite=self.vertexToSite,\
            peps_args=peps_args, global_args=global_args)
        return new_state

def read_ipeps_kagome(jsonfile, vertexToSite=None, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPEPS_KAGOME in json format
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
    :rtype: IPEPS_KAGOME
    
    See :meth:`ipeps.ipeps.read_ipeps`.
    """
    
    tmp_ipeps= ipeps.read_ipeps(jsonfile, vertexToSite=vertexToSite, aux_seq=aux_seq,\
        peps_args=peps_args, global_args=global_args)

    return IPEPS_KAGOME(tmp_ipeps.sites, vertexToSite=vertexToSite, lX=tmp_ipeps.lX,\
        lY=tmp_ipeps.lY, peps_args=peps_args, global_args=global_args)