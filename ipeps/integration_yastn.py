from __future__ import annotations
from typing import Sequence, Union, TypeVar
import json
import torch
from ipeps.tensor_io import NumPy_Encoder
import yastn.yastn as yastn
from yastn.yastn import Tensor, load_from_dict, save_to_dict
from yastn.yastn.tn.fpeps import Peps, RectangularUnitcell
import config as cfg
YASTN_CONFIG = TypeVar('YASTN_CONFIG')

# apply_and_copy = lambda nested_iterable, func: type(nested_iterable)(
#         apply_and_copy(item, func) if isinstance(item,  dict)) else if  func(item)
#         for item in (nested_iterable.items() if isinstance(nested_iterable, dict) else nested_iterable)
#     )

def apply_and_copy(nested_iterable, func, f_keys=None):
    if isinstance(nested_iterable,dict): 
        if 'type' in nested_iterable and nested_iterable['type'] in ['yastn.Tensor', 'Tensor']:
            # we don't traverse deeper - this is a leaf
            return func(nested_iterable)
        else:
            return { (f_keys(k) if f_keys else k): apply_and_copy(v,func) if isinstance(v,(list,tuple,set,dict)) else func(v) for k,v in nested_iterable.items() }
    elif isinstance(nested_iterable, (list, tuple, set)):
        return type(nested_iterable)( apply_and_copy(v) if isinstance(v,(list,tuple,set,dict)) else func(v) for v in nested_iterable )
    else:
        return func(nested_iterable)


# TODO: one should be able to pass parameters and a callable, which is invoked by sync_ to build on-site tensors
class PepsAD(Peps):
    # TODO accept pattern in the form of i) Sequence[Sequence[Tensor]], then ids of tensors can be labels for lower constructors
    #                        ii) analogously dict[tuple[int,int],Tensor]
    def __init__(
        self,
        geometry=None,
        parameters: Union[
            None, Sequence[Sequence[Tensor]], dict[tuple[int, int], Tensor]
        ] = None,
        get_tensors: callable = None,
        global_args=cfg.global_args,
    ):
        """
        Wrapper around YASTN'n PEPS.
        """
        self.parameters = parameters
        self.dtype = global_args.torch_dtype
        self.device = global_args.device
        
        super().__init__(geometry=geometry,tensors=None)
        self.sync_()


    def sync_(self):
        r"""
        Build on-site tensors of underlying Peps from parameters.
        
        In straightforward case, simply assigns parameter tensors to on-site tensor of underlying PEPS,
        i.e. acts as identity. More complex cases such extend PepsAD and override this function.

        .. Note::
            This function must be invoked as part of cost function to maintain comp. graph between
            logic which build on-site tensors from parameters and subsequent algos (CTMRG, ...).
        """
        for site in self.sites():
            self[site] = self.parameters[site]


    def add_noise_(self, noise : float =1.0):
        r"""
        Add random noise with magnitude ``noise`` to parameters.
        """
        self.parameters= apply_and_copy( self.parameters, lambda x: x + noise * yastn.rand_like(x) )
        self.sync_()


    def get_parameters(self):
        r"""
        :return: variational parameters of iPEPS
        :rtype: iterable

        This function is called by optimizer to access variational parameters of the state.
        """
        flatten = lambda nested_iterable: [t._data for ts in (nested_iterable.values() if isinstance(nested_iterable, dict) else nested_iterable) \
                for t in (flatten(ts.values()) if isinstance(ts, dict) else flatten(ts) if isinstance(ts, (list, tuple, set)) else [ts])]
        return flatten(self.parameters)


    def write_to_file(self, outputfile, tol=None, normalize=False):
        d= self.__dict__()
        # preprocess
        # Handles case when geometry pattern is given as a dictionary with non-compliant keys
        if isinstance(d['geometry']['pattern'],dict) and any([isinstance(k,tuple) for k in d['geometry']['pattern'].keys()]):
            pattern_key_to_id={}
            _pattern={}
            for k in d['geometry']['pattern'].keys():
                if isinstance(k,tuple):
                    new_k= str(k)
                    pattern_key_to_id[new_k]= k
                    _pattern[new_k]= d['geometry']['pattern'][k] 
                else:
                    _pattern[k]= d['geometry']['pattern'][k]
            d['pattern_key_to_id']= pattern_key_to_id
            d['geometry']['pattern']= _pattern
            
        # We don't make any assumption on (nested) structure of parameters. Hence, we remap keys of dicts
        # in parameters if necessary
        parameters_key_to_id={}
        def map_keys(k):
            if isinstance(k,tuple):
                new_k= str(k)+f"_{len(parameters_key_to_id)}"
                parameters_key_to_id[new_k]= k
                return new_k
            else:
                return k
        _parameters= apply_and_copy(self.parameters, save_to_dict, f_keys=map_keys)
        d['parameters_key_to_id']= parameters_key_to_id
        d['parameters']= _parameters
        with open(outputfile, 'w') as file:
            json.dump(d, file, indent=4, cls=NumPy_Encoder)


    def get_checkpoint(self):
        r"""
        :return: serializable representation of PepsAD state
        :rtype: dict

        Return dict containing serialized on-site (block-sparse) tensors. The individual
        blocks are serialized into Numpy ndarrays. This function is called by optimizer
        to create checkpoints during the optimization process.
        """
        return self.__dict__()
    

    def load_checkpoint_(self, yastn_config : YASTN_CONFIG, checkpoint_file):
        r"""
        :param checkpoint_file: path to checkpoint file
        :type checkpoint_file: str

        Initializes the state according to the supplied checkpoint file.
        """
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.parameters = apply_and_copy(checkpoint['parameters']['parameters'], \
                                     lambda x : load_from_dict(yastn_config,x) )
        self.sync_()


    def __repr__(self):
        return (
            f"PepsAD(geometry={self.geometry.__repr__()}, parameters={ self._data })"
        )


    def __dict__(self):
        """
        Serialize PepsAD into a dictionary.
        """
        d = {
            "type": type(self).__name__,
            "lattice": type(self.geometry).__name__,
            "dims": self.dims,
            "geometry": self.geometry.__dict__(),
            "parameters": apply_and_copy(self.parameters, save_to_dict),
        }
        return d

    @staticmethod
    def from_dict(yastn_config, d : dict) -> PepsAD:
        # post-process in case of prior need to remap uncompliant dict keys
        if 'parameters_key_to_id' in d and len(d['parameters_key_to_id'])>0:
            from ast import literal_eval
            def remap_keys(k):
                if k in d['parameters_key_to_id']:
                    k= k[:k.rfind('_')] if k.rfind('_') != -1 else k
                    return literal_eval(k)
                else:
                    return k
            _parameters= apply_and_copy(d['parameters'], lambda x: load_from_dict(yastn_config,x), f_keys=remap_keys)
            d['parameters']= _parameters
        return PepsAD(geometry=RectangularUnitcell(**d['geometry']),
            parameters= d['parameters'],
            global_args=cfg.global_args
        )


def load_PepsAD(yastn_config : YASTN_CONFIG, state_file : str)->PepsAD:
    r"""
    """
    with open(state_file) as f:
        d = json.load(f)
    return PepsAD.from_dict(yastn_config, d)


def load_checkpoint(yastn_config : YASTN_CONFIG, checkpoint_file : str)->PepsAD:
    r"""
    :param checkpoint_file: path to checkpoint file
    :type checkpoint_file: str

    Initializes the state according to the supplied checkpoint file.
    """
    checkpoint = torch.load(checkpoint_file, map_location=yastn_config.default_device, weights_only=False)
    return PepsAD.from_dict(yastn_config, checkpoint["parameters"])


# class PessHoneycomb(PepsAD):


#     def merge_tensor(self, tensors):
#         # Merge two tensors defined on the A, B sublattice of the honeycomb lattice
#         # tensors = [tensorA, tensorB] for A, B sublattice
#         #   0       2   1       t(0)  r(3)
#         #   |        \ /         \   /
#         #   A--3      B--3  =>     B--
#         #  / \        |            |   -> (4)
#         # 1   2       0            A--
#         #                        /   \
#         #                       l(1)  b(2)

#         A, B = tensors[0], tensors[1]
#         ncon_order = ((1, -1, -2, -4), (1, -3, -0, -5))
#         res = yastn.ncon([A, B], ncon_order)
#         if res.get_legs(axes=5).is_fused():
#             res = res.unfuse_legs(axes=5)
#         if res.get_legs(axes=4).is_fused():
#             res = res.unfuse_legs(axes=4)
#         if res.ndim == 6:
#             res = res.fuse_legs(axes=(0, 1, 2, 3, (4, 5)))
#             res = res.drop_leg_history(axes=4)
#         elif res.ndim == 8:
#             res = res.fuse_legs(axes=(0, 1, 2, 3, (4, 6), (5, 7)))
#             res = res.drop_leg_history(axes=4)
#             res = res.fuse_legs(axes=(0, 1, 2, 3, (4, 5)))
#         return res