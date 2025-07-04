import json
import os
import pickle
from itertools import cycle

import context
import config as cfg
import numpy as np
import torch

import matplotlib.pyplot as plt

from ipeps.integration_yastn import load_PepsAD
from edge_spec import load_env_from_dict

import yastn.yastn as yastn
from yastn.yastn.sym import sym_U1
from yastn.yastn.tn.fpeps.envs.rdm import op_order
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.backend import backend_torch as backend

def apply_TM_TAT(state, env, site, dirn, V, op=None):
    r"""
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param site: tuple (x,y) specifying vertex on a square lattice
    :param dirn: direction in which the transfer operator is applied
    :param V: tensor of dimensions :math:`\chi \times D^2 \times \chi (\times d_{aux})`,
                 potentially with 4th auxiliary index
    :param op: operator to be inserted into transfer matrix

    :type state: PepsAD
    :type env: yastn.fn.fpeps.EnvCTM
    :type site: yastn.tn.fpeps.Site
    :type dirn: tuple(int,int)
    :type edge: yastn.Tensor
    :type op: yastn.Tensor
    :return: Resulting tensor from applying the transfer matrix applied to V.
             The tensor either has an identical index structure as the original V
             or has an additional auxiliary index from op.
    :rtype: yastn.tensor
    """
    def get_dl_tensor(op, dirn):
        # Forming double tensor
        A_top, A_bot = state[site].unfuse_legs(axes=(0, 1)), state[site].unfuse_legs(axes=(0, 1))
        A_bot = A_bot.swap_gate(axes=(0, 1, 2, 3)) # t' x l', b' x r'
        if op is None: # identity operator
            dl_tensor = A_top.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r t' l' b' r'
            dl_tensor = dl_tensor.swap_gate(axes=(1, 4, 2, 7)) # l x t', b x r'
            dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7))) # [t t'] [l l'] [b b'] [r r']
            #   \ \
            # --|--A--------
            #   |  | \
            #   |  |  \                     \       \     / \
            #    \ |   \                ----Ah---= --\---Ac--\---
            # -----Ah---\---                 \        \ /     \
            #       \    \
        else:
            dims_op = op.get_shape()
            if len(dims_op) == 2: # no aux index
                # check if a dummy leg is fused with the physical leg
                leg = A_top.get_legs(axes=4)
                if leg.is_fused():
                    A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux

                    dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p
                    dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3,(5,4))) # t l b r [p p_aux]
                else:
                    dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p

                dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4))
                dl_tensor = dl_tensor.swap_gate(axes=(1, 4, 2, 7)) # l x t', b x r'
                dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7))) # [t t'] [l l'] [b b'] [r r']
                #   \ \
                # --|--A--------
                #   |  | \
                #   |  O  \                     \       \     / \
                #    \ |   \                ----Ah---= --\---Ac--\---
                # -----Ah---\---                 \        \ /     \
                #       \    \
            elif len(dims_op) == 3: #  op has an extra index to make it charge-neutrual
                if dirn in [(1, 0), (0, 1), (0, -1)]:
                    # check if a dummy leg is fused with the physical leg
                    leg = A_top.get_legs(axes=4)
                    if leg.is_fused():
                        A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux
                        dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p aux
                        dl_tensor = dl_tensor.swap_gate(axes=(4, 6))
                        dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3, (5,4), 6)) # t l b r [p p_aux] aux
                    else:
                        dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p aux


                    dl_tensor = dl_tensor.swap_gate(axes=(5, (2, 3))) # aux x b r
                    dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r aux t' l' b' r'
                    dl_tensor = dl_tensor.transpose(axes=(0, 1, 2, 3, 5, 6, 7, 8, 4)) # t l b r t' l' b' r' aux
                    dl_tensor = dl_tensor.swap_gate(axes=(1, 4, 2, 7)) # l x t', b x r'
                    dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7), 8)) # [t t'] [l l'] [b b'] [r r'] aux
                    #
                    #   \ \        ____ (aux)
                    # --|--A-----/---
                    #   |  | \ /
                    #   |  O-/\                     \       \     / \
                    #    \ |   \                ----Ah---= --\---Ac--\---
                    # -----Ah---\---                 \        \ /     \
                    #       \    \
                elif dirn in [(-1, 0)]:
                    # check if a dummy leg is fused with the physical leg
                    leg = A_top.get_legs(axes=4)
                    if leg.is_fused():
                        A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux
                        dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p aux
                        dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3, (5,4), 6))
                    else:
                        dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p aux

                    dl_tensor = dl_tensor.swap_gate(axes=(5, 1)) # aux x l
                    dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r aux t' l' b' r'
                    dl_tensor = dl_tensor.transpose(axes=(0, 1, 2, 3, 5, 6, 7, 8, 4)) # t l b r t' l' b' r' aux
                    dl_tensor = dl_tensor.swap_gate(axes=(4, (1, 8), 2, 7)) # t' x [l, aux], b x r'
                    dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7), 8)) # [t t'] [l l'] [b b'] [r r'] aux
                    # aux
                    #  \ \ \
                    # -|-|--A--------
                    # | |  | \
                    # --|--O  \                     \       \     / \
                    #    \ |   \                ----Ah---= --\---Ac--\---
                    # -----Ah---\---                 \        \ /     \
                    #       \    \

        return dl_tensor

    dl_tensor = get_dl_tensor(op, dirn)
    dl_dim = len(dl_tensor.get_shape())
    V_dim = len(V.get_shape())
    if dirn == (0, 1):
        # right action
        #               ___aux
        # ---0  --T_t-/---
        # |       | /
        # V--1  --A --           if dl_dim == 5 and V_dim == 3
        # |       |
        # ---2 --T_b--
        # Or
        # ---(aux)
        # ---0  --T_t--
        # |       |
        # V--1  --A --           if dl_dim == 4 and V_dim == 3 or 4
        # |       |
        # ---2 --T_b--

        T_t, T_b = env[site].t, env[site].b
        # 0 --T_t -- 2              1
        #     |                    |
        #     1               2---T_b---0

        res = V.tensordot(T_t, (0, 0))
        if V_dim == 4:
            res = res.transpose(axes=(0, 1, 3, 4, 2))
        # ------------- (4)
        # --------T_t----3
        # |       |
        # V--0    2
        # |
        # ---1

        if dl_dim == 4:
            res = res.tensordot(dl_tensor, ([0, 2], [1, 0]))
            if V_dim == 4:
                res = res.transpose(axes=(0,1,3,4,2))
            # -----------(4)
            #  -------T_t---1
            # |       |
            # V-------A----- 3
            # |       |
            # ---0    2
        else:
            assert dl_dim == 5 and V_dim == 3
            res = res.tensordot(dl_tensor, ([0, 2], [1, 0]))
            res = res.swap_gate(axes=(1, 4))
            #               -----4
            #  -------T_t-/--1
            # |       | /
            # V-------A----- 3
            # |       |
            # ---0   2


        res = res.tensordot(T_b, ([0, 2], [2, 1]))
        if len(res.get_shape()) == 4:
            res = res.transpose(axes=(0,1,3,2))
        # -------(3)
        #  -------T_t---0
        # |       |
        # V-------A-----1
        # |       |
        #  -------T_b---2

    elif dirn == (0, -1):
        # left action
        # aux --\
        #      --\-T_t--   0--
        #         \ |        |
        #      ----A---   1--V      if dl_dim == 5 and V_dim == 3
        #          |        |
        #      ---T_b--  2--
        # Or
        #       (aux)---
        # --T_t--   0---
        #   |         |
        # --A----  1--V      if dl_dim == 4 and V_dim == 3 or 4
        #   |         |
        # --T_b--  2--
        #

        T_t, T_b = env[site].t.to_dense(), env[site].b.to_dense()
        # 0 --T_t -- 2              1
        #     ||                   ||
        #     1               2---T_b---0

        res = V.tensordot(T_t, (0, 2))
        if V_dim == 4:
            res = res.transpose(axes=(0, 1, 3, 4, 2))
        #     (4) ---
        # 2--T_t-----
        #    |      |
        #    3 0----V
        #           |
        #      1----

        if dl_dim == 4:
            res = res.tensordot(dl_tensor, ([0, 3], [3, 0]))
            if V_dim == 4:
                res = res.transpose(axes=(0, 1, 3, 4, 2))
            #    (4)---
            # 1--T_t---
            #    |    |
            # 2--A----V
            #    |    |
            #    3 0--
        else:
            assert dl_dim == 5 and V_dim == 3
            res = res.tensordot(dl_tensor, ([0, 3], [3, 0]))
            res = res.swap_gate(axes=(1, 4))
            #  4
            # 1-\--T_t---
            #    \ |    |
            #   2--A----V
            #      |    |
            #      3 0--

        res = res.tensordot( T_b, ([0, 3], [0, 1]))
        if len(res.get_shape()) == 4:
            res = res.transpose(axes=(0,1,3,2))
        # (3)--------
        # 0--T_t----
        #    |     |
        # 1--A-----V
        #    |     |
        # 2--T_b----
    elif dirn == (-1, 0):
        # up action
        # \  \  \/ aux
        #  \  \/ \
        # T_l-A- T_r       if dl_dim == 5 and V_dim == 3
        #   0   1   2
        #    \   \  \
        #    ----V----
        # Or
        #  \  \  \   \ (aux)
        # T_l--A--T_r \      if dl_dim == 4 and V_dim == 3 or 4
        #   0   1  2   \
        #    \   \  \   \
        #      ----V------

        T_l, T_r = env[site].l, env[site].r
        #  2                0
        #  |                |
        #  T_l-1          1-T_r
        #  |               |
        #  0               2
        res = V.tensordot(T_l, ([0], [0]))
        if V_dim == 4:
            res = res.transpose(axes=(0, 1, 3, 4, 2))
        # 3
        #  \
        # T_l-- 2
        #   \   0   1   (4)
        #    \   \   \   \
        #     ------V------

        if dl_dim == 4:
            res = res.tensordot(dl_tensor, ([0, 2], [2, 1]))
            if V_dim == 4:
                res = res.transpose(axes=(0, 1, 3, 4, 2))
            # 1   2
            #  \   \
            # T_l---A--3
            #   \   \   0   (4)
            #    \   \   \   \
            #     ------V------
        else:
            assert dl_dim == 5 and V_dim == 3
            res = res.tensordot(dl_tensor, axes=([0,2], [2, 1]))

            # 1   2    4
            #  \   \ /
            # T_l---A--3
            #   \   \    0
            #    \   \    \
            #     ----V-----
        res = res.tensordot(T_r, ([0, 3], [2, 1]))
        if dl_dim == 5:
            res = res.swap_gate(axes=(2, 3))
            #         3   2(aux)
            # 0   1    \/
            #  \   \  / \
            # T_l---A---TR
            #   \   \    \
            #    \   \    \
            #     ----V-----
        if len(res.get_shape()) == 4:
            res = res.transpose(axes=(0, 1, 3, 2))
        # 0   1    2
        #  \   \    \
        # T_l---A---TR
        #   \   \    \ \ (aux)
        #    \   \    \ \
        #     ----V------

    elif dirn == (1, 0):
        # down action
        # ---V---
        #  \  \  \ /---
        #   \  \ /\    \
        #  T_l--A--T_r  \          if dl_dim == 5 and V_dim == 3
        #     \  \  \    \ aux
        # Or
        # ---V---------
        #  \  \  \    \
        #  T_l--A--T_r \          if dl_dim == 4 and V_dim == 3 or 4
        #    \  \  \    \ (aux)

        T_l, T_r = env[site].l, env[site].r
        #  2                0
        #  |                |
        #  T_l-1          1-T_r
        #  |               |
        #  0               2
        res = V.tensordot(T_l, ([0], [2]))
        if V_dim == 4:
            res = res.transpose(axes=(0,1,3,4,2))
        # ------V---------
        #  \     \    \   \
        #  T_l-3  0    1   \
        #    \              \ (4)
        #     2
        if dl_dim == 4:
            res = res.tensordot(dl_tensor, ([0, 3], [0, 1]))
            if V_dim == 4:
                res = res.transpose(axes=(0,1,3,4,2))
            #  ------V---------
            #  \     \      \   \
            #  T_l----A-- 3  0   \
            #    \     \          \ (4)
            #     1    2
        else:
            assert dl_dim == 5 and V_dim == 3
            res = res.tensordot(dl_tensor, ([0,3], [0,1]))
            res = res.swap_gate(axes=(0, 4))
            #  ------V-------
            #  \     \ /----\---
            #  T_l----A-- 3  \  \
            #    \     \      \  \
            #     1     2      0  4

        res = res.tensordot(T_r, ([0, 3], [0, 1]))
        if (len(res.get_shape())) == 4:
            res = res.transpose(axes=(0,1,3,2))

        # ---V---------
        #  \  \  \    \
        #  T_l--A--T_r \
        #    \  \  \    \ (aux)
    return res

def get_edge(state, env, site, dirn):
    """
    Build an edge of site ``coord`` by contracting one of the following networks
    depending on the chosen ``direction``::

            up=(0,-1)   left=(-1,0)  down=(0,1)   right=(1,0)

                         C--0         0  1  2       0--C
                         |            |  |  |          |
        E =  C--T--C     T--1         C--T--C       1--T
             |  |  |     |                             |
             0  1  2     C--2                       2--C

    """
    if dirn == (-1, 0):
        C1, C2 = env[site].tl, env[site].tr
        T = env[site].t
        res = C1.tensordot(T, axes=(1, 0))
        # C1--T--2
        # |   |
        # 0   1
        res = res.tensordot(C2, axes=(2, 0))
        # C1--T--C2
        # |   |  |
        # 0   1  2
    elif dirn == (0, -1):
        C1, C2 = env[site].tl, env[site].bl
        T = env[site].l
        res = C1.tensordot(T, axes=(0, 2))
        # C1--0
        # |
        # T--2
        # |
        # 1
        res = res.tensordot(C2, axes=(1, 1))
        # C1--0
        # |
        # T--1
        # |
        # C2--2
    elif dirn == (1, 0):
        C1, C2 = env[site].bl, env[site].br
        T = env[site].b
        res = C1.tensordot(T, axes=(0, 2))
        #  0   2
        #  |   |
        # C1 --T-- 1
        res = res.tensordot(C2, axes=(1, 1))
        #  0   1    2
        #  |   |    |
        # C1 --T-- C2
    elif dirn == (0, 1):
        C1, C2 = env[site].tr, env[site].br
        T = env[site].r
        res = C1.tensordot(T, axes=(1, 0))
        # 0 --C1
        #     |
        # 1 --T
        #     |
        #     2
        res = res.tensordot(C2, axes=(2, 0))
        # 0 --C1
        #     |
        # 1 --T
        #     |
        # 2 --C2
    return res

def apply_TM_TAT_contract_aux(state, env, site, dirn, V, op):
    r"""
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param site: tuple (x,y) specifying vertex on a square lattice
    :param dirn: direction in which the transfer operator is applied
    :param V: tensor of dimensions :math:`\chi \times D^2 \times \chi \times d_{aux}`
    :param op: operator to be inserted into transfer matrix, with an additional aux. index

    :type state: PepsAD
    :type env: yastn.fn.fpeps.EnvCTM
    :type site: yastn.tn.fpeps.Site
    :type dirn: tuple(int,int)
    :type edge: yastn.Tensor
    :type op: yastn.Tensor
    :return: Resulting tensor from applying the transfer matrix applied to V.
             The aux. index of tensor is contracted with the aux. index of op.
    :rtype: yastn.tensor
    """


    def get_dl_tensor(op, dirn):
        A_top, A_bot = state[site].unfuse_legs(axes=(0, 1)), state[site].unfuse_legs(axes=(0, 1))
        A_bot = A_bot.swap_gate(axes=(0, 1, 2, 3)) # t' x l', b' x r'
        dims_op = op.get_shape()
        assert len(dims_op) == 3 # extra index to make op charge-neutrual
        if dirn in [(-1, 0), (0, -1), (0, 1)]:
            leg = A_top.get_legs(axes=4)
            if leg.is_fused():
                A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux
                dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p aux
                dl_tensor = dl_tensor.swap_gate(axes=(4, 6))
                dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3, (5,4), 6)) # t l b r [p p_aux] aux
            else:
                dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p aux

            dl_tensor = dl_tensor.swap_gate(axes=(5, (2, 3))) # aux x b r
            dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r aux t' l' b' r'
            dl_tensor = dl_tensor.transpose(axes=(0, 1, 2, 3, 5, 6, 7, 8, 4)) # t l b r t' l' b' r' aux
            dl_tensor = dl_tensor.swap_gate(axes=(1, 4, 2, 7)) # l x t', b x r'
            dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7), 8)) # [t t'] [l l'] [b b'] [r r'] aux
            #
            #   \ \        ____ (aux)
            # --|--A-----/---
            #   |  | \ /
            #   |  O-/\                     \       \     / \
            #    \ |   \                ----Ah---= --\---Ac--\---
            # -----Ah---\---                 \        \ /     \
            #       \    \

        elif dirn in [(1, 0)]:
            leg = A_top.get_legs(axes=4)
            if leg.is_fused():
                A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux
                dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p aux
                dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3, (5,4), 6))
            else:
                dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p aux

            dl_tensor = dl_tensor.swap_gate(axes=(5, 1)) # aux x l
            dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r aux t' l' b' r'
            dl_tensor = dl_tensor.transpose(axes=(0, 1, 2, 3, 5, 6, 7, 8, 4)) # t l b r t' l' b' r' aux
            dl_tensor = dl_tensor.swap_gate(axes=(4, (1, 8), 2, 7)) # t' x [l, aux], b x r'
            dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7), 8)) # [t t'] [l l'] [b b'] [r r'] aux
            # aux
            #  \ \ \
            # -|-|--A--------
            # | |  | \
            # --|--O  \                     \       \     / \
            #    \ |   \                ----Ah---= --\---Ac--\---
            # -----Ah---\---                 \        \ /     \
            #       \    \

        return dl_tensor

    dl_tensor = get_dl_tensor(op, dirn)
    assert len(V.get_shape()) == 4
    if dirn == (0, 1):
        # right action
        # ----
        #  ----\--T_t---0
        # |     \ |
        # V-------A-----1
        # |       |
        #  -------T_b---2
        T_t, T_b = env[site].t, env[site].b
        # 0 --T_t -- 2              1
        #     |                    |
        #     1               2---T_b---0

        V = V.swap_gate(axes=(3, 0))
        # ------
        # -----\---0
        # |     (3)
        # V--1
        # |
        # ---2
        res = V.tensordot(T_t, (0, 0))
        res = res.transpose(axes=(0, 1, 3, 4, 2))
        # ----
        # ----\---T_t----3
        # |   (4) |
        # V--0    2
        # |
        # ---1

        res = res.tensordot(dl_tensor, ([0, 2, 4], [1, 0, 4]))
        # contract the auxiliary indices
        #  ----
        #  ----\-T_t---1
        # |     \ |
        # V-------A-----3
        # |       |
        # ---0    2

        res = res.tensordot(T_b, ([0, 2], [2, 1]))
        # ----
        #  ----\--T_t---0
        # |     \ |
        # V-------A-----1
        # |       |
        #  -------T_b---2
    elif dirn == (0, -1):
        # left action
        #          --
        # 0--T_t-/--
        #    | /   |
        # 1--A-----V
        #    |     |
        # 2--T_b----
        T_t, T_b = env[site].t, env[site].b
        # 0 --T_t -- 2              1
        #     ||                   ||
        #     1               2---T_b---0

        V = V.swap_gate(axes=(3, 0))
        #      --
        #  0-/--
        # (3)  |
        #  1---V
        #  2---|
        #
        res = V.tensordot(T_t, (0, 2))
        res = res.transpose(axes=(0, 1, 3, 4, 2))
        #           __
        # 2--T_t--/--
        #    | (4)  |
        #    3 0----V
        #           |
        #      1----

        res = res.tensordot(dl_tensor, ([0, 3, 4], [3, 0, 4]))
        #           __
        # 1--T_t--/---
        #    |  /    |
        # 2--A------ V
        #    |       |
        #    3  0----

        res = res.tensordot(T_b, ([0, 3], [0, 1]))
        #          --
        # 0--T_t-/--
        #    | /   |
        # 1--A-----V
        #    |     |
        # 2--T_b----

    elif dirn == (-1, 0):
        # up action
        # \  \   \/\
        #  \  \ / \ \
        #  T_l-A-T_r \
        #   \0  \1 \2 \(aux)
        #    ---V------
        T_l, T_r = env[site].l, env[site].r
        #  2                0
        #  |                |
        #  T_l-1          1-T_r
        #  |               |
        #  0               2
        res = V.tensordot(T_r, ([2], [2]))
        res = res.swap_gate(axes=(2,3))
        #      3
        #       \ /\
        #       /\  \
        #  (aux)2 \  \
        #     1 4-T_r \
        #  0   \   \   \
        #   \   \   \   \
        #   ----V---------
        res = res.tensordot(dl_tensor, ([1, 4, 2], [2, 3, 4]))
        #       1
        #     2 \ /\
        #     \ |\  \
        #      \| \  \
        #   3--A--T_r \
        #  0   \   \   \
        #   \   \   \   \
        #   ----V---------
        res = res.tensordot(T_l, ([0, 3], [0, 1]))
        res = res.transpose((2,1,0))
        # 0  1   2
        # \  \   \/\
        #  \  \ / \ \
        #  T_l-A-T_r \
        #   \  \  \   \(aux)
        #    ---V------
    elif dirn == (1, 0):
        # down action
        # -----V----\
        # \  \   \ __\(aux)
        #  \  \ / \
        #  T_l-A---T_r
        #   \  \    \

        T_l, T_r = env[site].l, env[site].r
        #  2                0
        #  |                |
        #  TL-1          1-TR
        #  |               |
        #  0               2
        res = V.tensordot(T_l, ([0], [2]))
        # ----V--------
        # \    \   \  \
        # \     0   1  2
        # T_l-4
        # \
        # 3
        res = res.swap_gate(axes=(1,2))
        # ----V--------
        # \    \   \  \
        # \     0   \ /
        # T_l-4     /\
        # \       2   1
        # 3

        res = res.tensordot(dl_tensor, ([0, 4, 2], [0, 1, 4]))
        # -----V----\
        # \  \   \ __\(aux)
        #  \  \ / \
        #  T_l-A-3 0
        #   \  \
        #   1   2

        res = res.tensordot(T_r, ([0, 3], [0, 1]))
        # -----V----\
        # \  \   \ __\(aux)
        #  \  \ / \
        #  T_l-A---T_r
        #   \  \    \
        #    0   1   2
    return res


def corr(state, env, site, dirn, op1, op2, dist, connected=True):
    c0 = site
    rev_dirn = (-dirn[0], -dirn[1])
    E0 = get_edge(state, env, site, rev_dirn)

    if connected:
        op1_val = env.measure_1site(op1, site=c0).item()
        op2_vals = []
    E_1 = apply_TM_TAT(state, env, c0, dirn, E0, op=op1)
    E_N =  apply_TM_TAT(state, env, c0, dirn, E0, op=None)

    corrf = torch.empty(dist, dtype=torch.complex128, device=state.device)

    for r in range(dist):
        #
        #       C--T--- [ --T-- ]^r --T---
        # E12 = T--O1-- [ --A-- ]   --O2--
        #       C--T--- [ --T-- ]   --T---
        c0 = state.nn_site(c0, dirn)
        if len(op2.get_shape()) == 3:
            E_12 = apply_TM_TAT_contract_aux(state, env, c0, dirn, E_1, op=op2)
        else:
            E_12 = apply_TM_TAT(state, env, c0, dirn, E_1, op=op2)
        if connected:
            op2_vals.append(env.measure_1site(op2, site=c0).item())
        E_1 = apply_TM_TAT(state, env, c0, dirn, E_1, op=None)
        E_N = apply_TM_TAT(state, env, c0, dirn, E_N, op=None)

        E_end = get_edge(state, env, c0, dirn)
        val_12 = E_12.tensordot(E_end, axes=((0,1,2), (0,1,2)))
        val_norm = E_N.tensordot(E_end, axes=((0,1,2), (0,1,2)))
        corrf[r] = val_12.to_number()/val_norm.to_number()

        # normalize by largest element of E_N
        max_elem_EN = yastn.linalg.norm(abs(E_N), p='inf')
        E_N=E_N/max_elem_EN
        E_1=E_1/max_elem_EN

    if connected:
        return corrf, np.array(op2_vals)*op1_val
    else:
        return corrf, None

if __name__ == "__main__":
    state_file = (
        # f"./t1t2t3_V1_grad/seed_123/V1_1.0_t1_0.1_t2_0.07_t3_-0.09_3x3_N9_D_6_chi_72_fullrank_cpu_cont4_state.json"
        f"./t1t2t3_V1_grad/CDW_3x3/V1_1.0_V2_0.5_V3_0.5_t1_0.05_t2_0.035_t3_-0.045_phi_0_3x3_N3_D_6_chi_72_fullrank_cpu_state.json"
    )

    yastn_config = yastn.make_config(
        backend=backend,
        sym=sym_U1,
        fermionic=True,
        default_device="cpu",
        default_dtype="complex128",
    )
    D = 6
    opt_chi = 72
    omp_cores = 16
    torch.set_num_threads(omp_cores)

    # Converge environment
    opts_svd = {
        "D_total": opt_chi,
        "tol": 1e-8,
        "eps_multiplet": 1e-8,
        "fix_signs": True,
        "truncate_multiplets": True,
    }

    env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(opt_chi,))
    # env_dict_filename = "./t1t2t3_V1_grad/seed_123/es/env_dict_V1_1.0_t1_0.1_t2_0.07_t3_-0.09_3x3_N9_D_6_chi_72_fullrank_cpu"
    env_dict_filename = "./t1t2t3_V1_grad/CDW_3x3/es/env_dict_V1_1.0_V2_0.5_V3_0.5_t1_0.05_t2_0.035_t3_-0.045_phi_0_3x3_N3_optchi_72_chi_72_fullrank_cpu"

    state = load_PepsAD(yastn_config, state_file)
    with open(env_dict_filename, "rb") as handle:
        d = pickle.load(handle)
    env = EnvCTM(state, init="eye", leg=env_leg)
    env = load_env_from_dict(env, d,  yastn_config)
    print("Loaded env")


    _tmp_config = {x: y for x, y in yastn_config._asdict().items() if x != "sym"}
    sf = yastn.operators.SpinfulFermions(sym=str(yastn_config.sym), **_tmp_config)
    n_A = sf.n(spin="u")  # parity-even operator, no swap gate needed
    n_B = sf.n(spin="d")
    c_A = sf.c(spin="u")
    cp_A = sf.cp(spin="u")
    c_B = sf.c(spin="d")
    cp_B = sf.cp(spin="d")
    I = sf.I()

    torch.set_printoptions(precision=10)

    ci_A, cjp_A = op_order(c_A, cp_A, ordered=True, fermionic=True)
    cip_A, cj_B = op_order(cp_A, c_B, ordered=True, fermionic=True)

    for i, site in enumerate(state.sites()):
        dist = 40
        # corrf, _ = corr(state, env, site, (0, 1), ci_A, cjp_A, dist, connected=False)
        corrf, op_vals = corr(state, env, site, (1, 0), n_B, n_A, dist, connected=True)
        o1 = env.measure_1site(n_B@n_A, site=site).item() - env.measure_1site(n_A, site=site).item()*env.measure_1site(n_B, site=site).item()
        # filename = f"obs/FCI_3x3_N9/cA_cpA_corrf_site_{i:d}_dirn_(0,1).npy"
        # filename = f"obs/CDW_3x3_N3/cA_cpA_corrf_site_{i:d}_dirn_(0,1).npy"
        filename = f"obs/CDW_3x3_N3/nB_nA_corrf_site_{i:d}_dirn_(1,0).npy"
        with open(filename, 'wb') as f:
            # np.save(f, np.arange(1, dist+1))
            # np.save(f, corrf - op_vals)
            # np.save(f, corrf)
            np.save(f, np.arange(0, dist+1))
            np.save(f, np.insert(corrf - op_vals, 0, o1))

        # # plt.style.use("science")
        # fig, ax = plt.subplots()
        # ax.plot(np.arange(1, dist+1), corrf-op_vals, lw=2, marker='o', markersize=6)
        # # ax.set_yscale('log')
        # # ax.set_ylabel(r"$|\langle c_{00, A} c^\dagger_{r0, A}\rangle|$")
        # ax.set_ylabel(r"$\langle n_{00}^{A} n_{r0}^{B}\rangle - \langle n_{00}^{A} \rangle \langle n_{r0}^{A}\rangle$")
        # ax.set_xlabel(r"$r$")
        # plt.savefig(f"figs/nA_nB_corrf_site_{i:d}_dirn_(1,0).pdf")

