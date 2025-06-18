import logging
from functools import lru_cache
from itertools import product
import gc, subprocess
import time

import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
import opt_einsum as oe  # type: ignore
from opt_einsum.contract import (  # type: ignore
    PathInfo,
    shape_only,
)
try:
    from opt_einsum.contract import _VALID_CONTRACT_KWARGS
except:
    _VALID_CONTRACT_KWARGS = {'optimize', 'memory_limit', 'einsum_call', 'use_blas', 'shapes'}
try:
    import arrayfire as af
except:
    print("Warning: Missing arrayfire. SVDAF is not available.")
from profiling import _debug_allocated_tensors

log = logging.getLogger(__name__)


def _preprocess_interleaved_to_expr_and_shapes(*args, unroll=[]):
    r"""Casts interleaved einsum input into default format, stripping
    away unrolled indices if any.
    Collects shapes of the input and output tensors, labeling shapes
    of unrolled indices as negative values.
    Collects shapes of unrolled indices.

    This functions preprocesses the input for _get_contraction_path_cached
    allowing for caching.

    :param args: input to einsum in interleaved format
    :param unroll: indices to unroll
    """
    # assert that unroll indices are contracted over, i.e. appear at least twice for
    # at least two different tensors
    if len(unroll) > 0:
        assert not any(
            [sum([u_i in x for x in (args[1::2] + (args[-1],))]) < 2 for u_i in unroll]
        ), "Invalid choice of unrolled index"

    # cast interleaved format to default einsum while dropping unrolled indices
    #
    # the interleaved format has a) even number of elements, if the (i) the result is a scalar
    #                               or (ii) tensor sorted in default index order
    #                            b) odd number of elements if the result is a tensor and order of output indices
    #                               is explicitly specified
    to_ints = set([i for ig in args[1::2] for i in ig])
    to_ints = {i: idx for idx, i in enumerate(to_ints)}

    expr = ",".join(
        [
            "".join(["" if y in unroll else oe.get_symbol(to_ints[y]) for y in x])
            for x in args[1::2]
        ]
    )
    expr += "->" + "".join(
        ["" if y in unroll else oe.get_symbol(to_ints[y]) for y in args[-1]]
    )

    # assign shape to each index label
    i_to_s = {
        i: s
        for ig, t in zip(args[1::2], args[0 : 2 * (len(args) // 2) : 2])
        for i, s in zip(ig, t.shape)
    }

    # create shapes information, labeling shapes on unrolled dimensions as negative
    shapes = tuple(
        tuple(i_to_s[i] if not (i in unroll) else -i_to_s[i] for i in ig)
        for ig in args[1::2] + (args[-1],)
    )
    unrolled_shapes = tuple(i_to_s[i] for i in unroll)

    return expr, shapes, unrolled_shapes

@lru_cache(maxsize=128)
def _log_input_mem_size(shapes : tuple[tuple[int]],names=None,who=None,**kwargs):
    # log the total size (memory footprint) of tensors entering contraction
    if kwargs.get("verbosity",1):
        if names:
            assert len(names) == len(shapes),"Number of names has to match number of operands"    
            in_mem_list=", ".join(f"{n} {t} {np.asarray(t).prod()}" for n, t in zip(names, shapes))
        else:
            in_mem_list=", ".join(f"{n} {t} {np.asarray(t).prod()}" for n, t in enumerate(shapes))
        log.info(f"{who} input sizes "+in_mem_list)
        log.info(f"{who} total size {sum(np.asarray(t).prod() for t in shapes)}")


def get_contraction_path(*tn_to_contract, unroll=[], names=None, who=None, **kwargs):
    r"""Returns optimal contraction path for tensor network contraction specified in interleaved
    format. Takes into account unrolled indices if any.

    :param tn_to_contract: input to einsum in interleaved format. Explicit index labeling
                           of output is required
    :param unroll: indices to unroll
    :param names: string labels for tensors used for more readable logging. The order of
                  names has to follow order of tensors as they appear in ``tn_to_contract``
    :param who: string id for logging identifying this optimal contraction path search
    """

    # require explicit specification of output index labels
    assert (
        len(tn_to_contract) % 2 == 1
    ), "Explicit specification of output index labels is required"
    _log_input_mem_size(tuple(tuple(t.shape) for t in tn_to_contract[:-1][0::2]),names=names,who=who,**kwargs)

    expr, shapes, unrolled_shapes = _preprocess_interleaved_to_expr_and_shapes(
        *tn_to_contract, unroll=unroll if unroll else []
    )
    t0= time.perf_counter()
    res= _get_contraction_path_cached(
        expr, shapes, unrolled=unrolled_shapes, names=names, who=who, **kwargs
    )
    t1= time.perf_counter()
    log.info(f"{who} contraction path search took {t1-t0} [s]")
    return res


@lru_cache(maxsize=128)
def _get_contraction_path_cached(
    expr, shapes, unrolled=(), names=None, who=None, **kwargs
):
    r"""Cachable function finding optimal contraction path for tensor network contraction
    specified in default einsum format with shapes only.

    :param expr: input to einsum in default format
    :param shapes: shapes of tensors to be contracted
    :param unrolled: shapes of unrolled indices
    :param names: string labels for tensors used for more readable logging. The order of
                  names has to follow order of tensors as they appear in ``tn_to_contract``
    :param who: string id for logging identifying this optimal contraction path search
    """
    optimizer = kwargs.pop("optimizer", None)
    if optimizer in [None, "default", "dynamic-programming"]:
        optimizer = oe.DynamicProgramming(
            minimize="flops",  # 'size' optimize for largest intermediate tensor size, 'flops' for computation complexity
            search_outer=False,  # search through outer products as well
            cost_cap=True,  # don't use cost-capping strategy
        )

    # pre-process shapes, by dropping negative values (unrolled index) and last tuple,
    # which holds shapes of output tensor
    shapes_unrolled = tuple(tuple(x for x in s if x > 0) for s in shapes[:-1])
    path = kwargs.pop("path", None)
    kwargs.pop("shapes", False)
    if not path:
        path, path_info = oe.contract_path(
            expr, *shapes_unrolled, optimize=optimizer, shapes=True, **kwargs
        )  # ,use_blas=)

    path_info, mem_list = _get_contraction_path_info(
        path, expr, *shapes_unrolled, unrolled=unrolled, names=names, shapes=True
    )
    log.info(
        f"{who} optimizer {optimizer}"
        + (f" unrolled {unrolled}" if len(unrolled) > 0 else "")
        + f"\n{path}\n{path_info}\npeak-mem {max(mem_list):4.3e} mem {[f'{x:4.3e}' for x in mem_list]}"
    )
    return path, path_info


def _get_contraction_path_info(path, *operands, **kwargs):
    r"""opt_einsum contraction path reporting function extended
    to use user-supplied tensor labels ``names`` for description of individual operations.

    :param names: string labels for tensors used for more readable logging. The order of
                  names has to follow the order of tensors as they appear in ``operands``
    """
    names = kwargs.pop("names", None)
    unrolled = kwargs.pop("unrolled", ())

    unknown_kwargs = set(kwargs) - _VALID_CONTRACT_KWARGS
    if len(unknown_kwargs):
        raise TypeError(
            "einsum_path: Did not understand the following kwargs: {}".format(
                unknown_kwargs
            )
        )

    shapes = kwargs.pop("shapes", False)
    use_blas = kwargs.pop("use_blas", True)

    # Python side parsing
    input_subscripts, output_subscript, operands = oe.parser.parse_einsum_input(
        operands
    )

    # Build a few useful list and sets
    input_list = input_subscripts.split(",")
    if names:
        inputs_to_names = list(names)
    input_sets = [set(x) for x in input_list]
    if shapes:
        input_shps = operands
    else:
        input_shps = [x.shape for x in operands]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(",", ""))

    # Get length of each unique dimension and ensure all dimensions are correct
    size_dict = {}
    for tnum, term in enumerate(input_list):
        sh = input_shps[tnum]

        if len(sh) != len(term):
            raise ValueError(
                "Einstein sum subscript '{}' does not contain the "
                "correct number of indices for operand {}.".format(
                    input_list[tnum], tnum
                )
            )
        for cnum, char in enumerate(term):
            dim = int(sh[cnum])

            if char in size_dict:
                # For broadcasting cases we always want the largest dim size
                if size_dict[char] == 1:
                    size_dict[char] = dim
                elif dim not in (1, size_dict[char]):
                    raise ValueError(
                        "Size of label '{}' for operand {} ({}) does not match previous "
                        "terms ({}).".format(char, tnum, size_dict[char], dim)
                    )
            else:
                size_dict[char] = dim

    # Compute size of each input array plus the output array
    size_list = [
        oe.helpers.compute_size_by_dict(term, size_dict)
        for term in input_list + [output_subscript]
    ]

    num_ops = len(input_list)

    # Compute naive cost
    # This isnt quite right, need to look into exactly how einsum does this
    # indices_in_input = input_subscripts.replace(',', '')

    inner_product = (sum(len(x) for x in input_sets) - len(indices)) > 0
    naive_cost = oe.helpers.flop_count(indices, inner_product, num_ops, size_dict)

    cost_list = []
    scale_list = []
    size_list = [] # sizes of outputs
    contraction_list = []
    mem_list= []

    # Build contraction tuple (positions, gemm, einsum_str, remaining)
    for cnum, contract_inds in enumerate(path):
        # Make sure we remove inds from right to left
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))

        try:
            contract_tuple = oe.helpers.find_contraction(
                contract_inds, input_sets, output_set
            )
        except TypeError as e:
            # inputt_sets have to be frozenset sets for v3.4.0
            contract_tuple = oe.helpers.find_contraction(
                contract_inds, [frozenset(s) for s in input_sets], output_set
            )
        out_inds, input_sets, idx_removed, idx_contract = contract_tuple

        # Compute cost, scale, and size
        cost = oe.helpers.flop_count(
            idx_contract, idx_removed, len(contract_inds), size_dict
        )
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(oe.helpers.compute_size_by_dict(out_inds, size_dict))

        tmp_inputs = [input_list.pop(x) for x in contract_inds]
        if names:
            tmp_inds_to_names = [inputs_to_names.pop(x) for x in contract_inds]
        # make list
        input_shps= list(input_shps) if type(input_shps)==tuple else input_shps
        tmp_shapes = [input_shps.pop(x) for x in contract_inds]

        if use_blas:
            do_blas = oe.blas.can_blas(tmp_inputs, out_inds, idx_removed, tmp_shapes)
        else:
            do_blas = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            # use tensordot order to minimize transpositions
            all_input_inds = "".join(tmp_inputs)
            idx_result = "".join(sorted(out_inds, key=all_input_inds.find))

        shp_result = oe.parser.find_output_shape(tmp_inputs, tmp_shapes, idx_result)

        input_list.append(idx_result)
        if names:
            inputs_to_names.append(f"_TMP_{cnum}")
        input_shps.append(np.asarray(shp_result))

        # sum the currently contracted ops and remaining ops
        # Here, all shapes have to by np.ndarray
        mem_list.append(sum([x.prod() if hasattr(x,'prod') else np.asarray(x).prod() for x in tmp_shapes+input_shps]))

        einsum_str= ",".join(tmp_inputs) + "->" + idx_result
        if names:
            einsum_str = ",".join(tmp_inds_to_names) + "->" + inputs_to_names[-1]

        # for large expressions saving the remaining terms at each step can
        # incur a large memory footprint - and also be messy to print
        if len(input_list) <= 20:
            remaining = tuple(input_list)
        else:
            remaining = None

        contraction = (contract_inds, idx_removed, einsum_str, remaining, do_blas)
        contraction_list.append(contraction)

    opt_cost = sum(cost_list)

    # TODO associate input_subscripts with names
    path_print = PathInfo(
        contraction_list,
        input_subscripts,
        output_subscript,
        indices,
        path,
        scale_list,
        naive_cost,
        opt_cost,
        size_list,
        size_dict,
    )

    return path_print, mem_list


def contract_with_unroll_compute_constants(*args, **kwargs):
    r"""Extension of opt_einsum's contract allowing for index unrolling
    and use of checkpointing over unrolled loop.

    :param args: input to einsum in interleaved format. Explicit index labeling
                 of output is required
    :param unroll: indices to unroll
    :param checkpoint_unrolled:
    """
    verbosity = kwargs.get("verbosity", 0)
    checkpoint_on_device = kwargs.pop("checkpoint_on_device",False)
    if checkpoint_on_device in ['NONE','none','None',None]: checkpoint_on_device= False

    if checkpoint_on_device:
        # split args into tensors and index groups
        igs,ts= args[1::2], args[0 : 2 * (len(args) // 2) : 2]
        source_device= ts[0].device

        def _core_f(*ts):
            ts_moved= (x.to(device=checkpoint_on_device) for x in ts)
            args_moved= tuple(a for t_ig in zip(ts_moved,igs) for a in t_ig) + (args[-1],) if len(args)%2==1 else ()
            res= contract_with_unroll(*args_moved,**kwargs)
            return res.to(device=source_device)

        res= checkpoint(_core_f,*ts)
        if verbosity>0:
            log.info("After checkpointed contract_with_unroll_mode2\n"
                +_debug_allocated_tensors(device=checkpoint_on_device,totals_only=True)
            )
        return res

    who = kwargs.pop("who","unknown")
    verbosity = kwargs.pop("verbosity", 0)
    unroll = kwargs.pop("unroll", [])
    checkpoint_unrolled = kwargs.pop("checkpoint_unrolled", False)

    if not unroll or len(unroll) == 0:
        return oe.contract(*args, **kwargs)

    # We are unrolling. In general, there will be several constant
    # tensors among the individual unrolled calls.
    # Strategy is to build opt_einsum's ContractExpression, which makes use of these
    # constants
    #
    # Although contract supports interleaved format in general, in _gen_expression mode
    # the default subscript format is expected instead
    subscripts, shapes, unrolled_shapes = _preprocess_interleaved_to_expr_and_shapes(
        *args, unroll=unroll
    )

    # Get positions of tensor arguments which are constants wrt to unrolled contraction
    constants = tuple(
        idx for idx, ig in enumerate(args[1::2]) if not any([i in unroll for i in ig])
    )

    kwargs["_constants_dict"] = {
        i: args[0 : 2 * (len(args) // 2) : 2][i] for i in constants
    }

    # Build operands, passing tensors for constants and opt_einsum's Shaped (just shapes) for rest of the ops
    shapes_and_constant_ops = tuple(
        t if idx in constants else shape_only(tuple(i for i in shapes[idx] if i > 0))
        for idx, t in enumerate(args[0 : 2 * (len(args) // 2) : 2])
    )

    kwargs["_gen_expression"] = True
    oe_backend = kwargs.pop("backend", "auto")
    _contract_unroll_loop_body= oe.contract(
        subscripts, *shapes_and_constant_ops, **kwargs
    )
    def contract_unroll_loop_body(*args):
        return _contract_unroll_loop_body(*args, backend=oe_backend)
    if checkpoint_unrolled:
        # force evaluation of all constants
        _contract_unroll_loop_body.evaluate_constants(backend=oe_backend)
        _expr_const_ts= _contract_unroll_loop_body._evaluated_constants[oe_backend]
        
        def _contract_unroll_loop_body_checkpointed(*args):
            # reassign constants so the checkpointed evaluation preserves gradient flow
            count,j=0,-1
            while j>=-len(_expr_const_ts):
                if not (_expr_const_ts[j] is None):
                    count+=1
                    _expr_const_ts[j]= args[-count]
                j-=1

            return _contract_unroll_loop_body(*args[:-count], backend=oe_backend)

        def contract_unroll_loop_body(*args):
            # get handle of evaluated constant tensors
            c_args= tuple(t for t in _expr_const_ts if not (t is None))
            joint_args= args+c_args
            return checkpoint(_contract_unroll_loop_body_checkpointed, *joint_args)    

    # assign shape to each index label
    i_to_s = {
        i: s
        for ig, t in zip(args[1::2], args[0 : 2 * (len(args) // 2) : 2])
        for i, s in zip(ig, t.shape)
    }

    # index groups stripped of unrolled indices
    igs = tuple(
        tuple(i for i in ig if not i in unroll) for ig in (args[1::2] + (args[-1],))
    )

    # prepare tensor to accumulate individual contractions
    shape_out = tuple(i_to_s[i] for i in args[-1])
    ig_out_contracted_unrolled = tuple(i for i in unroll if not (i in args[-1]))
    partials = torch.empty(
        shape_out + tuple(i_to_s[i] for i in ig_out_contracted_unrolled),\
        device=args[0].device, dtype=args[0].dtype
    )

    if verbosity>0:
        log.info(who+" before unrolled loop\n"
            +_debug_allocated_tensors(device=args[0].device,totals_only=True))

    for ui_vals in product(*tuple(range(i_to_s[i]) for i in unroll)):
        ui_map = {u: v for u, v in zip(unroll, ui_vals)}

        ig_out = tuple(ui_map[i] if i in unroll else slice(None) for i in args[-1])
        ig_contracted_unrolled = tuple(ui_map[i] for i in unroll if not (i in args[-1]))

        # ops containing *only* variable tensors, narrowed by unrolled indices if applicable
        unrolled_ops = tuple(
            t[tuple(ui_map[i] if i in unroll else slice(None) for i in ig)]
            for t, ig in zip(args[0 : 2 * (len(args) // 2) : 2], args[1::2])
            if len([i for i in unroll if i in ig]) > 0
        )

        partials[ig_out + ig_contracted_unrolled]= contract_unroll_loop_body(
            *unrolled_ops
        )

        if verbosity>1:
            log.info(who+f" unrolled loop {ui_vals}\n"
                +_debug_allocated_tensors(device=args[0].device,totals_only=True))

    result = oe.contract(
        partials, tuple(args[-1]) + ig_out_contracted_unrolled, args[-1]
    )

    if verbosity>0:
        log.info(who+" unrolled loop concluded\n"
            +_debug_allocated_tensors(device=args[0].device,totals_only=False))

    return result

# IF checkpoint_on_device moves all ops to checkpoint on device
# does not use constant expressions in opt_einsum contract
def contract_with_unroll(*args, **kwargs):
    r"""Extension of opt_einsum's contract allowing for index unrolling
    and use of checkpointing over unrolled loop.

    :param args: input to einsum in interleaved format. Explicit index labeling
                 of output is required
    :param unroll: indices to unroll
    :param use_checkpoint:
    """
    verbosity = kwargs.get("verbosity", 0)
    checkpoint_on_device = kwargs.pop("checkpoint_on_device",False)
    if checkpoint_on_device in ['NONE','none','None',None]: checkpoint_on_device= False

    if checkpoint_on_device:
        # split args into tensors and index groups
        igs,ts= args[1::2], args[0 : 2 * (len(args) // 2) : 2]
        source_device= ts[0].device

        def _core_f(*ts):
            ts_moved= (x.to(device=checkpoint_on_device) for x in ts)
            args_moved= tuple(a for t_ig in zip(ts_moved,igs) for a in t_ig) + (args[-1],) if len(args)%2==1 else ()
            res= contract_with_unroll(*args_moved,**kwargs)
            return res.to(device=source_device)

        res= checkpoint(_core_f,*ts)
        if verbosity>0:
            log.info("After checkpointed contract_with_unroll_mode2\n"
                +_debug_allocated_tensors(device=checkpoint_on_device,totals_only=True)
            )
        return res

    who = kwargs.pop("who","unknown")
    verbosity = kwargs.pop("verbosity", 0)
    unroll = kwargs.pop("unroll", [])
    checkpoint_unrolled = kwargs.pop("checkpoint_unrolled", False)

    if not unroll or len(unroll) == 0:
        return oe.contract(*args, **kwargs)

    # We are unrolling. In general, there will be several constant
    # tensors among the individual unrolled calls.
    # Strategy is to build opt_einsum's ContractExpression, which makes use of these
    # constants
    #
    # Although contract supports interleaved format in general, in _gen_expression mode
    # the default subscript format is expected instead
    subscripts, shapes, unrolled_shapes = _preprocess_interleaved_to_expr_and_shapes(
        *args, unroll=unroll
    )

    # Get positions of tensor arguments which are constants wrt to unrolled contraction
    constants = tuple(
        idx for idx, ig in enumerate(args[1::2]) if not any([i in unroll for i in ig])
    )

    if not checkpoint_unrolled:
        kwargs["_constants_dict"] = {
            i: args[0 : 2 * (len(args) // 2) : 2][i] for i in constants
        }

        # Build operands, passing tensors for constants and opt_einsum's Shaped (just shapes) for rest of the ops
        shapes_and_constant_ops = tuple(
            t if idx in constants else shape_only(tuple(i for i in shapes[idx] if i > 0))
            for idx, t in enumerate(args[0 : 2 * (len(args) // 2) : 2])
        )
    else:
        # Build operands, passing opt_einsum's Shaped (just shapes) for both constants and rest of the ops
        shapes_and_constant_ops = tuple(
            shape_only(tuple(i for i in shapes[idx] if i > 0)) 
            for idx, t in enumerate(args[0 : 2 * (len(args) // 2) : 2])
        )

    kwargs["_gen_expression"] = True
    oe_backend = kwargs.pop("backend", "auto")
    _contract_unroll_loop_body= oe.contract(
        subscripts, *shapes_and_constant_ops, **kwargs
    )
    def contract_unroll_loop_body(*args):
        return _contract_unroll_loop_body(*args, backend=oe_backend)

    # narrowing of ops by unrolled indices is done within checkpointed section
    def contract_unroll_loop_body_checkpointed(unrolled_ops_slices,*args):
        unrolled_ops = tuple(
            t[s] for t, s in zip(args, unrolled_ops_slices)
        )
        return _contract_unroll_loop_body(*unrolled_ops, backend=oe_backend)

    # assign shape to each index label
    i_to_s = {
        i: s
        for ig, t in zip(args[1::2], args[0 : 2 * (len(args) // 2) : 2])
        for i, s in zip(ig, t.shape)
    }

    # index groups stripped of unrolled indices
    igs = tuple(
        tuple(i for i in ig if not i in unroll) for ig in (args[1::2] + (args[-1],))
    )

    # prepare tensor to accumulate individual contractions
    shape_out = tuple(i_to_s[i] for i in args[-1])
    ig_out_contracted_unrolled = tuple(i for i in unroll if not (i in args[-1]))
    partials = torch.empty(
        shape_out + tuple(i_to_s[i] for i in ig_out_contracted_unrolled),\
        device=args[0].device, dtype=args[0].dtype
    )

    if verbosity>0:
        log.info(who+" before unrolled loop\n"
            +_debug_allocated_tensors(device=args[0].device,totals_only=True))

    all_ops=args[0 : 2 * (len(args) // 2) : 2]
    for ui_vals in product(*tuple(range(i_to_s[i]) for i in unroll)):
        ui_map = {u: v for u, v in zip(unroll, ui_vals)}

        ig_out = tuple(ui_map[i] if i in unroll else slice(None) for i in args[-1])
        ig_contracted_unrolled = tuple(ui_map[i] for i in unroll if not (i in args[-1]))

        # narrowing is done before checkpointing
        # if checkpoint_unrolled:
        #     # narrowed ops containing all tensors
        #     unrolled_ops = tuple(
        #         t[tuple(ui_map[i] if i in unroll else slice(None) for i in ig)]
        #         for t, ig in zip(args[0 : 2 * (len(args) // 2) : 2], args[1::2])
        #     )
        #     partials[ig_out + ig_contracted_unrolled]= checkpoint(
        #         contract_unroll_loop_body, *unrolled_ops
        #     )
        if checkpoint_unrolled:
            # ops containing all tensors
            unrolled_ops_slices = tuple(
                tuple(ui_map[i] if i in unroll else slice(None) for i in ig)
                for ig in args[1::2]
            )

            partials[ig_out + ig_contracted_unrolled]= checkpoint(
                contract_unroll_loop_body_checkpointed, unrolled_ops_slices, *all_ops )
        else:
            # ops containing *only* variable tensors, narrowed by unrolled indices if applicable
            unrolled_ops = tuple(
                t[tuple(ui_map[i] if i in unroll else slice(None) for i in ig)]
                for t, ig in zip(args[0 : 2 * (len(args) // 2) : 2], args[1::2])
                if len([i for i in unroll if i in ig]) > 0
            )

            partials[ig_out + ig_contracted_unrolled]= contract_unroll_loop_body(
                *unrolled_ops
            )

        if verbosity>1:
            log.info(who+f" unrolled loop {ui_vals}\n"
                +_debug_allocated_tensors(device=args[0].device,totals_only=True))

    result = oe.contract(
        partials, tuple(args[-1]) + ig_out_contracted_unrolled, args[-1]
    )

    if verbosity>0:
        log.info(who+" unrolled loop concluded\n"
            +_debug_allocated_tensors(device=args[0].device,totals_only=False))

    return result

# IF checkpoint_on_device move only unrolled ops at each iteration to device
def HIGHER_PEAK_MEM_contract_with_unroll(*args, **kwargs):
    r"""Extension of opt_einsum's contract allowing for index unrolling
    and use of checkpointing over unrolled loop.

    :param args: input to einsum in interleaved format. Explicit index labeling
                 of output is required
    :param unroll: indices to unroll
    :param use_checkpoint:
    """
    verbosity = kwargs.get("verbosity", 0)
    checkpoint_on_device = kwargs.pop("checkpoint_on_device",False)
    if checkpoint_on_device in ['NONE','none','None',None]: checkpoint_on_device= False

    who = kwargs.pop("who","unknown")
    verbosity = kwargs.pop("verbosity", 0)
    unroll = kwargs.pop("unroll", [])
    checkpoint_unrolled = kwargs.pop("checkpoint_unrolled", False)

    if not unroll or len(unroll) == 0:
        return oe.contract(*args, **kwargs)

    # We are unrolling. In general, there will be several constant
    # tensors among the individual unrolled calls.
    # Strategy is to build opt_einsum's ContractExpression, which makes use of these
    # constants
    #
    # Although contract supports interleaved format in general, in _gen_expression mode
    # the default subscript format is expected instead
    subscripts, shapes, unrolled_shapes = _preprocess_interleaved_to_expr_and_shapes(
        *args, unroll=unroll
    )

    # Get positions of tensor arguments which are constants wrt to unrolled contraction
    constants = tuple(
        idx for idx, ig in enumerate(args[1::2]) if not any([i in unroll for i in ig])
    )

    if not checkpoint_unrolled:
        kwargs["_constants_dict"] = {
            i: args[0 : 2 * (len(args) // 2) : 2][i] for i in constants
        }

        # Build operands, passing tensors for constants and opt_einsum's Shaped (just shapes) for rest of the ops
        shapes_and_constant_ops = tuple(
            t if idx in constants else shape_only(tuple(i for i in shapes[idx] if i > 0))
            for idx, t in enumerate(args[0 : 2 * (len(args) // 2) : 2])
        )
    else:
        # Build operands, passing opt_einsum's Shaped (just shapes) for both constants and rest of the ops
        shapes_and_constant_ops = tuple(
            shape_only(tuple(i for i in shapes[idx] if i > 0)) 
            for idx, t in enumerate(args[0 : 2 * (len(args) // 2) : 2])
        )

    kwargs["_gen_expression"] = True
    oe_backend = kwargs.pop("backend", "auto")
    _contract_unroll_loop_body= oe.contract(
        subscripts, *shapes_and_constant_ops, **kwargs
    )
    def contract_unroll_loop_body(*args):
        return _contract_unroll_loop_body(*args, backend=oe_backend)

    def contract_unroll_loop_body_2(*args):
        source_device=args[0].device
        if checkpoint_on_device:
            args_moved= (a.to(device=checkpoint_on_device) for a in args)
        res=_contract_unroll_loop_body(*args_moved, backend=oe_backend)
        return res.to(device=source_device)

    # assign shape to each index label
    i_to_s = {
        i: s
        for ig, t in zip(args[1::2], args[0 : 2 * (len(args) // 2) : 2])
        for i, s in zip(ig, t.shape)
    }

    # index groups stripped of unrolled indices
    igs = tuple(
        tuple(i for i in ig if not i in unroll) for ig in (args[1::2] + (args[-1],))
    )

    # prepare tensor to accumulate individual contractions
    shape_out = tuple(i_to_s[i] for i in args[-1])
    ig_out_contracted_unrolled = tuple(i for i in unroll if not (i in args[-1]))
    partials = torch.empty(
        shape_out + tuple(i_to_s[i] for i in ig_out_contracted_unrolled),\
        device=args[0].device, dtype=args[0].dtype
    )

    if verbosity>0:
        log.info(who+" before unrolled loop\n"
            +_debug_allocated_tensors(device=args[0].device,totals_only=True))

    for ui_vals in product(*tuple(range(i_to_s[i]) for i in unroll)):
        ui_map = {u: v for u, v in zip(unroll, ui_vals)}

        ig_out = tuple(ui_map[i] if i in unroll else slice(None) for i in args[-1])
        ig_contracted_unrolled = tuple(ui_map[i] for i in unroll if not (i in args[-1]))

        if checkpoint_unrolled:
            # ops containing all tensors, narrowed by unrolled indices if applicable
            unrolled_ops = tuple(
                t[tuple(ui_map[i] if i in unroll else slice(None) for i in ig)]
                for t, ig in zip(args[0 : 2 * (len(args) // 2) : 2], args[1::2])
            )

            partials[ig_out + ig_contracted_unrolled]= checkpoint(
                contract_unroll_loop_body_2, *unrolled_ops )
        else:
            # ops containing *only* variable tensors, narrowed by unrolled indices if applicable
            unrolled_ops = tuple(
                t[tuple(ui_map[i] if i in unroll else slice(None) for i in ig)]
                for t, ig in zip(args[0 : 2 * (len(args) // 2) : 2], args[1::2])
                if len([i for i in unroll if i in ig]) > 0
            )

            partials[ig_out + ig_contracted_unrolled]= contract_unroll_loop_body(
                *unrolled_ops
            )

        if verbosity>1:
            log.info(who+f" unrolled loop {ui_vals}\n"
                +_debug_allocated_tensors(device=args[0].device,totals_only=True))

    result = oe.contract(
        partials, tuple(args[-1]) + ig_out_contracted_unrolled, args[-1]
    )

    if verbosity>0:
        log.info(who+" unrolled loop concluded\n"
            +_debug_allocated_tensors(device=args[0].device,totals_only=False))

    return result


def contract_with_unroll_legacy(*args, **kwargs):
    r"""Extension of opt_einsum's contract allowing for index unrolling
    and use of checkpointing over unrolled loop.

    :param args: input to einsum in interleaved format. Explicit index labeling
                 of output is required
    :param unroll: indices to unroll
    :param use_checkpoint:
    """
    unroll = kwargs.pop("unroll", [])
    use_checkpoint = kwargs.pop("use_checkpoint", False)

    if len(unroll) == 0:
        return oe.contract(*args, **kwargs)

    # We are unrolling. In general, there will be several constant
    # tensors among the individual unrolled calls.

    # assign shape to each index label
    i_to_s = {
        i: s
        for ig, t in zip(args[1::2], args[0 : 2 * (len(args) // 2) : 2])
        for i, s in zip(ig, t.shape)
    }

    # index groups stripped of unrolled indices
    igs = tuple(
        tuple(i for i in ig if not i in unroll) for ig in (args[1::2] + (args[-1],))
    )

    # prepare tensor to accumulate individual contractions
    shape_out = tuple(i_to_s[i] for i in args[-1])
    ig_out_contracted_unrolled = tuple(i for i in unroll if not (i in args[-1]))
    partials = torch.empty(
        shape_out + tuple(i_to_s[i] for i in ig_out_contracted_unrolled),\
        device=args[0].device, dtype=args[0].dtype
    )

    for ui_vals in product(*tuple(range(i_to_s[i]) for i in unroll)):
        ui_map = {u: v for u, v in zip(unroll, ui_vals)}

        # ops narrowed by unrolled indices if applicable
        ops = tuple(
            t
            if len([i for i in unroll if i in ig]) == 0
            else t[tuple(ui_map[i] if i in unroll else slice(None) for i in ig)]
            for t, ig in zip(args[0 : 2 * (len(args) // 2) : 2], args[1::2])
        )

        unrolled_args = tuple(x for o_ig in zip(ops, igs[:-1]) for x in o_ig) + (
            igs[-1],
        )

        ig_out = tuple(ui_map[i] if i in unroll else slice(None) for i in args[-1])
        ig_contracted_unrolled = tuple(ui_map[i] for i in unroll if not (i in args[-1]))

        partials[ig_out + ig_contracted_unrolled]= oe.contract(*unrolled_args, **kwargs)
        

    result = torch.einsum(
        partials, tuple(args[-1]) + ig_out_contracted_unrolled, args[-1]
    )
    return result