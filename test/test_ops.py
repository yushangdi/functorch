# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from torch.testing._internal.common_utils import TestCase, run_tests, is_iterable_of_tensors
import torch
from torch import Tensor
import functools
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_device_type import \
     toleranceOverride, tol
from functorch_lagging_op_db import functorch_lagging_op_db
from functorch_additional_op_db import additional_op_db
from common_utils import (
    get_fallback_and_vmap_exhaustive,
    get_exhaustive_batched_inputs,
    xfail,
    skip,
    skipOps,
    tol1,
    # tol2,
    opsToleranceOverride,
    check_vmap_fallback,
)
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from functorch import grad, vjp, vmap, jacrev, jacfwd
import torch.autograd.forward_ad as fwAD
from functorch._src.eager_transforms import _as_tuple, jvp
aten = torch.ops.aten

# Version of autograd.grad that handles outputs that don't depend on inputs


def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True):
    inputs, inputs_spec = tree_flatten(inputs)
    result = [torch.zeros_like(inp) for inp in inputs]
    diff_argnums = tuple(i for i, inp in enumerate(inputs) if inp.requires_grad)
    inputs = tuple(inputs[i] for i in diff_argnums)
    if grad_outputs is None:
        diff_outputs = tuple(out for out in outputs if out.requires_grad)
    else:
        something = [(out, go) for out, go in zip(outputs, grad_outputs)
                     if out.requires_grad]
        if len(something) == 0:
            diff_outputs, grad_outputs = (), ()
        else:
            diff_outputs, grad_outputs = zip(*something)
    if len(diff_outputs) == 0:
        return tuple(torch.zeros_like(inp) for inp in inputs)
    grad_inputs = torch.autograd.grad(diff_outputs, inputs, grad_outputs,
                                      retain_graph=retain_graph,
                                      create_graph=create_graph,
                                      allow_unused=True)
    grad_inputs = tuple(torch.zeros_like(inp) if gi is None else gi
                        for gi, inp in zip(grad_inputs, inputs))
    for idx, grad_inp in zip(diff_argnums, grad_inputs):
        result[idx] = grad_inp
    return tree_unflatten(result, inputs_spec)


def diff_arg(arg, requires_grad=True):
    def is_differentiable_arg(arg):
        if requires_grad:
            return arg.requires_grad
        else:
            return arg.is_floating_point() or arg.is_complex()
    if is_iterable_of_tensors(arg):
        if all([is_differentiable_arg(a) for a in arg]):
            return True
        if all([not is_differentiable_arg(a) for a in arg]):
            return False
        raise RuntimeError("NYI: The test runner can't handle this")
    return isinstance(arg, Tensor) and is_differentiable_arg(arg)


# Given f, returns an f' such that:
# - f' takes only positional arguments
# - All arguments to f' are floating-point Tensors
# - All outputs of f' are floating-point Tensors
def normalize_op_input_output2(f, args, kwargs, output_process_fn_grad=None, requires_grad=True):
    flat_args, args_spec = tree_flatten(args)
    diff_argnums = tuple(i for i, arg in enumerate(flat_args) if diff_arg(arg, requires_grad=requires_grad))
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            # TODO: Remove the following hack for namedtuples
            result = tuple(result)
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result
    return wrapped, primals


# TODO: consolidate with normalize_op_input_output2
def normalize_op_input_output3(f, args, kwargs, sample_args, output_process_fn_grad=None):
    flat_args, args_spec = tree_flatten(args)
    flat_sample_args, _ = tree_flatten(sample_args)
    diff_argnums = tuple(i for i, (arg, sample) in enumerate(zip(flat_args, flat_sample_args))
                         if diff_arg(sample, requires_grad=True))
    assert len(diff_argnums) > 0
    primals = tuple(flat_args[i] for i in diff_argnums)

    @functools.wraps(f)
    def wrapped(*primals):
        _args = list(flat_args)
        for num, arg in zip(diff_argnums, primals):
            _args[num] = arg
        _args = tree_unflatten(_args, args_spec)
        result = f(*_args, **kwargs)
        if output_process_fn_grad is not None:
            result = output_process_fn_grad(result)
        if isinstance(result, tuple):
            # TODO: Remove the following hack for namedtuples
            result = tuple(result)
            result = tuple(r for r in result if torch.is_floating_point(r))
            assert len(result) > 0
        return result
    return wrapped, primals


def normalize_op_input_output(f, sample, requires_grad=True):
    args = tuple([sample.input] + list(sample.args))
    return normalize_op_input_output2(
        f, args, sample.kwargs, sample.output_process_fn_grad, requires_grad=requires_grad
    )


def ref_vjp(f, *primals):
    result = f(*primals)

    def wrapped(cotangents):
        return _autograd_grad(_as_tuple(result), primals, _as_tuple(cotangents))

    return result, wrapped


def simulate_jvp(f, primals, tangents):
    primals_out, tangents_out = torch.autograd.functional.jvp(f, primals, tangents)
    return primals_out, tangents_out


def ref_jvp(f, primals, tangents):
    with fwAD.dual_level():
        duals = tuple(fwAD.make_dual(p, t) for p, t in zip(primals, tangents))
        result_duals = f(*duals)
        result_duals, spec = tree_flatten(result_duals)
        primals_out, tangents_out = zip(*(fwAD.unpack_dual(d) for d in result_duals))
        return tree_unflatten(primals_out, spec), tree_unflatten(tangents_out, spec)


def get_sample_cotangents(f, sample):
    fn, primals = normalize_op_input_output(f, sample)
    output = fn(*primals)
    return tree_map(torch.randn_like, output)


# returns a new function g(*args, *cotangents)
# that computes vjps and (*args, cotangents)
def get_vjp_fn_and_args_with_cotangents(f, sample, cotangents):
    args = tuple([sample.input] + list(sample.args))
    kwargs = sample.kwargs
    flat_args, args_spec = tree_flatten(args)
    flat_cotangents, cotangents_spec = tree_flatten(cotangents)

    @functools.wraps(f)
    def wrapped(*args):
        assert len(args) == len(flat_args) + len(flat_cotangents)
        actual_args = args[:len(flat_args)]
        cotangents = args[len(flat_args):]
        actual_args = tree_unflatten(actual_args, args_spec)
        cotangents = tree_unflatten(cotangents, cotangents_spec)

        fn, primals = normalize_op_input_output3(f, actual_args, kwargs,
                                                 flat_args,
                                                 sample.output_process_fn_grad)
        _, vjp_fn = vjp(fn, *primals)
        return vjp_fn(cotangents)

    return wrapped, tuple(flat_args + flat_cotangents)


# returns a new function g(*args, *cotangents)
# that computes vjps and (*args, cotangents) using torch.autograd.grad
def get_autograd_fn_and_args_with_cotangents(f, sample, cotangents):
    args = tuple([sample.input] + list(sample.args))
    kwargs = sample.kwargs
    flat_args, args_spec = tree_flatten(args)
    flat_cotangents, cotangents_spec = tree_flatten(cotangents)

    @functools.wraps(f)
    def wrapped(*args):
        assert len(args) == len(flat_args) + len(flat_cotangents)
        actual_args = args[:len(flat_args)]
        cotangents = args[len(flat_args):]
        actual_args = tree_unflatten(actual_args, args_spec)
        cotangents = tree_unflatten(cotangents, cotangents_spec)

        fn, primals = normalize_op_input_output3(f, actual_args, kwargs,
                                                 flat_args,
                                                 sample.output_process_fn_grad)
        out = fn(*primals)
        diff_wrt = tuple(primal for primal in primals if (primal.requires_grad or primal.grad_fn is not None))
        if diff_wrt:
            return torch.autograd.grad(out, diff_wrt, grad_outputs=cotangents)
        else:
            return (torch.ones(()),)  # uuugh hack...this will need to be more generic

    return wrapped, tuple(flat_args + flat_cotangents)


# Returns a new function g(*args, *cotangents) that computes vjps and
# sample (*args, *cotangents)
def get_vjpfull_variant(f, sample):
    fn, primals = normalize_op_input_output(f, sample)
    result = fn(*primals)
    cotangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x, requires_grad=True), result))
    num_primals = len(primals)
    args = (*primals, *cotangents)

    @functools.wraps(f)
    def wrapped(*args):
        primals = args[:num_primals]
        cotangents = args[num_primals:]
        result, vjp_fn = vjp(fn, *primals)
        if isinstance(result, torch.Tensor):
            assert len(cotangents) == 1
            cotangents = cotangents[0]
        return vjp_fn(cotangents)

    return wrapped, args


def get_jvp_variant(f, sample):
    # We want this higher-order variant of jvp, so that it can
    # be used to wrap vmap
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    tangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x), primals))

    @functools.wraps(f)
    def wrapped(*args):
        tangents = args
        primals_out, tangents_out = jvp(fn, primals, tangents)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out, _ = tree_flatten(primals_out)
            flat_tangents_out, _ = tree_flatten(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, tangents


def get_jvp_variant_primals_tangents(f, sample):
    # We want this higher-order variant of jvp, so that it can
    # be used to wrap vmap
    fn, primals = normalize_op_input_output(f, sample, requires_grad=False)
    tangents = _as_tuple(
        tree_map(lambda x: torch.randn_like(x), primals))

    @functools.wraps(f)
    def wrapped(*args):
        primals_in = args[:len(primals)]
        tangents_in = args[len(primals):]
        primals_out, tangents_out = jvp(fn, primals_in, tangents_in)

        if isinstance(primals_out, torch.Tensor):
            return (primals_out, tangents_out)
        else:
            flat_primals_out, _ = tree_flatten(primals_out)
            flat_tangents_out, _ = tree_flatten(tangents_out)
            return tuple(flat_primals_out + flat_tangents_out)

    return wrapped, primals + tangents


def is_inplace(op, variant):
    if hasattr(variant, "__wrapped__"):
        return variant.__wrapped__ is op.get_inplace()
    return variant is op.get_inplace()


vjp_fail = {
    skip('nn.functional.dropout'),  # randomness testing artifact
    skip('nn.functional.rrelu'),  # randomness testing artifact
    skip('bernoulli'),  # randomness testing artifact
    skip('normal', ''),  # randomness testing artifact
    skip('normal', 'number_mean'),  # randomness testing artifact
    xfail('tensor_split'),
    xfail('to_sparse'),
    xfail('nn.functional.ctc_loss'),
    skip('nn.functional.feature_alpha_dropout', 'with_train'),  # fails on cuda, runs okay on cpu
    skip('nn.functional.feature_alpha_dropout', 'without_train'),  # fails on cuda, runs okay on cpu
    skip('pca_lowrank', ''),  # fails on cuda, runs okay on cpu
    skip('svd_lowrank', ''),  # fails on cuda, runs okay on cpu
    skip('nn.functional.dropout2d', ''),  # fails on cuda, runs okay on cpu
}


class TestOperators(TestCase):
    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_grad', vjp_fail.union({
        skip('nn.functional.fractional_max_pool2d'),  # fails on cuda, runs okay on cpu
        skip('nn.functional.fractional_max_pool3d'),  # fails on cuda, runs okay on cpu
    }))
    @opsToleranceOverride('TestOperators', 'test_grad', (
        tol1('nn.functional.binary_cross_entropy_with_logits',
             {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
    ))
    def test_grad(self, device, dtype, op):
        if op.name in vjp_fail:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            diff_argnums = tuple(i for i, arg in enumerate(args) if diff_arg(arg))
            assert len(diff_argnums) > 0
            diff_args = tuple(args[i] for i in diff_argnums)

            def wrapped_fn(*args, **kwargs):
                result = op(*args, **kwargs)
                if sample.output_process_fn_grad is not None:
                    result = sample.output_process_fn_grad(result)

                # Reduce into single value for grad
                if isinstance(result, torch.Tensor):
                    return result.sum()
                result = sum([res.sum() for res in result])
                return result

            result = grad(wrapped_fn, diff_argnums)(*args, **kwargs)
            expected = _autograd_grad(_as_tuple(wrapped_fn(*args, **kwargs)), diff_args)

            self.assertEqual(result, expected)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_jvp', set({
        skip('nn.functional.dropout'),  # randomness testing artifact; not actually a problem
        skip('nn.functional.rrelu'),  # randomness testing artifact; not actually a problem
        skip('nn.functional.fractional_max_pool2d'),  # fails on cuda, runs okay on cpu
        skip('nn.functional.fractional_max_pool3d'),  # fails on cuda, runs okay on cpu
        skip('nn.functional.max_pool1d'),  # fails on cpu, runs okay on cuda
        skip('nn.functional.feature_alpha_dropout', 'with_train'),  # fails on cuda, runs okay on cpu
        skip('nn.functional.feature_alpha_dropout', 'without_train'),  # fails on cuda, runs okay on cpu
        skip('pca_lowrank', ''),  # fails on cuda, runs okay on cpu
        skip('svd_lowrank', ''),  # fails on cuda, runs okay on cpu
        skip('nn.functional.dropout2d', ''),  # fails on cuda, runs okay on cpu

        # The following don't have a forward-mode AD formula in PyTorch core
        # (check derivatives.yaml).
        xfail('var_mean'),
        xfail('std_mean'),

        # =============================================
        # NB: The above failures also fail using PyTorch core's
        #     forward-mode AD and vmap.
        #     The failures below are functorch-specific issues
        # =============================================

        # Composite ops that do bad things. Need to be fixed in PyTorch core.
        # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
        xfail('tensor_split'),

        skip('bernoulli'),  # cuda set seed randomness issues

        # BUG: runs and produces numerical differences
        skip('nn.functional.max_unpool1d'),  # fails everywhere except on mac
        skip('nn.functional.max_unpool2d'),  # fails everywhere except on windows
        xfail('nn.functional.max_unpool3d'),
    }))
    @opsToleranceOverride('TestOperators', 'test_jvp', (
        tol1('nn.functional.conv_transpose3d',
             {torch.float32: tol(atol=1e-04, rtol=1.3e-06)}, device_type='cuda'),
        tol1('nn.functional.binary_cross_entropy_with_logits',
             {torch.float32: tol(atol=4e-04, rtol=4e-04)}),
    ))
    def test_jvp(self, device, dtype, op):
        # TODO: when we change supports_autograd to supports_backward_ad, also change in this file
        VJP_DECOMP = {
            'nn.functional.logsigmoid',
        }
        if op.name in VJP_DECOMP:
            ref_jvp_local = simulate_jvp
        else:
            ref_jvp_local = ref_jvp

        if not op.supports_forward_ad and op.name not in VJP_DECOMP:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            # NB: we used requires_grad=True to determine where the primals are,
            # but don't need that information otherwise
            fn, primals = normalize_op_input_output(op, sample, requires_grad=True)
            primals = tree_map(lambda x: x.detach(), primals)
            tangents = tree_map(lambda x: torch.randn_like(x), primals)
            primal_outs, tangent_outs = jvp(fn, primals, tangents)
            expected_primal_outs, expected_tangent_outs = ref_jvp_local(fn, primals, tangents)
            self.assertEqual(primal_outs, expected_primal_outs)
            self.assertEqual(tangent_outs, expected_tangent_outs)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjp', vjp_fail.union({
        skip('nn.functional.fractional_max_pool2d'),  # fails on cpu, runs okay on cuda
        skip('nn.functional.fractional_max_pool3d'),  # fails on cpu, runs okay on cuda
        xfail('nn.functional.feature_alpha_dropout', 'with_train'),
        xfail('pca_lowrank', ''),
        xfail('nn.functional.dropout2d', ''),
        xfail('nn.functional.feature_alpha_dropout', 'without_train'),
        xfail('svd_lowrank', ''),
    }))
    @opsToleranceOverride('TestOperators', 'test_vjp', (
        tol1('nn.functional.conv_transpose3d',
             {torch.float32: tol(atol=5e-05, rtol=9e-05)}, device_type='cuda'),
        tol1('nn.functional.binary_cross_entropy_with_logits',
             {torch.float32: tol(atol=1e-04, rtol=1e-04)}),
    ))
    def test_vjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        def _test(_op):
            for sample in samples:
                fn, primals = normalize_op_input_output(_op, sample)
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                out, vjp_fn = vjp(fn, *primals)
                self.assertEqual(out, result)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

        _test(op)
        for a_op in op.aliases:
            _test(a_op)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjpvjp', vjp_fail.union({
        skip('nn.functional.max_unpool1d'),  # Flaky
        skip('nn.functional.max_unpool2d'),  # Flaky
        skip('nn.functional.fractional_max_pool2d'), # randomness
        skip('nn.functional.fractional_max_pool3d'), # randomness
    }))
    @opsToleranceOverride('TestOperators', 'test_vjpvjp', (
        tol1('nn.functional.conv_transpose3d',
             {torch.float32: tol(atol=5e-05, rtol=9e-05)}, device_type='cuda'),
    ))
    def test_vjpvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, args = get_vjpfull_variant(op, sample)
            result = fn(*args)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            # Compute vjp of vjp
            _, vjp_fn = vjp(fn, *args)
            result_vjps = vjp_fn(cotangents)

            # Compute ref_vjp of vjp. We could have done ref_vjp of ref_vjp,
            # but since we're confident that vjp works by itself, this is
            # an equivalent way to test that.
            _, vjp_fn = ref_vjp(fn, *args)
            expected_vjps = vjp_fn(cotangents)

            self.assertEqual(result_vjps, expected_vjps)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    def test_vmapvjpvjp(self, device, dtype, op):
        self.skipTest("Skipped; these tests take too long")
        op_skip = set({
        })
        op_skip = op_skip.union(vjp_fail)
        if op.name in op_skip:
            self.skipTest("Skipped; Expected failures")
            return

        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, args = get_vjpfull_variant(op, sample)
            result = fn(*args)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)
            cotangents, _ = tree_flatten(cotangents)
            num_args = len(args)

            args_and_cotangents = tuple(args) + tuple(cotangents)

            def vjp_of_vjp(*args_and_cotangents):
                args = args_and_cotangents[:num_args]
                cotangents = args_and_cotangents[num_args:]
                result, vjp_fn = vjp(fn, *args)
                result_vjps = vjp_fn(cotangents)
                result, _ = tree_flatten(result)
                result_vjps, _ = tree_flatten(result_vjps)
                return (*result, *result_vjps)

            generator = get_fallback_and_vmap_exhaustive(vjp_of_vjp, args_and_cotangents, {}, opinfo=op)
            for loop_out, batched_out in generator:
                self.assertEqual(loop_out, batched_out)

    vmapvjp_fail = vjp_fail.union({
        # The following are not bugs and are expected behavior
        xfail('masked_select'),  # Not possible due to dynamic shapes
        skip('bernoulli'),  # randomness
        skip('normal', ''),  # randomness
        skip('normal', 'number_mean'),  # randomness
        xfail('nn.functional.dropout'),  # randomness
        xfail('as_strided'),  # as_strided is too wild for us to support, wontfix
        xfail('index_put', ''),  # not possible due to dynamic shapes; we support a subset
        xfail('masked_scatter'),  # dynamic
        xfail('nn.functional.fractional_max_pool2d'),  # random
        xfail('nn.functional.fractional_max_pool3d'),  # random
        xfail('take'),  # dynamic

        # All of the following are bugs and need to be fixed
        skip('linalg.svdvals'),  # # really annoying thing where it passes correctness check but not has_batch_rule
        xfail('__getitem__', ''),
        xfail('_masked.prod'),  # calls aten::item
        xfail('eig'),  # calls aten::item
        xfail('linalg.det', ''),  # calls .item()
        xfail('linalg.eig'),  # Uses aten::allclose
        xfail('linalg.eigh'),  # needs diag_scatter
        xfail('linalg.householder_product'),  # needs select_scatter
        xfail('linalg.slogdet'),  # calls .item()
        xfail('logdet'),  # calls .item()
        xfail('matrix_exp'),  # would benefit from narrow_scatter
        xfail('nanquantile'),  # checks q via a .item() call
        xfail('nn.functional.gaussian_nll_loss'),  # checks var for if any value < 0
        xfail('prod'),  # calls nonzero
        xfail('put'),
        xfail('quantile'),  # checks q via a .item() call
        xfail('stft'),
        xfail('symeig'),  # would benefit from diag_scatter
        xfail('view_as_complex'),

        # required rank 4 tensor to use channels_last format
        xfail('bfloat16'),
        xfail('double'),
        xfail('float'),
        xfail('half'),

        xfail('scatter_reduce', 'prod'),  # item call

        # NYI: querying is_contiguous inside of vmap for memory_format other than torch.contiguous_format
        xfail('nn.functional.max_unpool2d'),
        xfail('nn.functional.max_unpool2d', 'grad'),
    })

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @skipOps('TestOperators', 'test_vmapvjp', vmapvjp_fail)
    def test_vmapvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            cotangents = get_sample_cotangents(op, sample)
            fn, args = get_vjp_fn_and_args_with_cotangents(op, sample, cotangents)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(fn, args, {}, opinfo=op):
                self.assertEqual(loop_out, batched_out)

    # There are several variations we care about
    # 1) primal batched (TODO)
    # 2) tangent batched (batched grads) <--
    # 3) both batched (TODO)
    # The below tests (2) only.
    @ops(functorch_lagging_op_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @skipOps('TestOperators', 'test_vmapjvp', {
        skip('nn.functional.dropout'),  # randomness
        skip('nn.functional.rrelu'),  # randomness
        skip('nn.functional.fractional_max_pool2d'),  # randomness
        skip('nn.functional.fractional_max_pool3d'),  # randomness
        skip('bernoulli', ''),  # randomness
        skip('nn.functional.max_pool1d'),  # fails on cpu, runs on cuda

        # TODO: fails in core due to in-place batched nto non-batched
        # but fails here for a different reason
        xfail('linalg.householder_product'),

        # Try to in-place batched tensor into non-batched tensor
        xfail('matrix_exp'),

        # Apprently these support forward AD, but we get "Trying to use forward AD..."
        # These are cases where OpInfo has supports_forward_ad=True, but disables
        # the test
        xfail('var_mean'),
        xfail('std_mean'),

        # RuntimeError: expand: the number of sizes provided (1) must be greater or
        # equal to the number of dimensions in the tensor (2)
        xfail('nanquantile'),
        xfail('quantile'),

        # Not implemented
        xfail('scatter'),

        # =============================================
        # NB: The above failures also fail in PyTorch core.
        #     The failures below only fail in functorch
        # =============================================

        # Composite ops that do bad things. Need to be fixed in PyTorch core.
        # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
        xfail('tensor_split'),

        # Causing multiple forward mode AD issues, needs investigation
        xfail('nn.functional.batch_norm'),
        xfail('nn.functional.batch_norm', 'without_cudnn', device_type='cuda'),

        skip('nn.functional.feature_alpha_dropout', 'with_train'),
        skip('pca_lowrank', ''),
        skip('nn.functional.dropout2d', ''),
        skip('nn.functional.feature_alpha_dropout', 'without_train'),
        skip('svd_lowrank', ''),
        xfail('nn.functional.soft_margin_loss', ''),
        xfail('stft'),  # something weird is happening with shapes

        xfail('double'),  # required rank 4 tensor to use channels_last format

        # BUG: runs and produces numerical differences
        xfail('nn.functional.max_unpool1d', device_type='cpu'),
        xfail('nn.functional.max_unpool2d'),
        xfail('nn.functional.max_unpool3d'),

        xfail('nn.functional.embedding'),  # embedding_renorm_ does not support fwd AD
        xfail('put'),  # calls put_ during vmap with only vmaps over other, not self
    })
    def test_vmapjvp(self, device, dtype, op):
        if is_inplace(op, op.get_op()):
            # TODO: test in-place
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=False)

        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        for sample in samples:
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            args = tuple([*arg_values, *kwarg_values])
            fn, args = get_jvp_variant(op, sample)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(fn, args, {}, opinfo=op, bdims=(0,)):
                self.assertEqual(loop_out, batched_out)

    vmapjvpall_fail = {
        # The following are expected (not a bug)
        skip('bernoulli', ''),  # randomness
        skip('nn.functional.dropout'),  # randomness
        skip('nn.functional.rrelu'),  # randomness
        skip('nn.functional.dropout2d', ''),
        skip('nn.functional.feature_alpha_dropout', 'without_train'),
        skip('nn.functional.feature_alpha_dropout', 'with_train'),
        xfail('nn.functional.fractional_max_pool2d'),  # Cannot access data pointer of Tensor that doesn't have storage
        xfail('nn.functional.fractional_max_pool3d'),  # Cannot access data pointer of Tensor that doesn't have storage

        # The following are bugs that we should fix
        skip('nn.functional.max_pool1d'),  # fails on cpu, runs on cuda
        xfail('nn.functional.batch_norm', device_type='cuda'),
        xfail('nn.functional.batch_norm', 'without_cudnn', device_type='cuda'),
        xfail('nn.functional.hinge_embedding_loss', device_type='cuda'),
        xfail('_masked.mean'),
        xfail('_masked.prod'),

        # Causing issues with multiple cpu levels of forward mode AD
        xfail('nn.functional.batch_norm', device_type='cpu'),
        xfail('nn.functional.hinge_embedding_loss', device_type='cpu'),

        xfail('nn.functional.soft_margin_loss', ''),
        xfail('nn.functional.binary_cross_entropy_with_logits', ''),
        xfail('linalg.householder_product'),
        xfail('tensor_split'),
        xfail('quantile'),
        xfail('var_mean'),
        xfail('as_strided'),
        xfail('nn.functional.gaussian_nll_loss'),
        xfail('std_mean'),
        xfail('scatter'),
        xfail('matrix_exp'),
        xfail('nanquantile'),
        xfail('view_as_complex'),
        xfail('prod'),

        skip('pca_lowrank', ''),
        skip('svd_lowrank', ''),

        xfail('stft'),  # transpose_ fallback

        xfail('double'),  # required rank 4 tensor to use channels_last format

        skip('nn.functional.max_unpool1d'),  # Flaky, seems to sometimes his max_unpool2d
        # BUG: runs and produces numerical differences
        xfail('nn.functional.max_unpool2d'),
        xfail('nn.functional.max_unpool3d'),

        xfail('nn.functional.embedding'),  # embedding_renorm_ does not support fwd AD
        xfail('put'),  # calls put_ during vmap with only vmaps over other, not self
        xfail('nn.functional.prelu'),  # Call Tensor.as_strided
    }

    @ops(functorch_lagging_op_db, allowed_dtypes=(torch.float,))
    @opsToleranceOverride('TestOperators', 'test_vmapjvpall', (
        tol1('nn.functional.conv_transpose3d',
             {torch.float32: tol(atol=2e-04, rtol=9e-3)}, device_type='cuda'),
    ))
    @skipOps('TestOperators', 'test_vmapjvpall', vmapjvpall_fail)
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    # This is technically a superset of test_vmapjvp. We should either delete test_vmapjvp
    # or figure out if we can split vmapjvpall. It's useful to keep test_vmapjvp intact
    # because that coresponds to "batched forward-mode AD" testing in PyTorch core
    def test_vmapjvpall(self, device, dtype, op):
        if is_inplace(op, op.get_op()):
            # TODO: test in-place
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=False)

        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        for sample in samples:
            arg_values = [sample.input] + list(sample.args)
            kwarg_values = sample.kwargs
            args = tuple([*arg_values, *kwarg_values])
            fn, args = get_jvp_variant_primals_tangents(op, sample)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(fn, args, {}, opinfo=op):
                self.assertEqual(loop_out, batched_out)

    @ops(functorch_lagging_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vmapjvpall_has_batch_rule', vmapjvpall_fail.union({
        xfail('linalg.solve_triangular'),
        xfail('nn.functional.huber_loss'),
        xfail('nn.functional.poisson_nll_loss'),
        xfail('lu'),
        xfail('cumprod'),
        xfail('lu_solve'),
        xfail('linalg.lstsq', 'grad_oriented'),
        xfail('linalg.cholesky'),
        xfail('linalg.qr'),
        xfail('cross'),
        xfail('qr'),
        xfail('linalg.pinv'),
        xfail('masked_fill'),
        xfail('copysign'),
        xfail('linalg.solve'),
        xfail('linalg.eig'),
        xfail('complex'),
        xfail('linalg.pinv', 'hermitian'),
        xfail('pinverse'),
        skip('_masked.mean'),  # ???
        xfail('linalg.cholesky_ex'),
        xfail('masked_scatter'),
        xfail('index_fill'),
        xfail('take'),
        xfail('linalg.eigvals'),
        xfail('linalg.qr'),
        xfail('linalg.tensorsolve'),
        xfail('nn.functional.max_pool3d'),
        xfail('vdot'),
        xfail('linalg.cross'),
        xfail('nn.functional.feature_alpha_dropout', 'without_train'),
        xfail('linalg.lu_factor', ''),
        xfail('nn.functional.dropout2d', ''),
        xfail('nn.functional.kl_div', ''),
        xfail('pca_lowrank', ''),
        xfail('svd_lowrank', ''),
        xfail('linalg.lu_factor_ex', ''),
        xfail('nn.functional.feature_alpha_dropout', 'with_train'),
        xfail('special.log_ndtr', ''),
        xfail('fft.ihfft2'),  # conj_physical fallback
        xfail('fft.ihfftn'),  # conj_physical fallback
        xfail('istft'),  # col2im fallback
        xfail('polar'),  # complex fallback
        xfail('nn.functional.l1_loss', ''),
        xfail('nn.functional.max_unpool3d', 'grad'),
        xfail('nn.functional.smooth_l1_loss', ''),
        xfail('nn.functional.max_unpool2d', 'grad'),
        xfail('nn.functional.soft_margin_loss', ''),
        xfail('nn.functional.binary_cross_entropy_with_logits', ''),
        xfail('nn.functional.max_unpool1d', 'grad'),
        xfail('lu_unpack'),
        xfail('nn.functional.glu'),
        xfail('nn.functional.bilinear'),  # trilinear doesn't have batching rule
    }))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    def test_vmapjvpall_has_batch_rule(self, device, dtype, op):
        if is_inplace(op, op.get_op()):
            # TODO: test in-place
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=False)

        if not op.supports_forward_ad:
            self.skipTest("Skipped! Forward AD not supported.")
            return

        def test():
            for sample in samples:
                arg_values = [sample.input] + list(sample.args)
                kwarg_values = sample.kwargs
                args = tuple([*arg_values, *kwarg_values])
                fn, args = get_jvp_variant_primals_tangents(op, sample)
                for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                        fn, args, {}, opinfo=op, compute_loop_out=False):
                    pass
        check_vmap_fallback(self, test, op, dry_run=False)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @toleranceOverride({torch.float32: tol(atol=1e-04, rtol=1e-04)})
    @skipOps('TestOperators', 'test_vmapvjp_has_batch_rule', vmapvjp_fail.union({
        xfail('view_as_complex'),
        xfail('__getitem__', ''),
        xfail('cholesky'),
        xfail('complex'),
        xfail('copysign'),
        xfail('cummax'),
        xfail('cummin'),
        xfail('cumprod'),
        xfail('eig'),
        xfail('nansum'),
        xfail('nanmean'),
        xfail('fmin'),
        xfail('fmax'),
        xfail('special.log_ndtr'),
        xfail('index_copy'),
        xfail('index_fill'),
        xfail('linalg.cholesky'),
        xfail('linalg.cholesky_ex'),
        xfail('linalg.det'),
        xfail('linalg.eig'),
        xfail('linalg.eigh'),
        xfail('linalg.eigvals'),
        xfail('linalg.householder_product'),
        xfail('linalg.lstsq', ''),
        xfail('linalg.lstsq', 'grad_oriented'),
        xfail('linalg.pinv'),
        xfail('linalg.qr'),
        xfail('linalg.pinv', 'hermitian'),
        xfail('linalg.slogdet'),
        xfail('linalg.solve'),
        xfail('logdet'),
        xfail('lu'),
        xfail('lu_solve'),
        xfail('lu_unpack'),
        xfail('masked_fill'),
        xfail('masked_scatter'),
        xfail('masked_select'),
        xfail('matrix_exp'),
        xfail('nanquantile'),
        xfail('pinverse'),
        xfail('prod'),
        xfail('put'),
        xfail('quantile'),
        xfail('renorm'),
        xfail('symeig'),
        xfail('take'),
        xfail('tensor_split'),
        xfail('to_sparse'),
        xfail('unfold'),
        xfail('vdot'),
        xfail('nn.functional.dropout'),
        xfail('_masked.prod'),
        xfail('fft.ihfft2'),
        xfail('fft.ihfftn'),
        xfail('cross'),
        xfail('linalg.cross'),
        xfail('nn.functional.gaussian_nll_loss'),
        xfail('nn.functional.huber_loss'),
        xfail('nn.functional.poisson_nll_loss'),
        xfail('nn.functional.bilinear'),
        xfail('nn.functional.fractional_max_pool3d'),
        xfail('as_strided'),
        xfail('linalg.solve_triangular'),
        xfail('stft'),
        xfail('nn.functional.rrelu'),
        xfail('nn.functional.embedding_bag'),
        xfail('nn.functional.max_pool3d'),
        xfail('istft'),
        xfail('nn.functional.fractional_max_pool2d'),
        xfail('linalg.tensorsolve'),
        xfail('linalg.lu_factor', ''),
        xfail('nn.functional.feature_alpha_dropout', 'with_train'),
        xfail('nn.functional.kl_div', ''),
        xfail('pca_lowrank', ''),
        xfail('nn.functional.dropout2d', ''),
        xfail('nn.functional.feature_alpha_dropout', 'without_train'),
        xfail('svd_lowrank', ''),
        xfail('linalg.lu_factor_ex', ''),

        xfail('nn.functional.max_unpool2d', ''),
        xfail('nn.functional.multi_margin_loss', ''),
        xfail('nn.functional.multilabel_margin_loss', ''),
        xfail('nn.functional.pdist', ''),
        xfail('nn.functional.smooth_l1_loss', ''),
        xfail('scatter_reduce', 'prod'),
        xfail('scatter_reduce', 'amax'),
        xfail('nn.functional.max_unpool1d', ''),
        xfail('nn.functional.max_unpool3d', ''),
        xfail('scatter_reduce', 'sum'),
        xfail('scatter_reduce', 'mean'),
        xfail('nn.functional.max_unpool3d', 'grad'),
        xfail('nn.functional.soft_margin_loss', ''),
        xfail('scatter_reduce', 'amin'),
        xfail('nn.functional.max_unpool1d', 'grad'),
        xfail('nn.functional.l1_loss', ''),
        xfail('nn.functional.max_unpool2d', 'grad'),
        xfail('qr'),
    }))
    def test_vmapvjp_has_batch_rule(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        def test():
            for sample in samples:
                cotangents = get_sample_cotangents(op, sample)
                fn, args = get_vjp_fn_and_args_with_cotangents(op, sample, cotangents)
                for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                        fn, args, {}, opinfo=op, compute_loop_out=False):
                    pass
                for a_op in op.aliases:
                    fn, args = get_vjp_fn_and_args_with_cotangents(a_op, sample, cotangents)
                    for loop_out, batched_out in get_fallback_and_vmap_exhaustive(
                            fn, args, {}, opinfo=op, compute_loop_out=False):
                        pass

        check_vmap_fallback(self, test, op, dry_run=False)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_vjpvmap', vjp_fail.union({
        skip('bernoulli', ''),  # vjpvmap testing can't handle randomness
        skip('normal', ''),  # vjpvmap testing can't handle randomness
        skip('normal', 'number_mean'),  # vjpvmap testing can't handle randomness

        # fallback path doesn't work
        # All of the following are bugs and need to be fixed
        xfail('__getitem__', ''),
        xfail('clamp', ''),
        xfail('index_put', ''),
        xfail('matrix_exp'),
        xfail('view_as_complex'),
        xfail('nn.functional.gaussian_nll_loss'),
        xfail('masked_select'),
        skip('nn.functional.fractional_max_pool3d'),  # generator works on cpu, fails on cuda
        xfail('__rpow__'),  # https://github.com/pytorch/functorch/issues/617
        xfail('as_strided'),
        skip('nn.functional.fractional_max_pool2d'),  # generator works on cpu, fails on cuda
        xfail('column_stack', ''),
        xfail('nn.functional.dropout2d', ''),
        xfail('svd_lowrank', ''),
        xfail('pca_lowrank', ''),
        xfail('nn.functional.feature_alpha_dropout', 'without_train'),
        xfail('nn.functional.feature_alpha_dropout', 'with_train'),
        # something weird happening with channels_last
        xfail('bfloat16'),
        xfail('double'),
        xfail('float'),
        xfail('half'),
    }))
    def test_vjpvmap(self, device, dtype, op):
        # NB: there is no vjpvmap_has_batch_rule test because that is almost
        # certainly redundant with the vmap_has_batch_rule test in test_vmap.py

        # one-off skip
        if op.name == 'nn.functional.dropout':
            self.skipTest("Skipped!")

        if not op.supports_autograd:
            # If the op doesn't support autograd, vmap(op) won't either
            self.skipTest("Skipped! Autograd not supported.")
            return

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        batch_norm_fns = ("nn.functional.batch_norm", "nn.functional.instance_norm")  # instance norm calls batch norm
        is_batch_norm = op.name in batch_norm_fns

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            generator = get_exhaustive_batched_inputs(args, kwargs, for_batch_norm=is_batch_norm)

            for batched_args, in_dims, kwargs in generator:
                vmapped_op = vmap(op, in_dims)
                fn, primals = normalize_op_input_output2(vmapped_op, batched_args, kwargs,
                                                         sample.output_process_fn_grad)
                result = fn(*primals)
                cotangents = tree_map(lambda x: torch.randn_like(x), result)

                _, vjp_fn = vjp(fn, *primals)
                result_vjps = vjp_fn(cotangents)

                _, vjp_fn = ref_vjp(fn, *primals)
                expected_vjps = vjp_fn(cotangents)

                self.assertEqual(result_vjps, expected_vjps)

    def _compare_jacobians_of_vjp(self, fn, cotangents_and_primals, argnums=None, atol_rtol=None):
        if argnums is None:
            argnums = tuple(range(len(cotangents_and_primals)))

        def get_vjp(cotangents, *primals):
            _, vjp_fn = vjp(fn, *primals)
            return vjp_fn(cotangents)

        jacobian_jvp = jacfwd(get_vjp, argnums)(*cotangents_and_primals)
        jacobian_vjp = jacrev(get_vjp, argnums)(*cotangents_and_primals)

        # For dtype changing operations, the jacobians have different dtype.
        jacobian_jvp = tree_map(lambda x: x.to(torch.float), jacobian_jvp)
        jacobian_vjp = tree_map(lambda x: x.to(torch.float), jacobian_vjp)

        if atol_rtol is not None:
            (atol, rtol) = atol_rtol
            self.assertEqual(jacobian_jvp, jacobian_vjp, atol=atol, rtol=rtol)
        else:
            self.assertEqual(jacobian_jvp, jacobian_vjp)

    @ops(functorch_lagging_op_db + additional_op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestOperators', 'test_jvpvjp', vjp_fail.union({
        # These are weirdly non-deterministic
        skip('nn.functional.fractional_max_pool2d'),  # Random
        skip('nn.functional.fractional_max_pool3d'),  # Random

        # RuntimeError: Trying to set a forward gradient that has a different size than that of the original Tensor,
        # this is not supported. Tensor is of size [5, 2, 3] while the given forward gradient is of size [1, 2, 3].
        xfail('normal', ''),
        xfail('_masked.amax', ''),
        xfail('_masked.amin', ''),
        xfail('_masked.log_softmax', ''),
        xfail('_masked.softmax', ''),
        xfail('_masked.softmin', ''),
        xfail('amax', ''),
        xfail('amin', ''),
        xfail('cdist', ''),
        xfail('cholesky', ''),
        xfail('eig', ''),
        xfail('linalg.det', ''),
        xfail('linalg.matrix_norm', ''),
        xfail('linalg.slogdet', ''),
        xfail('logcumsumexp', ''),
        xfail('logdet', ''),
        xfail('nanmean', ''),
        xfail('nansum', ''),
        xfail('nn.functional.batch_norm', ''),
        xfail('nn.functional.batch_norm', 'without_cudnn', device_type='cuda'),
        xfail('nn.functional.embedding', ''),
        xfail('nn.functional.embedding', 'functorch'),
        xfail('nn.functional.embedding_bag', ''),
        xfail('nn.functional.grid_sample', ''),
        xfail('nn.functional.hardsigmoid', ''),
        xfail('nn.functional.hardswish', ''),
        xfail('nn.functional.huber_loss', ''),
        xfail('nn.functional.instance_norm', ''),
        xfail('nn.functional.logsigmoid', ''),
        xfail('nn.functional.pad', 'circular'),
        xfail('nn.functional.prelu', ''),
        xfail('nn.functional.softmin', ''),
        xfail('nn.functional.softmin', 'with_dtype'),
        xfail('nn.functional.softplus', ''),
        xfail('renorm', ''),
        xfail('std_mean', ''),
        xfail('symeig', ''),
        xfail('var_mean', ''),
        xfail('nn.functional.feature_alpha_dropout', 'with_train'),
        xfail('nn.functional.kl_div', ''),
        xfail('pca_lowrank', ''),
        xfail('nn.functional.dropout2d', ''),
        xfail('nn.functional.feature_alpha_dropout', 'without_train'),
        xfail('svd_lowrank', ''),
        xfail('nn.functional.multilabel_margin_loss', ''),
        xfail('nn.functional.multilabel_soft_margin_loss', ''),
        xfail('scatter_reduce', 'amax'),
        xfail('scatter_reduce', 'amin'),
        xfail('nn.functional.soft_margin_loss', ''),
        xfail('nn.functional.pdist', ''),
        xfail('scatter_reduce', 'sum'),
        xfail('nn.functional.multi_margin_loss', ''),
        xfail('nn.functional.smooth_l1_loss', ''),
        xfail('scatter_reduce', 'mean'),
        xfail('scatter_reduce', 'prod'),
        skip('linalg.householder_product', '', device_type='cuda'),  # flaky, I'm not sure why
        xfail('nn.functional.binary_cross_entropy_with_logits'),
    }))
    def test_jvpvjp(self, device, dtype, op):
        if not op.supports_autograd:
            self.skipTest("Skipped! Autograd not supported.")
            return

        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # TODO: test in-place
        if is_inplace(op, op.get_op()):
            self.skipTest("Skipped! NYI: inplace-testing not supported.")
            return

        for sample in samples:
            fn, primals = normalize_op_input_output(op, sample)
            result = fn(*primals)
            cotangents = tree_map(lambda x: torch.randn_like(x), result)

            primals_tangents = tree_map(lambda x: torch.randn_like(x), primals)
            cotangents_tangents = tree_map(lambda x: torch.randn_like(x), cotangents)

            if isinstance(primals[0], torch.Tensor) and primals[0].numel() == 0:
                # typically the first primal arg is the input. If the input has no elements, we will typically run
                # into an issue of "Expected Tensor but got None"
                continue

            def push_vjp(primals, cotangents):
                _, vjp_fn = vjp(fn, *primals)
                return vjp_fn(cotangents)

            result = jvp(push_vjp, (primals, cotangents), (primals_tangents, cotangents_tangents))
            self.assertEqual(len(result), 2)

            def tree_map2(fn, first, second):
                flat_first, spec_first = tree_flatten(first)
                flat_second, spec_second = tree_flatten(second)
                assert spec_first == spec_second
                flat_result = [fn(f, s) for f, s in zip(flat_first, flat_second)]
                return tree_unflatten(flat_result, spec_first)

            def reference(primals, cotangents, primals_tangents, cotangents_tangents):
                with fwAD.dual_level():
                    primal_duals = tree_map2(fwAD.make_dual, primals, primals_tangents)
                    _, vjp_fn = ref_vjp(fn, *primal_duals)

                    cotangent_duals = tree_map2(fwAD.make_dual, cotangents, cotangents_tangents)
                    result = vjp_fn(cotangent_duals)

                    flat_result, spec = tree_flatten(result)
                    primals_out, tangents_out = zip(*[fwAD.unpack_dual(r) for r in flat_result])
                    tangents_out = [t if t is not None else torch.zeros_like(p)
                                    for p, t in zip(primals_out, tangents_out)]
                    expected = (tree_unflatten(primals_out, spec), tree_unflatten(tangents_out, spec))
                return expected

            # HACK: obviously pytorch should also have the same coverage
            # For things that do have the same coverage, we test that jvp x vjp
            # are the same between PyTorch and functorch. For things that don't,
            # we check that jacfwd(vjp) and jacrev(vjp) are the same. This results
            # in slower tests.
            FUNCTORCH_HAS_FORMULA_BUT_NOT_PYTORCH = {
                'nn.functional.nll_loss',
                'nn.functional.l1_loss',
                'nn.functional.mse_loss',
                'softmax',
                'log_softmax',
                'nn.functional.cross_entropy',
                'nn.functional.binary_cross_entropy',
                'nn.functional.layer_norm'
            }
            if op.name in FUNCTORCH_HAS_FORMULA_BUT_NOT_PYTORCH:
                self.assertFalse(op.supports_fwgrad_bwgrad,
                                 f"{op.name} now supports forward over reverse without a decomposition. " +
                                 "Please remove the decomposition version")
                def is_differentiable(t):
                    return isinstance(t, torch.Tensor) and t.dtype == torch.float32
                args = (cotangents, *primals)
                if op.name == 'nn.functional.binary_cross_entropy':
                    argnums = (0, 1)  # targets is float32 but isn't differentiable
                    atol_rtol = 1.5e-4, 1.3e-06
                else:
                    argnums = tuple(i for i in range(len(args)) if is_differentiable(args[i]))
                    atol_rtol = None
                self._compare_jacobians_of_vjp(fn, args, argnums, atol_rtol)
            else:
                expected = reference(primals, cotangents, primals_tangents, cotangents_tangents)
                self.assertEqual(result, expected)

    def _make_extremal_inputs(self, shape, device):
        if shape == None:
            return (None,)
        return (
            torch.full(shape, -1000., device=device),
            torch.zeros(shape, device=device),
            torch.full(shape, 1000., device=device),
        )

    def _arg_and_kwarg_options(self, args_options, kwargs_options):
        return itertools.product(*args_options, kwargs_options)

    def test_extremal_numerics_nll_loss(self, device):
        N, C = 3, 4
        d1, d2, d3 = 5, 6, 7
        shapes = (
            ((N, C), (N,), (C,)),
            ((N, C), (N,), None),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), None),
        )
        kwargs_options = ({'ignore_index': 0, 'reduction': 'mean'}, {'reduction': 'sum'}, {'reduction': 'none'}, {})
        for input_shape, target_shape, weight_shape in shapes:
            input_options = self._make_extremal_inputs(input_shape, device)
            for input, kwargs in self._arg_and_kwarg_options((input_options,), kwargs_options):
                if weight_shape is None:
                    weight = None
                else:
                    weight = torch.randn(weight_shape, device=device)
                target = torch.randint(0, C, target_shape, device=device)
                target[0] = 1  # since we're ignoring index 0, at least one element must be non-zero

                fn = functools.partial(torch.nn.functional.nll_loss, target=target, weight=weight, **kwargs)
                result = fn(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input))

    def test_extremal_numerics_l1_loss(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({'reduction': 'sum'}, {'reduction': 'none'}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            target_options = self._make_extremal_inputs(shape, device)
            for input, target, kwargs in self._arg_and_kwarg_options((input_options, target_options), kwargs_options):
                result = torch.nn.functional.l1_loss(input, target)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(torch.nn.functional.l1_loss, (cotangents, input, target))

    def test_extremal_numerics_mse_loss(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({'reduction': 'sum'}, {'reduction': 'none'}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            target_options = self._make_extremal_inputs(shape, device)
            for input, target, kwargs in self._arg_and_kwarg_options((input_options, target_options), kwargs_options):
                result = torch.nn.functional.mse_loss(input, target)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(torch.nn.functional.mse_loss, (cotangents, input, target))

    def test_extremal_numerics_softmax(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({'dim': 1}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            for input, kwargs in self._arg_and_kwarg_options((input_options,), kwargs_options):
                result = torch.nn.functional.softmax(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(torch.nn.functional.softmax, (cotangents, input))


    def test_extremal_numerics_log_softmax(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        kwargs_options = ({'dim': 1}, {})
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            for input, kwargs in self._arg_and_kwarg_options((input_options,), kwargs_options):
                result = torch.nn.functional.log_softmax(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(torch.nn.functional.log_softmax, (cotangents, input))

    def test_extremal_numerics_cross_entropy(self, device):
        N, C = 3, 4
        d1, d2, d3 = 5, 6, 7
        shapes = (
            ((N, C), (N,), (C,)),
            ((N, C), (N,), None),
            ((N, C), (N, C), (C,)),
            ((N, C), (N, C), None),
            ((C,), (), (C,)),
            ((C,), (), None),
            ((C,), (C,), (C,)),
            ((C,), (C,), None),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, d1, d2, d3), None),
            ((N, C, d1, d2, d3), (N, C, d1, d2, d3), (C,)),
            ((N, C, d1, d2, d3), (N, C, d1, d2, d3), None),
        )
        for input_shape, target_shape, weight_shape in shapes:
            input_options = self._make_extremal_inputs(input_shape, device)
            kwargs_options = [{'reduction': 'sum'}, {'reduction': 'none'}, {}]
            if input_shape != target_shape:
                kwargs_options.append({'ignore_index': 0, 'reduction': 'mean'})

            for input, kwargs in self._arg_and_kwarg_options((input_options,), kwargs_options):
                if weight_shape is None:
                    weight = None
                else:
                    weight = torch.randn(weight_shape, device=device)

                if input_shape == target_shape:
                    target = torch.rand(target_shape, device=device)
                elif len(target_shape) == 0:
                    target = torch.tensor(1, device=device)  # must be non-zero since ignore_index may be 0
                else:
                    target = torch.randint(0, C, target_shape, device=device)

                fn = functools.partial(torch.nn.functional.cross_entropy, target=target, weight=weight, **kwargs)
                result = fn(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input), atol_rtol=(1e-4, 1e-5))

    def test_extremal_numerics_binary_cross_entropy(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        for shape in shapes:
            weight_options = self._make_extremal_inputs(shape, device)
            kwargs_options = [{'reduction': 'sum'}, {'reduction': 'none'}, {}]

            for weight, kwargs in self._arg_and_kwarg_options((weight_options,), kwargs_options):
                input = torch.rand(shape, device=device)
                target = torch.rand(shape, device=device)
                fn = functools.partial(torch.nn.functional.binary_cross_entropy, target=target, weight=weight, **kwargs)
                result = fn(input)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input), atol_rtol=(1e-4, 2e-5))

    def test_extremal_numerics_layer_norm(self, device):
        N, C, H, W = 3, 4, 5, 6
        shapes = ((N, C), (N, C, H), (N, C, H, W))
        for shape in shapes:
            input_options = self._make_extremal_inputs(shape, device)
            normalized_shape = shape[1:]
            weight_options = self._make_extremal_inputs(normalized_shape, device)
            bias_options = self._make_extremal_inputs(normalized_shape, device)

            for input, bias, weight in self._arg_and_kwarg_options((input_options, bias_options, weight_options), ()):
                def fn(input, weight, bias):
                    return torch.nn.functional.layer_norm(input, normalized_shape, weight=weight, bias=bias)
                result = fn(input, weight, bias)
                cotangents = torch.randn_like(result, device=device)
                self._compare_jacobians_of_vjp(fn, (cotangents, input, weight, bias))

    @ops(filter(lambda op: op.name == "nn.functional.group_norm", functorch_lagging_op_db + additional_op_db),
         allowed_dtypes=(torch.float32, torch.double))  # TODO: generalize
    def test_group_norm_backward(self, device, dtype, op):
        # hacky, only works since no group norm inputs can be scalars
        def was_skipped_from_batched_tensors(batched_out, batch_size):
            return batched_out.shape == (batch_size,) and all(tuple(e == 1 for e in batched_out))

        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)

        for sample_input in sample_inputs:
            cotangents = get_sample_cotangents(op, sample_input)
            f, args = get_autograd_fn_and_args_with_cotangents(op, sample_input, cotangents)
            for loop_out, batched_out in get_fallback_and_vmap_exhaustive(f, args, {}, opinfo=op):
                if all(was_skipped_from_batched_tensors(bo, lo.shape[0]) for (bo, lo) in zip(batched_out, loop_out)):
                    continue  # we weren't able to use the batched tensor in autograd.grad
                self.assertEqual(loop_out, batched_out)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestOperators, globals(), only_for=only_for)

if __name__ == '__main__':
    run_tests()
