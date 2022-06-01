# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from torch.testing._internal.common_utils import (
    TestCase, run_tests, parametrize, subtest, instantiate_parametrized_tests
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import warnings
import math
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCPU
from torch.testing._internal.common_dtype import get_all_fp_dtypes
from torch.testing._internal.common_utils import IS_WINDOWS
from functools import partial
from functorch.experimental import replace_all_batch_norm_modules_

import functorch
from functorch import (
    grad, vjp, vmap, jacrev, jacfwd, grad_and_value, hessian,
    jvp, make_functional, make_functional_with_buffers,
    combine_state_for_ensemble, make_fx
)
from functorch._src.make_functional import (
    functional_init, functional_init_with_buffers,
)
from functorch._src.eager_transforms import _argnums_partial, enable_fwd_grad
from functorch.experimental import functionalize

if not IS_WINDOWS:
    from functorch._src.custom_function import custom_vjp

# NB: numpy is a testing dependency!
import numpy as np

USE_TORCHVISION = False
try:
    import torchvision  # noqa: F401
    USE_TORCHVISION = True
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try "
                  "to install it with commands from pytorch.org, post-fixed with "
                  "`--no-deps` to avoid overwriting the pytorch installation",
                  UserWarning)

# TestCase for _argnums_partial, an important helper funciton


class TestArgnumsPartial(TestCase):
    def test_invalid_argnum_type(self):
        x = torch.randn(3)
        args = (x,)
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            _argnums_partial(torch.sin, args, 0.0)
        with self.assertRaisesRegex(RuntimeError, "int or Tuple"):
            _argnums_partial(torch.sin, args, [0])
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            _argnums_partial(torch.sin, args, (0.0,))

        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        def f(a, b, c, d, e):
            return a
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            _argnums_partial(torch.sin, args, ((0, 1), 2))

    def test_out_of_bounds_argnum_values(self):
        x = torch.randn(3)
        args = (x,)
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _argnums_partial(torch.sin, args, 1)
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _argnums_partial(torch.sin, args, -2)
        with self.assertRaisesRegex(RuntimeError, "positional inputs"):
            _argnums_partial(torch.sin, args, (-2,))

    def test_not_enough_argnums(self):
        x = torch.randn(3)
        args = (x,)
        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            _argnums_partial(torch.sin, args, ())

    def test_duplicate_argnums(self):
        x = torch.randn(3)
        args = (x, x)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _argnums_partial(torch.add, args, (0, 0))
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            _argnums_partial(torch.add, args, (0, -2))

    def test_flat_args_with_positive_int_argnum(self):
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        def f(a, b, c, d, e):
            return a

        f_new, res = _argnums_partial(f, args, 0)
        self.assertEqual(res, (0.1,))
        self.assertEqual(f_new(*res), 0.1)

        f_new, res = _argnums_partial(f, args, 4)
        self.assertEqual(res, (4.1,))
        self.assertEqual(f_new(*res), 0.1)

    def test_flat_args_with_negative_int_argnum(self):
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        def f(a, b, c, d, e):
            return a

        expected = f(*args)
        f_new, res = _argnums_partial(f, args, -1)
        self.assertEqual(res, (4.1,))
        self.assertEqual(f_new(*res), expected)

        f_new, res = _argnums_partial(f, args, -5)
        self.assertEqual(res, (0.1,))
        self.assertEqual(f_new(*res), expected)

    def test_flat_args_with_tuple_argnum(self):
        args = (0.1, 1.1, 2.1, 3.1, 4.1)

        def f(a, b, c, d, e):
            return a

        f_new, res = _argnums_partial(f, args, (0, 1, 2, 3, 4))
        self.assertEqual(f_new(*res), 0.1)
        self.assertEqual(res, args)

        f_new, res = _argnums_partial(f, args, (0, -3))
        self.assertEqual(f_new(*res), 0.1)
        self.assertEqual(res, (0.1, 2.1))

    def test_pytree_args(self):
        args = ((0.1, 1.1), 2.0, [3.1])

        def f(a, b, c):
            return a[0] + a[1] + b + c[0]

        expected = f(*args)

        f_new, res = _argnums_partial(f, args, 0)
        self.assertEqual(res, args[0:1])
        self.assertEqual(f_new(*res), expected)

        f_new, res = _argnums_partial(f, args, (0,))
        self.assertEqual(res, args[0:1])
        self.assertEqual(f_new(*res), expected)

        f_new, res = _argnums_partial(f, args, -1)
        self.assertEqual(res, args[-1:])
        self.assertEqual(f_new(*res), expected)

        f_new, res = _argnums_partial(f, args, (0, -2))
        self.assertEqual(res, args[0:2])
        self.assertEqual(f_new(*res), expected)

    def test_argnums_reorders(self):
        args = ((0.1, 1.1, 2.1), 3.1, 4.1)

        def f(a, b, c):
            return a[0] + a[1] + a[2] + b + c

        expected = f(*args)
        f_new, res = _argnums_partial(f, args, (1, 0))
        self.assertEqual(res, (args[1], args[0]))
        self.assertEqual(f_new(*res), expected)

    def test_function_with_default_args(self):
        args = ((0.1, 1.1, 2.1), 3.1)

        def f(a, b, c=4.1):
            return a[0] + a[1] + a[2] + b + c

        expected = f(*args)
        f_new, res = _argnums_partial(f, args, -2)
        self.assertEqual(res, args[0:1])
        self.assertEqual(f_new(*res), expected)

        args = ((0.1, 1.1, 2.1), 3.1, 5.1)
        expected = f(*args)
        f_new, res = _argnums_partial(f, args, -1)
        self.assertEqual(res, args[-1:])
        self.assertEqual(f_new(*res), expected)


class TestGradTransform(TestCase):
    def test_primitive(self, device):
        x = torch.randn([], device=device)
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    def test_composite_simple(self, device):
        x = torch.randn(2, 3, 4, device=device)
        result = grad(lambda x: torch.flatten(x).sum())(x)
        self.assertEqual(result, torch.ones_like(x))

    def test_fn_with_kwargs(self, device):
        def foo(x, y):
            return (x * y).sum()

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        expected = grad(foo)(x, y)
        result = grad(foo)(x, y=y)
        self.assertEqual(result, expected)

    def test_composite_complicated(self, device):
        x = torch.randn(3, device=device)
        y = torch.randn(3, 5, device=device)

        def foo(x, y):
            result = x @ y
            return result.sum()

        result = grad(foo)(x, y)

        x.requires_grad_()
        out = foo(x, y)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_composite_two_ops(self, device):
        N, C = 2, 5
        y = torch.randn(N, C, device=device)
        targets = torch.randint(0, C, (N,), device=device)

        def foo(y, targets):
            return F.cross_entropy(y, targets)

        result = grad(foo)(y, targets)

        y.requires_grad_()
        expected, = torch.autograd.grad(foo(y, targets), y)

        self.assertEqual(result, expected)

    def _test_attributes(self, get_attr_lambda, device):
        x = torch.randn(2, 3, 5, dtype=torch.double, device=device)
        expected = get_attr_lambda(x)

        def foo(x):
            self.assertEqual(get_attr_lambda(x), expected)
            return x.sum()

        grad(foo)(x)

    def test_shape(self, device):
        self._test_attributes(lambda x: x.shape, device)

    def test_dtype(self, device):
        self._test_attributes(lambda x: x.dtype, device)

    def test_is_cuda(self, device):
        self._test_attributes(lambda x: x.is_cuda, device)

    def test_numel(self, device):
        self._test_attributes(lambda x: x.numel(), device)

    def test_inplace(self, device):
        x = torch.randn([], device=device)

        def foo(x):
            return x.clone().sin_()

        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_inplace_on_view(self, device):
        x = torch.randn(3, device=device)

        def foo(x):
            y = x.clone()
            y0 = y[0]
            y0.sin_()
            return y.sum()

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_inplace_on_view_base(self, device):
        x = torch.randn(3, device=device)

        def foo(x):
            y = x.clone()
            y0 = y[0]
            y.sin_()
            return y0

        result = grad(foo)(x)

        x.requires_grad_()
        out = foo(x)
        expected, = torch.autograd.grad(out, x)

        self.assertEqual(result, expected)

    def test_inplace_on_captures(self, device):
        x = torch.tensor([1., 2., 3.], device=device)
        captured = torch.randn(3, device=device)

        def foo(x):
            captured.copy_(x)
            return (x * captured).sum()

        with self.assertRaisesRegex(RuntimeError, 'mutate a captured Tensor'):
            grad(foo)(x)

    def test_nesting_simple(self, device):
        x = torch.randn([], device=device)
        result = grad(grad(torch.sin))(x)
        self.assertEqual(result, -torch.sin(x))

    def test_escaped_wrappers_are_marked_as_dead(self, device):
        x = torch.randn([], device=device)
        escaped = []

        def foo(x):
            y = x.sin()
            escaped.append(y)
            return y

        grad(foo)(x)
        self.assertEqual(functorch._C.dlevel(escaped[0]), -1)

    def test_escaped_wrappers_are_ignored(self, device):
        x = torch.randn([], device=device)
        escaped = []

        def foo(x):
            y = x.sin()
            escaped.append(y)
            return y

        grad(foo)(x)

        something = escaped[0].sum()
        self.assertEqual(functorch._C.dlevel(something), 0)
        self.assertEqual(something, x.sin().sum())

    def test_vjp(self, device):
        x = torch.randn([], device=device)
        out, vjp_fn = vjp(torch.sin, x)
        self.assertEqual(out, x.sin())

        v = torch.randn([], device=device)
        result, = vjp_fn(v)
        self.assertEqual(result, v * x.cos())

    def test_vjp_two_outputs(self, device):
        def f(x):
            return x, x
        result, vjp_fn = vjp(f, torch.tensor(1.))
        vjp_fn(result)

    def test_conj_bit(self):
        x = torch.tensor(1+1j)

        def foo(x):
            assert not x.is_conj()
            y = x.conj()
            assert y.is_conj()
            return y
        res = grad(foo)(x)
        self.assertEqual(res, torch.ones_like(res))

    def test_composed_with_autograd(self, device):
        x = torch.randn([], requires_grad=True, device=device)

        y = grad(torch.sin)(x)
        result, = torch.autograd.grad(y, x)
        self.assertEqual(result, -x.sin())

    def test_grad_of_vjp_composition(self, device):
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            out, vjp_fn = vjp(torch.sin, x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_vjp_of_grad_composition(self, device):
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            out, vjp_fn = vjp(grad(torch.sin), x)
            return vjp_fn(y)[0]

        result = foo(x, y)
        expected = -y * x.sin()
        self.assertEqual(result, expected)

    def test_grad_of_vjp_of_grad_composition(self, device):
        x = torch.randn([], device=device)
        y = torch.randn([], device=device)

        def foo(x, y):
            df, vjp_fn = vjp(grad(lambda x: -torch.cos(x)), x)
            return grad(lambda y: vjp_fn(y)[0])(y)

        result = foo(x, y)
        expected = x.cos()
        self.assertEqual(result, expected)

    def test_views(self, device):
        x = torch.randn([], requires_grad=True, device=device)
        y = torch.randn([], requires_grad=True, device=device)

        def silly_sin(x):
            x = x.view([])
            x = x.sin()
            return x

        def foo(x, y):
            z1 = grad(silly_sin)(x)
            z2 = torch.cos(y)
            return z1 + z2

        result = foo(x, y)
        grads = torch.autograd.grad(result, [x, y])
        self.assertEqual(grads[0], -x.sin())
        self.assertEqual(grads[1], -y.sin())

    def test_view_inplace_simple(self, device):
        def foo(x):
            x = x.clone()
            x.view([]).sin_()
            return x

        x = torch.randn([], requires_grad=True, device=device)
        result = grad(foo)(x)
        self.assertEqual(result, x.cos())

    def test_invalid_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        with self.assertRaisesRegex(RuntimeError, 'but only'):
            grad(torch.mul, argnums=-3)(x, y)
        with self.assertRaisesRegex(RuntimeError, 'but only'):
            grad(torch.mul, argnums=2)(x, y)
        with self.assertRaisesRegex(RuntimeError, 'int or Tuple'):
            grad(torch.mul, argnums=[0])(x, y)
        with self.assertRaisesRegex(RuntimeError, 'must be int'):
            grad(torch.mul, argnums=('0',))(x, y)
        with self.assertRaisesRegex(RuntimeError, 'must be unique'):
            grad(torch.mul, argnums=(0, 0))(x, y)
        with self.assertRaisesRegex(RuntimeError, 'must be unique'):
            grad(torch.mul, argnums=(0, -2))(x, y)

    def test_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        gx = grad(torch.mul, argnums=0)(x, y)
        self.assertEqual(gx, y)

        gy = grad(torch.mul, argnums=1)(x, y)
        self.assertEqual(gy, x)

        gx, = grad(torch.mul, argnums=(0,))(x, y)
        self.assertEqual(gx, y)

        gx, gy = grad(torch.mul, argnums=(0, 1))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_out_of_order_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        gy, gx = grad(torch.mul, argnums=(1, 0))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_negative_argnums(self, device):
        x = torch.randn([])
        y = torch.randn([])
        gx = grad(torch.mul, argnums=-2)(x, y)
        self.assertEqual(gx, y)

        gy = grad(torch.mul, argnums=-1)(x, y)
        self.assertEqual(gy, x)

        gx, = grad(torch.mul, argnums=(-2,))(x, y)
        self.assertEqual(gx, y)

        gx, gy = grad(torch.mul, argnums=(-2, -1))(x, y)
        self.assertEqual(gx, y)
        self.assertEqual(gy, x)

    def test_grad_pytree_inputs(self, device):
        x = torch.randn([], device=device)

        def f(a, b):
            x, y = a
            return 1 * x + 2 * y + 3 * b['foo']

        args = ((x, x), {'foo': x})

        gx, gy = grad(f)(*args)
        self.assertEqual(gx, torch.tensor(1., device=device))
        self.assertEqual(gy, torch.tensor(2., device=device))

        (gx, gy), = grad(f, argnums=(0,))(*args)
        self.assertEqual(gx, torch.tensor(1., device=device))
        self.assertEqual(gy, torch.tensor(2., device=device))

        (gx, gy), gz = grad(f, argnums=(0, 1))(*args)
        self.assertEqual(gx, torch.tensor(1., device=device))
        self.assertEqual(gy, torch.tensor(2., device=device))
        self.assertEqual(gz['foo'], torch.tensor(3., device=device))

    def test_grad_aux_tensor(self, device):

        x = torch.randn(3, device=device)

        with self.assertRaisesRegex(
            RuntimeError,
            r'grad_and_value\(f\)\(\*args\): output of function f should be a tuple'
        ):
            grad(lambda t: [t, t], has_aux=True)(x)

        with self.assertRaisesRegex(
            RuntimeError,
            r'grad_and_value\(f\)\(\*args\): output of function f should be a tuple'
        ):
            grad(lambda t: (t, t + 2, t + 3), has_aux=True)(x)

        def f(t):
            y = t.sin()
            return y.sum(), t.cos()

        out, aux = grad(f, has_aux=True)(x)
        self.assertEqual(aux, x.cos())
        self.assertEqual(out, x.cos())

    def test_grad_aux_pytree(self, device):
        def f(x):
            y = x.sin()
            return y.sum(), {'a': x.cos(), 'b': [x.tan()]}

        x = torch.randn(3, device=device)

        out, aux = grad(f, has_aux=True)(x)
        _, expected_aux = f(x)
        self.assertEqual(aux, expected_aux)
        self.assertEqual(out, x.cos())

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(RuntimeError, r"Expected tensors, got unsupported type"):
                _ = grad(lambda x: (x.sum(), aux), has_aux=True)(x)
            with self.assertRaisesRegex(RuntimeError, r"Expected tensors, got unsupported type"):
                _ = grad(lambda x: (x.sum(), [x, aux]), has_aux=True)(x)

    def test_zero_grad(self, device):
        def f(x):
            return (x['a']**2.0).sum()
        inps = ({'a': torch.randn(10, device=device) + 3, 'b': torch.randn(10, device=device)})
        grads = grad(f)(inps)
        self.assertNotEqual(grads['a'].sum(), 0.0)
        self.assertEqual(grads['b'].sum(), 0.0)

    def test_unrelated_grad(self, device):
        x = torch.tensor(1., device=device)
        y = torch.tensor(2., device=device)

        def unrelated(x):
            return y

        result = grad(unrelated)(x)
        self.assertEqual(result, torch.zeros_like(x))

    def test_unrelated_vjp(self, device):
        x = torch.tensor(1., device=device)
        y = torch.tensor(2., device=device)
        v = torch.tensor(1., device=device)

        def unrelated(x):
            return y

        out, vjp_fn = vjp(unrelated, x)
        result = vjp_fn(v)
        expected = (torch.zeros_like(x),)
        self.assertEqual(result, expected)

    def test_unrelated_vjp_multiple_inputs_outputs(self, device):
        w = torch.tensor(3., device=device)
        x = torch.tensor(4., device=device)
        y = torch.tensor(2., device=device)
        v = torch.tensor(1., device=device)

        def unrelated(w, x):
            return y, y, x

        out, vjp_fn = vjp(unrelated, w, x)
        result = vjp_fn((v, v, v))
        expected = (torch.zeros_like(x), torch.ones_like(x))
        self.assertEqual(result, expected)

    # TODO: https://github.com/zou3519/functorch/issues/12
    @onlyCPU
    def test_unrelated_hessian(self, device):
        N = 5
        M = 3
        W = torch.randn(N, M, device=device)

        def f(x):
            return W @ x

        x = torch.randn(M)
        result = jacrev(jacrev(f))(x)
        expected = torch.zeros(N, M, M, device=device)
        self.assertEqual(result, expected)

    def test_vjp_pytree_input(self, device):
        def f(x):
            return x[0] * x[1][0]

        x = torch.randn([], device=device)
        v = torch.randn([], device=device)
        out, vjp_fn = vjp(f, (x, (x, x)))
        self.assertEqual(out, x * x)
        result = vjp_fn(v)
        self.assertEqual(result, ((x * v, (x * v, 0.)),))

    def test_vjp_pytree_output(self, device):
        def f(x):
            return x, (x, x)

        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        _, vjp_fn = vjp(f, x)
        result, = vjp_fn((v1, (v2, v3)))
        self.assertEqual(result, v1 + v2 + v3)

    def test_vjp_outputs_can_any_pytree(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError, r"vjp\(f, \*primals\): Expected f to be a function that has non-empty output"
            ):
                _, vjp_fn = vjp(lambda _: output, x)
                vjp_fn(t)

        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"vjp\(f, \*primals\): expected f\(\*primals\) to return only tensors"
            ):
                _, vjp_fn = vjp(lambda _: output, x)
                vjp_fn(t)

        # Check list output
        output, vjp_fn = vjp(lambda x: [x, x.sum()], x)
        vjp_out, = vjp_fn([t, t.sum()])
        assert isinstance(output, list) and len(output) == 2
        assert isinstance(vjp_out, torch.Tensor)

        # Check dict output
        output, vjp_fn = vjp(lambda x: {"x": x, "xsum": x.sum()}, x)
        vjp_out, = vjp_fn({"x": t, "xsum": t.sum()})
        assert isinstance(output, dict) and len(output) == 2 and "xsum" in output
        assert isinstance(vjp_out, torch.Tensor)

        def composite_output(x):
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        output, vjp_fn = vjp(composite_output, x)
        vjp_out, = vjp_fn([(t.sum(), {"a": t, "out": [t, t.sum()]}), ])
        assert isinstance(output, list)
        assert isinstance(output[0], tuple) and isinstance(output[0][1], dict)
        assert isinstance(vjp_out, torch.Tensor)

    def test_vjp_pytree_error(self, device):
        def f(x):
            return x, (x, x)

        x = torch.randn([], device=device)
        v1 = torch.randn([], device=device)
        v2 = torch.randn([], device=device)
        v3 = torch.randn([], device=device)
        _, vjp_fn = vjp(f, x)
        with self.assertRaisesRegex(RuntimeError, 'Expected pytree structure'):
            result, = vjp_fn(((v1, (v2, v3)),))

    def test_vjp_aux_tensor(self, device):

        x = torch.randn(3, device=device)

        with self.assertRaisesRegex(RuntimeError, r'vjp\(f, \*primals\): output of function f should be a tuple'):
            vjp(lambda t: [t, t], x, has_aux=True)

        with self.assertRaisesRegex(RuntimeError, r'vjp\(f, \*primals\): output of function f should be a tuple'):
            vjp(lambda t: (t, t + 2, t + 3), x, has_aux=True)

        def f(t):
            y = t.sin()
            return y, t.cos()

        out, vjp_fn, aux = vjp(f, x, has_aux=True)
        self.assertEqual(aux, x.cos())
        self.assertEqual(out, x.sin())

        v = torch.randn(3, device=device)
        grad_x, = vjp_fn(v)
        self.assertEqual(grad_x, v * x.cos())

    def test_vjp_aux_pytree(self, device):
        def f(x):
            y = x.sin()
            return y, {'a': x.cos(), 'b': [x.tan()]}

        x = torch.randn(3, device=device)

        out, vjp_fn, aux = vjp(f, x, has_aux=True)
        expected_out, expected_aux = f(x)
        self.assertEqual(out, expected_out)
        self.assertEqual(aux, expected_aux)

        v = torch.randn(3, device=device)
        grad_x, = vjp_fn(v)
        self.assertEqual(grad_x, v * x.cos())

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(RuntimeError, r"Expected tensors, got unsupported type"):
                _ = vjp(lambda x: (x, aux), x, has_aux=True)
            with self.assertRaisesRegex(RuntimeError, r"Expected tensors, got unsupported type"):
                _ = vjp(lambda x: (x, [x, aux]), x, has_aux=True)

    def test_functional_init(self, device):
        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        B = 10
        weights, fn, _ = functional_init(MLPClassifier, (B,), device=device)(32, 2)
        inputs = torch.randn(B, 7, 2, device=device)
        vmap(fn)(weights, (inputs,))

    def test_functional_init_with_buffers(self, device):
        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.bn = nn.BatchNorm1d(self.hidden_dim, affine=True)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.bn(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        B = 10
        weights, buffers, fn, _, _ = \
            functional_init_with_buffers(MLPClassifier, [B], device=device)(32, 2)
        inputs = torch.randn(B, 7, 2, device=device)
        vmap(fn)(weights, buffers, (inputs,))

    def test_advanced_indexing(self, device):
        def f(value):
            log_prob = torch.ones((), device=device)
            val = (torch.zeros(()) > 0)
            log_prob[val] = 0
            return value

        result = grad(f)(torch.randn((), device=device))
        self.assertEqual(result, torch.ones_like(result))

        def f2(value):
            value = value.clone()
            value[value > 0] = 0
            return value.sum()

        x = torch.randn(100, device=device)
        result = grad(f2)(x)
        self.assertEqual(result, (x <= 0).type_as(x))

    def test_tensor_ctor_inside_grad(self, device):
        def foo(x):
            return x * torch.tensor(2., device=device)

        x = torch.tensor(3.14, device=device)
        functorch.grad(foo)(x)

    @parametrize("op_list_data", [
        subtest(([vmap, ], [(4, 2), (64, 3, 32, 32)]), name='vmap'),
        subtest(([vmap, vmap], [(4, 3, 2), (64, 3, 32, 32)]), name='vmap_vmap'),
        subtest(([grad, ], [(0, ), [], (4, 2), (64, 3, 32, 32)]), name='grad'),
        subtest(([grad, grad], [[], ]), name='grad_grad'),
        subtest(([vmap, grad], [(4, 2)]), name='vmap_grad'),
    ])
    def test_tensor_print(self, device, op_list_data):

        op_list, shapes = op_list_data

        for dt in get_all_fp_dtypes():
            data = [torch.randn(s, dtype=dt, device=device) for s in shapes]

            for x in data:
                buf = None

                def foo(t):
                    nonlocal buf
                    buf = repr(t)
                    return t.mean()

                fn = foo
                bdim = 0
                for op in reversed(op_list):
                    if op == vmap:
                        fn = op(fn, in_dims=bdim)
                        bdim += 1
                    else:
                        fn = op(fn)

                expected = f"{repr(x)}"
                level = 0
                for op in op_list:
                    level += 1
                    if op == grad:
                        expected = f"GradTrackingTensor(lvl={level}, value={expected})"
                    elif op == vmap:
                        bdim -= 1
                        expected = f"BatchedTensor(lvl={level}, bdim={bdim}, value={expected})"

                fn(x)
                buf = buf.replace("\n", "").replace("  ", "")
                expected = expected.replace("\n", "").replace("  ", "")
                self.assertEqual(expected, buf)

    def test_no_grad_outside(self, device):
        x = torch.randn([], device=device, requires_grad=True)
        with torch.no_grad():
            y = grad(torch.sin)(x)
        self.assertEqual(y, x.cos())
        self.assertFalse(y.requires_grad)

    def test_no_grad_inside(self, device):
        def f(x):
            with torch.no_grad():
                shift = x ** 2
            return x ** 2 - shift

        x = torch.randn([], device=device)
        y = grad(f)(x)
        self.assertEqual(y, 2 * x)
        y = grad(grad(f))(x)
        self.assertEqual(y, 2)

        x = torch.randn([], device=device, requires_grad=True)
        y = grad(f)(x)
        z, = torch.autograd.grad(y, x)
        self.assertEqual(z, 2)

    def test_no_grad_mixed(self, device):
        def f(x):
            with torch.no_grad():
                shift = x ** 2
            return x ** 2 - shift

        x = torch.randn([], device=device, requires_grad=True)
        with torch.no_grad():
            y = grad(f)(x)

        self.assertEqual(y, 2 * x)
        self.assertFalse(y.requires_grad)

    def test_no_grad_nested_simple(self, device):
        def h(x):
            with torch.no_grad():
                shift = grad(lambda x: 0.25 * x ** 4)(x)
            return x ** 3 - shift

        x = torch.tensor(1.5, device=device, requires_grad=True)
        y = grad(h)(x)
        self.assertEqual(y, 3 * x ** 2)

        z, = torch.autograd.grad(y, x)
        self.assertEqual(z, 6 * x)

    def test_no_grad_nested_complicated(self, device):
        def f(x):
            with torch.no_grad():
                shift = x ** 3
            return x ** 3 - shift

        def g(x):
            r1 = grad(f)(x)
            with torch.no_grad():
                shift = grad(f)(x)
            return r1 - shift

        x = torch.randn([], requires_grad=True, device=device)
        y = grad(g)(x)
        # The only differential part of g is x ** 3
        self.assertEqual(y, 6 * x)

        z, = torch.autograd.grad(y, x)
        self.assertEqual(z, 6)

    def test_no_grad_value(self, device):
        def h(x):
            with torch.no_grad():
                gvalue, value = grad_and_value(lambda x: x ** 3)(x)
            return x ** 3 - value

        x = torch.tensor(1.6, device=device, requires_grad=True)
        y = grad(h)(x)
        self.assertEqual(y, 3 * x ** 2)

        z, = torch.autograd.grad(y, x)
        self.assertEqual(z, 6 * x)

    def test_no_grad_outside_vjp(self, device):
        def h(x):
            return x ** 2

        x = torch.tensor(2., requires_grad=True, device=device)
        with torch.no_grad():
            out, vjp_fn = vjp(h, x)
            y, = vjp_fn(torch.tensor(1., device=device))

        self.assertEqual(y, 2 * x)
        self.assertFalse(y.requires_grad)
        self.assertFalse(out.requires_grad)

    def test_no_grad_outside_vjp_fn(self, device):
        def h(x):
            return x ** 2

        x = torch.tensor(3.14, requires_grad=True, device=device)
        out, vjp_fn = vjp(h, x)
        with torch.no_grad():
            y, = vjp_fn(torch.tensor(1., device=device))

        self.assertEqual(y, 2 * x)
        self.assertFalse(y.requires_grad)
        self.assertTrue(out.requires_grad)

        z, = torch.autograd.grad(out, x)
        self.assertEqual(z, 2 * x)

    def test_no_grad_outside_vjp_only(self, device):
        def h(x):
            return x ** 2

        x = torch.tensor(3.14, requires_grad=True, device=device)
        with torch.no_grad():
            out, vjp_fn = vjp(h, x)
        y, = vjp_fn(torch.tensor(1., device=device))

        self.assertEqual(y, 2 * x)
        self.assertFalse(out.requires_grad)

        # This one is a little weird...
        self.assertTrue(y.requires_grad)

        z, = torch.autograd.grad(y, x)
        self.assertEqual(z, 2)


class TestVmapOfGrad(TestCase):
    def test_per_sample_grads_inplace_view(self, device):
        def compute_loss(weight, x, t):
            x = x.mm(weight)
            y = x.squeeze_(0)
            return (y - t).sum()

        weight = torch.randn(16, 2, device=device)
        x = torch.randn(64, 1, 16, device=device)
        t = torch.randn(64, 2, device=device)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def test_new_zeros_materializes_tensor(self, device):
        N = 3
        C = 5

        def foo(y, x):
            result = x.new_zeros((C,))
            result.copy_(y)
            return result.sum()

        x = torch.randn(N, device=device)
        y = torch.randn(N, C, device=device)
        result = vmap(grad(foo))(y, x)
        self.assertEqual(result, torch.ones_like(y))

    def test_new_empty_materializes_tensor(self, device):
        N = 3
        C = 5

        def foo(y, x):
            result = x.new_empty((C,))
            result.copy_(y)
            return result.sum()

        x = torch.randn(N, device=device)
        y = torch.randn(N, C, device=device)
        result = vmap(grad(foo))(y, x)
        self.assertEqual(result, torch.ones_like(y))

    def test_per_sample_grads_simple(self, device):
        def compute_loss(weight, x, t):
            y = x @ weight
            return ((y - t) ** 2).sum()

        weight = torch.randn(16, 2, device=device)
        x = torch.randn(64, 16, device=device)
        t = torch.randn(64, 2, device=device)
        result = vmap(partial(grad(compute_loss), weight))(x, t)
        expected = [grad(compute_loss)(weight, x[i], t[i]) for i in range(64)]
        expected = torch.stack(expected)
        # TODO: Check if the rtol is a problem
        self.assertEqual(result, expected, atol=0, rtol=5e-4)

    def test_per_sample_grads_embeddingnet(self, device):
        class SampleNet(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.emb = nn.Embedding(vocab_size, 16)
                self.fc1 = nn.Linear(16, 16)
                self.fc2 = nn.Linear(16, 2)

            def forward(self, x):
                x = self.emb(x)
                x = torch.transpose(x, -1, -2)
                x = torch.mean(x, -1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                return x

            def name(self):
                return "SampleNet"

        # Create our inputs...
        vocab_size = 1000
        batch_shape = [64]
        words_per_sentence = 5
        data = torch.randint(0, vocab_size, (*batch_shape, words_per_sentence), device=device)
        targets = torch.randint(0, 1, (*batch_shape,), device=device)

        # Construct our module
        net = SampleNet(vocab_size).to(device=device)
        criterion = nn.CrossEntropyLoss()

        net_func, weights = make_functional(net)

        def compute_loss(weights, data, target):
            output = net_func(weights, data)
            result = criterion(output, target)
            return result

        expected = [grad(compute_loss)(weights, data[i], targets[i]) for i in range(64)]
        expected = zip(*expected)
        expected = tuple(torch.stack(shards) for shards in expected)

        result = vmap(partial(grad(compute_loss), weights))(data, targets)
        for r, e in zip(result, expected):
            # TODO: Check if the rtol is a problem
            self.assertEqual(r, e, atol=0, rtol=1e-3)

    def test_log_softmax(self, device):
        x = torch.randn(3, 5, device=device)
        v = torch.randn(5, device=device)

        def foo(x, v):
            _, vjp_fn = vjp(partial(torch.log_softmax, dim=-1), x)
            return vjp_fn(v)[0]

        result = vmap(foo, (0, None))(x, v)

        v = v.expand_as(x)
        x.requires_grad_()
        output = torch.log_softmax(x, dim=-1)
        output.backward(v)
        self.assertEqual(result, x.grad)


jacrev_and_jacfwd = parametrize("jacapi", [subtest(jacrev, name='jacrev'), subtest(jacfwd, name='jacfwd')])

FIXME_jacrev_only = parametrize("jacapi", [subtest(jacrev, name='jacrev')])


class TestJac(TestCase):
    @jacrev_and_jacfwd
    def test_simple(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = jacapi(torch.sin)(x)
        expected = torch.diagflat(x.cos())
        assert torch.allclose(y, expected)

    @jacrev_and_jacfwd
    def test_simple_not_flat(self, device, jacapi):
        x = torch.randn(2, 3, device=device)
        y = jacapi(torch.sin)(x)
        expected = torch.diagflat(x.view(-1).cos())
        expected = expected.view(2, 3, 2, 3)
        assert torch.allclose(y, expected)

    @FIXME_jacrev_only
    def test_diff_numel(self, device, jacapi):
        x = torch.randn(2, 4, device=device)

        # Tensor[2, 4] -> Tensor[3, 1]
        def f(x):
            return x[0, 1:].unsqueeze(-1)

        y = jacapi(f)(x)
        self.assertEqual(y.shape, (3, 1, 2, 4))

        expected = x.new_zeros(3, 1, 2, 4)
        expected[0, 0, 0, 1] = 1
        expected[1, 0, 0, 2] = 1
        expected[2, 0, 0, 3] = 1
        self.assertEqual(y, expected)

    @FIXME_jacrev_only
    def test_vmap_on_jac_simple(self, device, jacapi):
        x = torch.randn(2, 3, device=device)
        y = vmap(jacapi(torch.sin))(x)
        expected = torch.stack([torch.diagflat(x[i].cos()) for i in range(2)])
        assert torch.allclose(y, expected)

    @FIXME_jacrev_only
    def test_nested_jac_simple(self, device, jacapi):
        def foo(x):
            return x.sin().sum()

        x = torch.randn(3, device=device)
        y = jacapi(jacapi(foo))(x)
        expected = torch.diagflat(-x.sin())
        assert torch.allclose(y, expected)

    @jacrev_and_jacfwd
    def test_multiple_args(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=1)(x, y)
        expected = torch.diagflat(x)
        assert torch.allclose(z, expected)

    @jacrev_and_jacfwd
    def test_multiple_outputs_multiple_argnums(self, device, jacapi):
        def f(x, y):
            return 2 * x + 3 * y, 4 * x + 5 * y

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(f, argnums=(0, 1))(x, y)
        expected_out0_x = torch.diagflat(torch.full_like(x, 2))
        expected_out0_y = torch.diagflat(torch.full_like(y, 3))
        expected_out1_x = torch.diagflat(torch.full_like(x, 4))
        expected_out1_y = torch.diagflat(torch.full_like(y, 5))

        self.assertEqual(len(z), 2)
        self.assertTrue(isinstance(z, tuple))
        self.assertEqual(len(z[0]), 2)
        self.assertTrue(isinstance(z[0], tuple))
        self.assertEqual(z[0][0], expected_out0_x)
        self.assertEqual(z[0][1], expected_out0_y)
        self.assertEqual(z[1][0], expected_out1_x)
        self.assertEqual(z[1][1], expected_out1_y)

    @jacrev_and_jacfwd
    def test_multiple_outputs_single_argnums(self, device, jacapi):
        def f(x, y):
            return 2 * x + 3 * y, 4 * x + 5 * y

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        expected_out0_x = torch.diagflat(torch.full_like(x, 2))
        expected_out1_x = torch.diagflat(torch.full_like(x, 4))

        z = jacapi(f, argnums=0)(x, y)
        self.assertEqual(len(z), 2)
        self.assertTrue(isinstance(z, tuple))
        self.assertEqual(z, (expected_out0_x, expected_out1_x))

        z = jacapi(f, argnums=(0,))(x, y)
        self.assertEqual(len(z), 2)
        self.assertTrue(isinstance(z, tuple))
        self.assertTrue(isinstance(z[0], tuple))
        self.assertEqual(z, ((expected_out0_x,), (expected_out1_x,)))

    @FIXME_jacrev_only
    def test_multiple_outputs_pytree(self, device, jacapi):
        def f(x, y):
            return {'left': 2 * x + 3 * y, 'right': 4 * x + 5 * y}

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(f, argnums=(0, 1))(x, y)
        expected_left_x = torch.diagflat(torch.full_like(x, 2))
        expected_left_y = torch.diagflat(torch.full_like(y, 3))
        expected_right_x = torch.diagflat(torch.full_like(x, 4))
        expected_right_y = torch.diagflat(torch.full_like(y, 5))
        expected = {
            'left': (expected_left_x, expected_left_y),
            'right': (expected_right_x, expected_right_y),
        }
        self.assertTrue(isinstance(z, dict))
        self.assertTrue(isinstance(z['left'], tuple))
        self.assertTrue(isinstance(z['right'], tuple))
        self.assertEqual(z, expected)

    @jacrev_and_jacfwd
    def test_multiple_inputs_pytree(self, device, jacapi):
        def f(a, b, c):
            a0, a1 = a
            return a0 + a1 * 2 + b * 3 + c * 4

        x = torch.randn([], device=device)
        args = ((x, x), x, x)

        result = jacapi(f, argnums=(0, 1, 2))(*args)
        expected = (
            (torch.tensor(1., device=device), torch.tensor(2., device=device)),
            torch.tensor(3., device=device),
            torch.tensor(4., device=device),
        )
        self.assertEqual(result, expected)

        result = jacapi(f, argnums=(0,))(*args)
        expected = ((torch.tensor(1., device=device), torch.tensor(2., device=device)),)
        self.assertEqual(result, expected)

        result = jacapi(f)(*args)
        expected = (torch.tensor(1., device=device), torch.tensor(2., device=device))
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_dimensionality(self, device, jacapi):
        def f(x):
            return x

        x = torch.randn([], device=device)
        result = jacapi(f)(x)
        self.assertEqual(result.dim(), 0)
        self.assertEqual(result, torch.ones_like(x))

        x = torch.randn([1], device=device)
        result = jacapi(f)(x)
        self.assertEqual(result.dim(), 2)
        self.assertEqual(result, x.new_ones(1, 1))

    @FIXME_jacrev_only
    def test_aux_tensor(self, device, jacapi):
        def f(x):
            y = x.clone()
            return y, y.cos()

        x = torch.randn(3, device=device)
        result, aux = jacapi(f, has_aux=True)(x)

        self.assertEqual(result, torch.eye(3, 3, device=device))
        self.assertEqual(aux, x.cos())

    @jacrev_and_jacfwd
    def test_aux_pytree(self, device, jacapi):
        def f(x):
            y = x.clone()
            return y, {'a': y.cos(), 'b': [y.tan()]}

        x = torch.randn(3, device=device)

        result, aux = jacapi(f, has_aux=True)(x)
        self.assertEqual(result, torch.eye(3, 3, device=device))
        _, expected_aux = f(x)
        self.assertEqual(aux, expected_aux)

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(RuntimeError, r"Expected tensors, got unsupported type"):
                _ = jacapi(lambda x: (x, aux), has_aux=True)(x)
            with self.assertRaisesRegex(RuntimeError, r"Expected tensors, got unsupported type"):
                _ = jacapi(lambda x: (x, [x, aux]), has_aux=True)(x)

    @jacrev_and_jacfwd
    def test_outputs_can_any_pytree(self, device, jacapi):
        x = torch.randn(2, 3, device=device)

        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError, r"(vjp|jvp).+: Expected f to be a function that has non-empty output"
            ):
                jacapi(lambda _: output)(x)

        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"(vjp|jvp).+: expected f\(\*primals\) to return only tensors"
            ):
                jacapi(lambda _: output)(x)

        # Check list output
        out = jacapi(lambda x: [x, x.sum()])(x)
        assert isinstance(out, list) and len(out) == 2

        # Check dict output
        out = jacapi(lambda x: {"x": x, "xsum": x.sum()})(x)
        assert isinstance(out, dict) and len(out) == 2 and "xsum" in out

        def composite_output(x):
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        out = jacapi(composite_output)(x)
        assert isinstance(out, list)
        assert isinstance(out[0], tuple) and isinstance(out[0][1], dict)

    @jacrev_and_jacfwd
    def test_multiple_inputs_outputs_pytree(self, device, jacapi):
        def f(a, b, c):
            a0, a1 = a
            return a0 + a1 * 2, {'foo': b * 3 + c * 4}

        x = torch.randn([], device=device)
        zero = torch.zeros([], device=device)
        args = ((x, x), x, x)

        result = jacapi(f)(*args)
        expected = (
            (torch.tensor(1., device=device), torch.tensor(2., device=device)),
            {'foo': (zero, zero)},
        )
        self.assertEqual(result, expected)

        result = jacapi(f, argnums=(0,))(*args)
        expected = (
            ((torch.tensor(1., device=device), torch.tensor(2., device=device)),),
            {'foo': ((zero, zero),)},
        )
        self.assertEqual(result, expected)

        result = jacapi(f, argnums=(0, 1))(*args)
        expected = (
            ((torch.tensor(1., device=device), torch.tensor(2., device=device)), zero),
            {'foo': ((zero, zero), torch.tensor(3., device=device))},
        )
        self.assertEqual(result, expected)

    @FIXME_jacrev_only
    def test_multiple_inputs_outputs_pytree_multidim(self, device, jacapi):
        def f(dct):
            a = dct['a']
            b = dct['b']
            return {'c': a.sin(), 'd': b.cos()}

        x = torch.randn(3, device=device)
        args = ({'a': x, 'b': x},)

        result = jacapi(f)(*args)
        expected = {
            'c': {'a': x.cos().diagflat(), 'b': x.new_zeros(3, 3)},
            'd': {'a': x.new_zeros(3, 3), 'b': -x.sin().diagflat()},
        }
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_unrelated_input(self, device, jacapi):
        def f(x, y):
            return x

        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)

        result = jacapi(f, argnums=(0, 1))(x, y)
        expected0 = torch.eye(6, 6, device=device).view(2, 3, 2, 3)
        expected1 = y.new_zeros(2, 3, 2, 3)
        expected = (expected0, expected1)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_unrelated_output(self, device, jacapi):
        y = torch.randn(2, 3, device=device)

        def f(x):
            return y

        x = torch.randn(2, 3, device=device)

        result = jacapi(f)(x)
        expected = x.new_zeros(2, 3, 2, 3)
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_empty_output(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)

        def f(x, y):
            return ()

        with self.assertRaisesRegex(RuntimeError, 'xpected'):
            jacapi(f)(x, y)

    @jacrev_and_jacfwd
    def test_argnums_tuple(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=(0, 1))(x, y)
        expected0 = torch.diagflat(y)
        expected1 = torch.diagflat(x)
        assert len(z) == 2
        assert torch.allclose(z[0], expected0)
        assert torch.allclose(z[1], expected1)

    @jacrev_and_jacfwd
    def test_argnums_effect_on_return(self, device, jacapi):
        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=(0,))(x, y)
        expected0 = torch.diagflat(y)
        assert isinstance(z, tuple)
        assert len(z) == 1
        assert torch.allclose(z[0], expected0)

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(torch.multiply, argnums=0)(x, y)
        expected0 = torch.diagflat(y)
        assert isinstance(z, torch.Tensor)
        assert torch.allclose(z, expected0)

    @jacrev_and_jacfwd
    def test_argnums_defaults_to_zero(self, device, jacapi):
        def f(x, y):
            return x * 2 + y * 3

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        z = jacapi(f)(x, y)
        expected = torch.diagflat(torch.full_like(x, 2))
        self.assertEqual(z, expected)

    @jacrev_and_jacfwd
    def test_empty_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            jacapi(torch.sin, argnums=())(x)

    @jacrev_and_jacfwd
    def test_out_of_bounds_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "only 1 positional inputs"):
            jacapi(torch.sin, argnums=2)(x)

    @jacrev_and_jacfwd
    def test_negative_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "only 1 positional inputs"):
            jacapi(torch.sin, argnums=-2)(x)

    @jacrev_and_jacfwd
    def test_repeated_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be unique"):
            jacapi(torch.sin, argnums=(0, 0))(x)

    @jacrev_and_jacfwd
    def test_float_argnums(self, device, jacapi):
        x = torch.randn(3, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be int or Tuple"):
            jacapi(torch.sin, argnums=0.0)(x)
        with self.assertRaisesRegex(RuntimeError, "must be int"):
            jacapi(torch.multiply, argnums=(1, 0.0))(x, x)

    def test_hessian_simple(self, device):
        def f(x):
            return x.sin()

        x = torch.randn(3, device=device)
        hessian(f)(x)

    def _test_against_reference(self, f, inputs, jacapi):
        def foo(inputs):
            return f(*inputs)

        expected = torch.autograd.functional.jacobian(f, inputs)
        result = jacapi(foo)(inputs)
        self.assertEqual(result, expected)

    @jacrev_and_jacfwd
    def test_against_reference_simple(self, device, jacapi):
        def f(x):
            return 3 * x ** 2

        x = torch.randn(2, 3, 5, device=device)
        self._test_against_reference(f, (x,), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_multi_input(self, device, jacapi):
        def f(x, y):
            return (x.cos() * x) @ y.sin()

        x = torch.randn(2, 3, device=device)
        y = torch.randn(3, 5, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_multi_input_multi_output(self, device, jacapi):
        def f(x, y):
            return (x * x) @ y, x @ (x.sum(1) * y), y.sum()

        x = torch.randn(5, 3, device=device)
        y = torch.randn(3, 5, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_unrelated_outputs(self, device, jacapi):
        def f(x, y):
            return x, y, x, y

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_zero_dim(self, device, jacapi):
        # zero-dim output
        def f(x, y):
            return x.sum(), y.sum(), x * y

        x = torch.randn(3, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y), jacapi)

        # zero-dim input
        def g(x):
            return torch.stack([x, x, x])

        x = torch.randn([], device=device)
        self._test_against_reference(g, (x,), jacapi)

        # Mixed zero-dim input / zero-dim output
        def h(x, y):
            return y.sum(), x * y

        x = torch.randn([], device=device)
        y = torch.randn(1, device=device)
        self._test_against_reference(h, (x, y), jacapi)

    @jacrev_and_jacfwd
    def test_against_reference_correctness_different_devices(self, device, jacapi):
        def f(x, y):
            return x * y, (x * y).to(device=device)

        x = torch.randn(3)
        y = torch.randn(3)
        self._test_against_reference(f, (x, y), jacapi)


class TestHessian(TestCase):
    def _test_against_reference(self, f, inputs):
        def foo(inputs):
            return f(*inputs)

        expected = torch.autograd.functional.hessian(f, inputs)
        result = hessian(foo)(inputs)
        self.assertEqual(result, expected)

    def test_hessian_vectorize_correctness_simple(self, device):
        def f(x):
            return (3 * x ** 2).sum()

        x = torch.randn(2, 3, 5, device=device)
        self._test_against_reference(f, (x,))

    def test_hessian_vectorize_correctness_multi_input(self, device):
        def f(x, y, z):
            return ((x.relu() * x) @ y.sin() @ z).sum()

        x = torch.randn(2, 3, device=device)
        y = torch.randn(3, 5, device=device)
        z = torch.randn(5, 5, device=device)
        self._test_against_reference(f, (x, y, z))

    def test_hessian_vectorize_correctness_unrelated_outputs(self, device):
        # output unrelated to one input
        def f(x, y):
            return (x ** 2).sum()

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y))

        # output unrelated to all inputs
        def f(x, y):
            return torch.ones([])

        x = torch.randn(2, device=device)
        y = torch.randn(3, device=device)
        self._test_against_reference(f, (x, y))

    def test_jacfwd_different_levels(self, device):
        # Test case from:
        # https://github.com/pytorch/functorch/issues/597
        b = 8
        n = 100
        d = 2
        x1 = torch.randn(b, n, d, device=device)
        x2 = x1
        A = 0.1 * torch.randn(b, d, d, device=device)

        def loss(A, x1, x2):
            x2_hat = (A@(x1.T)).T
            res = x2-x2_hat
            res_sqr = res**2
            return res_sqr.sum()

        hess1 = vmap(jacrev(jacrev(loss)))(A, x1, x2)
        hess2 = vmap(hessian(loss))(A, x1, x2)
        self.assertEqual(hess2, hess1)


class TestJvp(TestCase):
    def test_inplace_on_captures(self, device):
        x = torch.tensor([1., 2., 3.], device=device)
        captured = torch.randn(3, device=device)

        def foo(x):
            captured.copy_(x)
            return (x * captured).sum()

        with self.assertRaisesRegex(RuntimeError, 'mutate a captured Tensor'):
            grad(foo)(x)

    def test_simple(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)
        result = jvp(torch.sin, (x,), (t,))
        expected = (x.sin(), x.cos() * t)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_multiple_inputs(self, device):
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        def f(x, y):
            return x * y

        result = jvp(f, (x, y), (tx, ty))
        expected = (x * y, y * tx + x * ty)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_pytree_inputs(self, device):
        def f(x, y, z):
            a, b = x
            return a + 2 * b + 3 * y + 4 * z

        one = torch.tensor(1., device=device)
        primal_outs, tangent_outs = jvp(f, ((one, one), one, one), ((one, one), one, one))
        self.assertEqual(primal_outs, one * 10)
        self.assertEqual(tangent_outs, one * 10)

    def test_pytree_inputs_error_cases(self, device):
        def f(x):
            return x

        one = torch.tensor(1., device=device)

        with self.assertRaisesRegex(RuntimeError, 'Expected primals to be a tuple'):
            jvp(f, one, one)
        with self.assertRaisesRegex(RuntimeError, 'same python structure'):
            jvp(f, ((one, one), one), (one, one))
        with self.assertRaisesRegex(RuntimeError, 'only contain Tensors'):
            jvp(f, ((one, one), 1), ((one, one), one))
        with self.assertRaisesRegex(RuntimeError, 'only contain Tensors'):
            jvp(f, ((one, one), 1), ((1, one), one))
        with self.assertRaisesRegex(RuntimeError, 'at least one Tensor'):
            jvp(f, ((),), ((),))

    def test_unrelated_input(self, device):
        def f(x, y):
            return x

        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        result = jvp(f, (x, y), (tx, ty))
        expected = (x, tx)
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_unrelated_output(self, device):
        y = torch.randn(2, 3, device=device)

        def f(x):
            return y

        x = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)

        result = jvp(f, (x,), (tx,))
        expected = (y, torch.zeros_like(y))
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_strict_mode(self, device):
        y = torch.randn(2, 3, device=device)

        def f(x):
            return x, y

        x = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)

        with self.assertRaisesRegex(RuntimeError, "strict"):
            jvp(f, (x,), (tx,), strict=True)

    def test_multiple_outputs(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        def f(x):
            return torch.sin(x), torch.cos(x)

        result = jvp(f, (x,), (t,))
        expected = (f(x), (x.cos() * t, -x.sin() * t))
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_multiple_inputs_outputs(self, device):
        x = torch.randn(2, 3, device=device)
        y = torch.randn(2, 3, device=device)
        tx = torch.randn(2, 3, device=device)
        ty = torch.randn(2, 3, device=device)

        def f(x, y):
            return 2 * x + 3 * y, 4 * x + 5 * y

        result = jvp(f, (x, y), (tx, ty))
        expected = (f(x, y), f(tx, ty))
        self.assertTrue(isinstance(result, tuple))
        self.assertEqual(result, expected)

    def test_primals_tangents_length_mismatch(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        msg = "same python structure"
        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(torch.sin, (x,), (t, t))
        with self.assertRaisesRegex(RuntimeError, msg):
            jvp(torch.sin, (x, x), (t, t, t))

    def test_nonempty_primals_and_tangents(self, device):
        with self.assertRaisesRegex(RuntimeError, "at least one Tensor"):
            jvp(torch.sin, (), ())

    def test_inputs_are_tuples_of_tensors(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        with self.assertRaisesRegex(RuntimeError, 'be a tuple'):
            jvp(torch.sin, x, (t,))
        with self.assertRaisesRegex(RuntimeError, 'same python structure'):
            jvp(torch.sin, (x,), t)
        with self.assertRaisesRegex(RuntimeError, 'same python structure'):
            jvp(torch.sin, (x,), [t])
        with self.assertRaisesRegex(RuntimeError, 'only contain Tensors'):
            jvp(torch.sin, (1.,), (t,))
        with self.assertRaisesRegex(RuntimeError, 'only contain Tensors'):
            jvp(torch.sin, (x,), (1.,))

    def test_outputs_can_any_pytree(self, device):
        x = torch.randn(2, 3, device=device)
        t = torch.randn(2, 3, device=device)

        for output in [None, ()]:
            with self.assertRaisesRegex(
                RuntimeError, r"jvp\(f, primals, tangents\): Expected f to be a function that has non-empty output"
            ):
                jvp(lambda _: output, (x,), (t,))

        for output in [1, True, 12.2, "abc"]:
            with self.assertRaisesRegex(
                RuntimeError, r"jvp\(f, primals, tangents\): expected f\(\*primals\) to return only tensors"
            ):
                jvp(lambda _: output, (x,), (t,))

        # Check list output
        out = jvp(lambda x: [x, x.sum()], (x,), (t,))
        for i in range(2):
            assert isinstance(out[i], list) and len(out[i]) == 2

        # Check dict output
        out = jvp(lambda x: {"x": x, "xsum": x.sum()}, (x,), (t,))
        for i in range(2):
            assert isinstance(out[i], dict) and len(out[i]) == 2 and "xsum" in out[i]

        def composite_output(x):
            out = x.sum()
            return [
                (out, {"a": x, "out": [x, out]}),
            ]

        out = jvp(composite_output, (x,), (t,))
        for i in range(2):
            assert isinstance(out[i], list)
            assert isinstance(out[i][0], tuple) and \
                isinstance(out[i][0][1], dict)

    def test_aux_tensor(self, device):

        x = torch.randn(3, device=device)
        t = torch.randn(3, device=device)

        with self.assertRaisesRegex(
            RuntimeError, r'jvp\(f, primals, tangents\): output of function f should be a tuple'
        ):
            jvp(lambda t: [t, t], (x, ), (t, ), has_aux=True)

        with self.assertRaisesRegex(
            RuntimeError, r'jvp\(f, primals, tangents\): output of function f should be a tuple'
        ):
            jvp(lambda t: (t, t + 2, t + 3), (x, ), (t, ), has_aux=True)

        def f(z):
            y = z.sin()
            return y, z.cos()

        out, jvp_out, aux = jvp(f, (x, ), (t, ), has_aux=True)
        self.assertEqual(aux, x.cos())
        self.assertEqual(out, x.sin())
        self.assertEqual(jvp_out, t * x.cos())

    def test_aux_pytree(self, device):
        def f(x):
            y = x.sin()
            return y, {'a': x.cos(), 'b': [x.tan()]}

        x = torch.randn(3, device=device)
        t = torch.randn(3, device=device)

        out, jvp_out, aux = jvp(f, (x, ), (t, ), has_aux=True)
        expected_out, expected_aux = f(x)
        self.assertEqual(out, expected_out)
        self.assertEqual(aux, expected_aux)
        self.assertEqual(jvp_out, t * x.cos())

        for aux in [1, 1.0, "abc"]:
            with self.assertRaisesRegex(RuntimeError, r"Expected tensors, got unsupported type"):
                _ = jvp(lambda x: (x, aux), (x, ), (t, ), has_aux=True)
            with self.assertRaisesRegex(RuntimeError, r"Expected tensors, got unsupported type"):
                _ = jvp(lambda x: (x, [x, aux]), (x, ), (t, ), has_aux=True)

    def test_fwd_grad_enabled(self, device):
        # Tests some private helper functions to enable/disable fwd grad mode
        enabled = functorch._C.get_fwd_grad_enabled()
        self.assertTrue(enabled)

        try:
            functorch._C.set_fwd_grad_enabled(False)
            enabled = functorch._C.get_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            functorch._C.set_fwd_grad_enabled(True)

        enabled = functorch._C.get_fwd_grad_enabled()
        self.assertTrue(enabled)

    def test_autograd_function_disables_fwd_grad(self, device):
        # Sanity check. We don't really assume this anywhere so
        # it's fine if this breaks one day.
        class MySquare(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                enabled = functorch._C.get_fwd_grad_enabled()
                self.assertFalse(enabled)
                return x * x

            @staticmethod
            def backward(ctx, gx):
                return gx

        x = torch.randn(3, requires_grad=True)
        MySquare.apply(x)

    def test_enable_fwd_grad(self, device):
        # Tests a private helper function
        try:
            functorch._C.set_fwd_grad_enabled(False)
            enabled = functorch._C.get_fwd_grad_enabled()
            self.assertFalse(enabled)

            with enable_fwd_grad():
                enabled = functorch._C.get_fwd_grad_enabled()
                self.assertTrue(enabled)

            enabled = functorch._C.get_fwd_grad_enabled()
            self.assertFalse(enabled)
        finally:
            functorch._C.set_fwd_grad_enabled(True)

    def test_disable_fwd_grad_outside(self, device):
        x = torch.randn([], device=device)
        t = torch.ones_like(x)
        with enable_fwd_grad(False):
            _, y = jvp(torch.sin, (x,), (t,))
        self.assertEqual(y, x.cos())

    def test_disable_fwd_grad_inside(self, device):
        def f(x):
            with enable_fwd_grad(False):
                shift = x ** 2
            return x ** 2 - shift

        x = torch.randn([], device=device)
        t = torch.ones_like(x)
        _, y = jvp(f, (x,), (t,))
        self.assertEqual(y, 2 * x)
        _, y = jvp(lambda x: jvp(f, (x,), (t,))[1], (x,), (t,))
        self.assertEqual(y, 2)

    def test_disable_fwd_grad_mixed(self, device):
        def f(x):
            with enable_fwd_grad(False):
                shift = x ** 2
            return x ** 2 - shift

        x = torch.randn([], device=device)
        t = torch.ones_like(x)
        with enable_fwd_grad():
            _, y = jvp(f, (x,), (t,))

        self.assertEqual(y, 2 * x)

    def test_jvp_inside_autograd_function(self, device):
        class MySin(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                t = torch.ones_like(x)
                _, neg_sin_x = jvp(torch.cos, (x,), (t,))
                ctx.save_for_backward(x)
                return -neg_sin_x

            @staticmethod
            def backward(ctx, gx):
                x, = ctx.saved_tensors
                t = torch.ones_like(x)
                _, cos_x = jvp(torch.sin, (x,), (t,))
                return gx * cos_x

        x = torch.randn([], device=device, requires_grad=True)
        y = MySin.apply(x)
        self.assertEqual(y, x.sin())

        gx, = torch.autograd.grad(y, x)
        self.assertEqual(gx, x.cos())

    def test_zerotensor_vmapjvp_interaction(self, device):
        dummy = torch.ones(4, 1)
        x = torch.randn(4, 2)
        x_tangent = torch.randn(2)

        def push_jvp(dummy, x):
            result = jvp(torch.cov, (x,), (x_tangent,))
            return result

        # Should not error
        vmap(vmap(push_jvp, (0, None)))(dummy, x)


class TestCustomFunction(TestCase):
    @unittest.skipIf(IS_WINDOWS, "Prototype of custom_vjp doesn't link on windows")
    @onlyCPU
    def test_basic(self, device):
        called_impl = False
        called_vjp = False

        def my_sin_impl(args):
            x, = args
            nonlocal called_impl
            called_impl = True
            return x.sin(), x

        def my_sin_vjp(args):
            grad_y, result, x = args
            nonlocal called_vjp
            called_vjp = True
            return (grad_y * 3 * x.cos(),)

        def filter_fn(args):
            return args[0]

        my_sin = custom_vjp('my_sin', filter_fn, my_sin_impl, my_sin_vjp)

        x = torch.tensor([1., 2.], requires_grad=True, device=device)

        y = my_sin(x)
        self.assertTrue(called_impl)

        y.sum().backward()
        self.assertTrue(called_vjp)

        assert torch.allclose(x.grad, 3 * x.cos())


class TestComposability(TestCase):
    def test_grad_grad(self, device):
        x = torch.randn([], device=device)
        y = grad(grad(torch.sin))(x)
        self.assertEqual(y, -x.sin())

    def test_grad_vmap(self, device):
        def foo(x):
            y = vmap(torch.sin)(x)
            return y.sum()

        x = torch.randn(3, device=device)
        y = grad(foo)(x)
        self.assertEqual(y, x.cos())

    def test_grad_vjp(self, device):
        x = torch.randn(3, device=device)

        def foo(x):
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)[0].sum()

        y = grad(foo)(x)
        expected = grad(lambda x: (x * x.cos()).sum())(x)
        self.assertEqual(y, expected)

    def test_vmap_grad(self, device):
        x = torch.randn(3, device=device)
        y = vmap(grad(torch.sin))(x)
        self.assertEqual(y, x.cos())

    def test_vmap_vmap(self, device):
        x = torch.randn(2, 3, device=device)
        y = vmap(vmap(torch.sin))(x)
        self.assertEqual(y, x.sin())

    def test_vmap_vjp(self, device):
        x = torch.randn(3, device=device)
        _, vjp_fn = vjp(torch.sin, x)

        def foo(x):
            _, vjp_fn = vjp(torch.sin, x)
            return vjp_fn(x)

        y = vmap(foo)(x)
        self.assertEqual(y, vjp_fn(x))

        # TODO: there's a very interesting error message when the following
        # is on CPU
        xs = torch.randn(5, 3, device=device)
        expected = torch.stack([vjp_fn(x)[0] for x in xs])
        result = vmap(lambda x: vjp_fn(x)[0])(xs)
        self.assertEqual(result, expected)

    def test_vjp_grad(self, device):
        x = torch.randn([], device=device)
        y, vjp_fn = vjp(grad(torch.sin), x)
        self.assertEqual(y, x.cos())

        v = torch.randn([])
        self.assertEqual(vjp_fn(v)[0], -x.sin() * v)

    def test_vjp_vmap(self, device):
        x = torch.randn(3, device=device)
        y, vjp_fn = vjp(vmap(torch.sin), x)
        self.assertEqual(y, x.sin())

        v = torch.randn(3, device=device)
        self.assertEqual(vjp_fn(v)[0], x.cos() * v)

    def test_vjp_vjp(self, device):
        x = torch.randn(3, device=device)
        y, vjp_fn = vjp(torch.sin, x)
        self.assertEqual(y, x.sin())

        y, vjp_fn = vjp(lambda x: vjp_fn(x)[0], x)
        self.assertEqual(y, x * x.cos())

        y = vjp_fn(x)[0]
        # Honestly IDK what the result here is... but at least it runs

    def test_make_fx_vmap(self, device):
        def f(x):
            return torch.sin(x)
        inp = torch.randn(5, 3)
        f = vmap(f)
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(5, 3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_jacrev(self, device):
        def f(x):
            return x.sin().sum()
        inp = torch.randn(3)
        f = jacrev(jacrev(f))
        fx_f = make_fx(f)(inp)
        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_make_fx_vjp(self, device):
        def f(x):
            return torch.sin(x).sum()

        primals = torch.randn(3)
        _, vjp_fn = vjp(f, primals)
        cotangent = torch.randn(())
        fx_f = make_fx(vjp_fn)(cotangent, True, True)
        new_cotangent = torch.randn(())
        self.assertEqual(fx_f(new_cotangent, True, True), vjp_fn(new_cotangent))


class TestMakeFunctional(TestCase):
    @parametrize('disable_autograd_tracking', [True, False])
    def test_disable_autograd_tracking(self, disable_autograd_tracking):
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return x

        mod = Foo()
        _, params = make_functional(mod, disable_autograd_tracking=disable_autograd_tracking)
        self.assertEqual(len(params), 2)
        for param in params:
            self.assertEqual(param.requires_grad, not disable_autograd_tracking)

    def test_parameter_tying(self):
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(3))
                self.linear = nn.Linear(3, 3)
                self.linear.bias = self.bias
                self.linear_tied = self.linear

            def forward(self, x):
                x = self.linear(x)
                x = self.linear_tied(x)
                x = x + self.bias
                return x

        torch.manual_seed(1)
        mod = Foo()
        func, _ = make_functional(mod)

        torch.manual_seed(0)
        mod = Foo()
        _, params = make_functional(mod)
        self.assertEqual(len(params), 2)

        x = torch.randn(2, 3)
        result = func(params, x)
        expected = mod(x)
        self.assertEqual(result, expected)

    def test_buffer_tying(self):
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(3))
                self.linear = nn.Linear(3, 3)
                self.register_buffer('buffer', torch.randn(3))
                self.register_buffer('buffer_tied', self.buffer)

            def forward(self, x):
                x = self.linear(x)
                x = x + self.bias
                x = x + self.buffer
                x = x + self.buffer_tied
                return x

        torch.manual_seed(1)
        mod = Foo()
        func, _, _ = make_functional_with_buffers(mod)

        torch.manual_seed(0)
        mod = Foo()
        _, params, buffers = make_functional_with_buffers(mod)
        self.assertEqual(len(params), 3)
        self.assertEqual(len(buffers), 1)

        x = torch.randn(2, 3)
        result = func(params, buffers, x)
        expected = mod(x)
        self.assertEqual(result, expected)

    @parametrize('disable_autograd_tracking', [True, False])
    def test_with_buffers_disable_autograd_tracking(self, disable_autograd_tracking):
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.register_buffer('buffer', torch.randn(3))

            def forward(self, x):
                x = self.linear(x)
                x = x + self.buffer
                return x

        mod = Foo()
        _, params, buffers = make_functional_with_buffers(mod, disable_autograd_tracking=disable_autograd_tracking)
        self.assertEqual(len(params), 2)
        self.assertEqual(len(buffers), 1)
        for param in params:
            self.assertEqual(param.requires_grad, not disable_autograd_tracking)

    def test_parameter_tying_grad(self):
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.weight = self.linear.weight
                self.bias = self.linear.bias

            def forward(self, x):
                x = self.linear(x)
                x = F.linear(x, self.weight, self.bias)
                return x

        x = torch.randn(2, 3)
        torch.manual_seed(0)
        mod = Foo()
        loss = mod(x).sum()
        expected = torch.autograd.grad(loss, mod.parameters())

        mod = Foo()
        fmod, _, _ = make_functional_with_buffers(mod)
        torch.manual_seed(0)
        mod = Foo()
        _, params, buffers = make_functional_with_buffers(mod)

        def compute_loss(params, buffers, x):
            return fmod(params, buffers, x).sum()

        result = grad(compute_loss)(params, buffers, x)

        self.assertEqual(result, expected)

    def test_parameter_tying_ensemble(self):
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.weight = self.linear.weight
                self.bias = self.linear.bias
                self.register_buffer('buffer', torch.randn(3))
                self.register_buffer('buffer_tied', self.buffer)

            def forward(self, x):
                x = self.linear(x)
                x = F.linear(x, self.weight, self.bias)
                x = x + self.buffer
                x = x + self.buffer_tied
                return x

        num_models = 2
        xs = torch.randn(num_models, 64, 3)
        models = [Foo() for _ in range(num_models)]
        fmodel, _, _ = combine_state_for_ensemble(models)

        torch.manual_seed(0)
        models = [Foo() for _ in range(num_models)]
        _, params, buffers = combine_state_for_ensemble(models)
        result = vmap(fmodel)(params, buffers, xs)

        torch.manual_seed(0)
        models = [Foo() for _ in range(num_models)]
        expected = torch.stack([model(x) for model, x in zip(models, xs)])

        self.assertEqual(result, expected)

    def test_correctness_mnist(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, x):
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, 320)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x)

        x = torch.randn(64, 1, 32, 32)
        torch.manual_seed(301)
        fnet, _ = make_functional(Net())

        torch.manual_seed(0)
        _, params = make_functional(Net())
        result = fnet(params, x)

        torch.manual_seed(0)
        net = Net()
        expected = net(x)

        self.assertEqual(result, expected)

    def test_combine_state_for_ensemble_error(self):
        in_features = 2
        out_features = 2

        models = []
        with self.assertRaisesRegex(RuntimeError, "Expected at least one model"):
            _ = combine_state_for_ensemble(models)

        num_models = 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1].eval()
        with self.assertRaisesRegex(RuntimeError, "same training/eval mode"):
            _ = combine_state_for_ensemble(models)

        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        models[1] = torch.nn.Conv2d(3, 3, (3, 3))
        with self.assertRaisesRegex(RuntimeError, "models to be of the same class"):
            _ = combine_state_for_ensemble(models)

    def test_combine_state_for_ensemble_smoke(self):
        in_features = 2
        out_features = 2
        num_models = 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        _ = combine_state_for_ensemble(models)


class TestExamplesCorrectness(TestCase):
    def test_maml_regression(self, device):
        class ThreeLayerNet(nn.Module):
            def __init__(self):
                super(ThreeLayerNet, self).__init__()
                self.fc1 = nn.Linear(1, 40)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Linear(40, 40)
                self.relu2 = nn.ReLU()
                self.fc3 = nn.Linear(40, 1)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.fc3(x)
                return x

        # TODO: should replace with F.mse_loss
        def mse_loss(x, y):
            return torch.mean((x - y) ** 2)

        net, params = make_functional(ThreeLayerNet().to(device))
        K = 20
        num_tasks = 4
        alpha = 0.1

        def sample_tasks(outer_batch_size, inner_batch_size):
            # Select amplitude and phase for the task
            As = []
            phases = []
            for _ in range(outer_batch_size):
                As.append(np.random.uniform(low=0.1, high=.5))
                phases.append(np.random.uniform(low=0., high=np.pi))

            def get_batch():
                xs, ys = [], []
                for A, phase in zip(As, phases):
                    x = np.random.uniform(low=-5., high=5., size=(inner_batch_size, 1))
                    y = A * np.sin(x + phase)
                    xs.append(x)
                    ys.append(y)
                return torch.tensor(xs, dtype=torch.float, device=device), \
                    torch.tensor(ys, dtype=torch.float, device=device)
            x1, y1 = get_batch()
            x2, y2 = get_batch()
            return x1, y1, x2, y2

        def get_loss_for_task(use_transform, x1, y1, x2, y2):
            def inner_loss(params, x1, y1):
                f = net(params, x1)
                loss = mse_loss(f, y1)
                return loss

            if use_transform:
                grads = grad(inner_loss)(params, x1, y1)
            else:
                loss = inner_loss(params, x1, y1)
                grads = torch.autograd.grad(loss, params, create_graph=True)
            new_params = [(params[i] - alpha*grads[i]) for i in range(len(params))]

            v_f = net(new_params, x2)
            return mse_loss(v_f, y2)

        task = sample_tasks(num_tasks, K)

        # Compute with vmap+grad
        inner_losses = vmap(partial(get_loss_for_task, True))(task[0], task[1], task[2], task[3])
        loss2 = sum(inner_losses)/len(inner_losses)
        result_grads = torch.autograd.grad(loss2, params)

        # Compute without vmap+grad
        inner_losses = [
            get_loss_for_task(False, task[0][i], task[1][i], task[2][i], task[3][i])
            for i in range(num_tasks)
        ]
        loss2 = sum(inner_losses)/len(inner_losses)
        expected_grads = torch.autograd.grad(loss2, params)

        self.assertEqual(result_grads, expected_grads)

    def test_maml_omniglot(self, device):
        # TODO: there appears to be precision issues for float32
        dtype = torch.double

        # TODO: We don't support inplace relu?
        inplace_relu = False
        n_way = 5
        n_inner_iter = 2
        num_tasks = 2

        # real example uses batch norm but it's numerically unstable in the first
        # iteration, when near 0, and won't produce same gradients. Uses group norm instead
        net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.GroupNorm(64, 64, affine=True),
            nn.ReLU(inplace=inplace_relu),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.GroupNorm(64, 64, affine=True),
            nn.ReLU(inplace=inplace_relu),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.GroupNorm(64, 64, affine=True),
            nn.ReLU(inplace=inplace_relu),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, n_way)).to(device).to(dtype)

        fnet, params, buffers = make_functional_with_buffers(net)
        net = (params, buffers, fnet)

        def loss_for_task(net, n_inner_iter, use_transform, x_spt, y_spt, x_qry, y_qry):
            params, buffers, fnet = net
            querysz = x_qry.size(0)

            def compute_loss(new_params, buffers, x, y):
                logits = fnet(new_params, buffers, x)
                loss = F.cross_entropy(logits, y)
                return loss

            new_params = params
            for _ in range(n_inner_iter):
                if use_transform:
                    grads = grad(compute_loss)(new_params, buffers, x_spt, y_spt)
                else:
                    res = compute_loss(new_params, buffers, x_spt, y_spt)
                    grads = torch.autograd.grad(res, new_params, create_graph=True)
                new_params = [p - g * 1e-1 for p, g, in zip(new_params, grads)]

            qry_logits = fnet(new_params, buffers, x_qry)
            qry_loss = F.cross_entropy(qry_logits, y_qry)
            qry_acc = (qry_logits.argmax(
                dim=1) == y_qry).sum() / querysz

            return qry_loss, qry_acc

        # Get some sample inputs...
        x_spt = torch.randn(num_tasks, 25, 1, 28, 28, dtype=dtype, device=device)
        y_spt = torch.randint(0, 5, (num_tasks, 25), device=device)
        x_qry = torch.randn(num_tasks, 75, 1, 28, 28, dtype=dtype, device=device)
        y_qry = torch.randint(0, 5, (num_tasks, 75), device=device)

        # compute with vmap + grad
        compute_loss = partial(loss_for_task, net, n_inner_iter, True)
        qry_losses, _ = vmap(compute_loss)(x_spt, y_spt, x_qry, y_qry)
        result_grads = torch.autograd.grad(qry_losses.sum(), params)

        # compute without vmap + grad
        compute_loss = partial(loss_for_task, net, n_inner_iter, False)
        losses = [compute_loss(x_spt[i], y_spt[i], x_qry[i], y_qry[i])[0]
                  for i in range(num_tasks)]
        expected_grads = torch.autograd.grad(sum(losses), params)

        self.assertEqual(result_grads, expected_grads)

    @parametrize('originally_track_running_stats', [True, False])
    def test_update_batch_norm(self, device, originally_track_running_stats):
        dtype = torch.double
        inplace_relu = False
        classes = 5
        num_batches = 2
        net = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, affine=True, track_running_stats=originally_track_running_stats),
            nn.ReLU(inplace=inplace_relu),
            nn.Flatten(),
            nn.Linear(43264, classes)).to(device).to(dtype)

        replace_all_batch_norm_modules_(net)
        transformed_net = net
        fnet, params, buffers = make_functional_with_buffers(transformed_net)
        net = (params, buffers, fnet)
        criterion = nn.CrossEntropyLoss()

        def compute_loss(x, y, params, buffers):
            return criterion(fnet(params, buffers, x), y)

        # Get some sample inputs...
        x = torch.randn(num_batches, 1, 64, 28, 28, device=device, dtype=dtype)
        y = torch.randint(0, classes, (num_batches, 1), device=device)

        # compute some per sample grads with vmap + grad
        result_grads = vmap(grad(compute_loss, argnums=2), in_dims=(0, 0, None, None))(x, y, params, buffers)

        # compute some per sample grads without vmap + grad
        fnet, params, buffers = make_functional_with_buffers(transformed_net)
        expected_grads = [
            torch.autograd.grad(compute_loss(x[i], y[i], params, buffers), params)
            for i in range(num_batches)
        ]
        expected_grads = [torch.stack(shards) for shards in zip(*expected_grads)]

        self.assertEqual(result_grads, expected_grads)

    @parametrize('jac', ['jacfwd', 'jacrev'])
    def test_lennard_jones_batched_jac(self, device, jac):
        sigma = 0.5
        epsilon = 4.

        jac = getattr(functorch, jac)

        def lennard_jones(r):
            return epsilon * ((sigma / r)**12 - (sigma / r)**6)

        def lennard_jones_force(r):
            """Get magnitude of LJ force"""
            return \
                -epsilon * ((-12 * sigma**12 / r**13) + (6 * sigma**6 / r**7))

        r = torch.linspace(0.5, 2 * sigma, steps=100, requires_grad=True, device=device)
        drs = torch.outer(r, torch.tensor([1.0, 0, 0], device=device))
        norms = torch.norm(drs, dim=1).reshape(-1, 1)
        training_energies = \
            torch.stack(list(map(lennard_jones, norms))).reshape(-1, 1)
        training_forces = torch.stack(
            [force * dr
             for force, dr in zip(map(lennard_jones_force, norms), drs)])

        model = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        ).to(device)

        def make_prediction(model, drs, use_functorch):
            norms = torch.norm(drs, dim=1).reshape(-1, 1)
            energies = model(norms)

            if use_functorch:
                network_derivs = vmap(jac(model))(norms).squeeze(-1)
                forces = -network_derivs * drs / norms
            else:
                forces = []
                for r, dr in zip(norms, drs):
                    network_deriv = torch.autograd.functional.jacobian(
                        model, r, create_graph=True)
                    force = -network_deriv * dr / r
                    forces.append(force)
                forces = torch.cat(forces)
            return energies, forces

        def loss_fn(energies, forces, predicted_energies, predicted_forces):
            return F.mse_loss(energies, predicted_energies) + \
                0.01 * F.mse_loss(forces, predicted_forces) / 3

        energies, forces = make_prediction(model, drs, use_functorch=True)
        loss = loss_fn(training_energies, training_forces, energies, forces)
        result = torch.autograd.grad(loss, model.parameters())

        energies, forces = make_prediction(model, drs, use_functorch=False)
        loss = loss_fn(training_energies, training_forces, energies, forces)
        expected = torch.autograd.grad(loss, model.parameters())

        self.assertEqual(result, expected)

    def test_ensemble_regression(self, device):
        def make_spirals(n_samples, noise_std=0., rotations=1.):
            ts = torch.linspace(0, 1, n_samples)
            rs = ts ** 0.5
            thetas = rs * rotations * 2 * math.pi
            signs = torch.randint(0, 2, (n_samples,)) * 2 - 1
            labels = (signs > 0).to(torch.long)

            xs = rs * signs * torch.cos(thetas) + torch.randn(n_samples) * noise_std
            ys = rs * signs * torch.sin(thetas) + torch.randn(n_samples) * noise_std
            points = torch.stack([xs, ys], dim=1)
            return points.to(device), labels.to(device)

        points, labels = make_spirals(100, noise_std=0.05)

        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.fc1 = nn.Linear(2, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        loss_fn = nn.NLLLoss()

        func_model, weights = make_functional(MLPClassifier().to(device))

        def train_step_fn(use_transform, weights, batch, targets, lr=0.2):
            def compute_loss(weights, batch, targets):
                output = func_model(weights, batch)
                loss = loss_fn(output, targets)
                return loss

            if use_transform:
                grad_weights, loss = grad_and_value(compute_loss)(weights, batch, targets)
            else:
                loss = compute_loss(weights, batch, targets)
                grad_weights = torch.autograd.grad(loss, weights)

            new_weights = []
            with torch.no_grad():
                for grad_weight, weight in zip(grad_weights, weights):
                    new_weights.append(weight - grad_weight * lr)
            # NB: return looks weird because torch.vmap must return Tensors
            return (loss, *new_weights)

        def unpack(train_result):
            return train_result[0], train_result[1:]

        def init_fn(num_models):
            models = tuple(MLPClassifier().to(device) for _ in range(num_models))
            weights = tuple(make_functional(model)[1] for model in models)
            weights = tuple(zip(*weights))
            weights = tuple(torch.stack(shards).detach() for shards in weights)
            return weights

        def slice_weights(batched_weights, index):
            return tuple(weight[index].detach().requires_grad_() for weight in batched_weights)

        batched_weights = init_fn(num_models=2)
        parallel_train_step_fn = vmap(partial(train_step_fn, True), in_dims=(0, None, None))

        result_loss, result_weights = unpack(parallel_train_step_fn(batched_weights, points, labels))

        loss0, weights0 = unpack(train_step_fn(False, slice_weights(batched_weights, 0), points, labels))
        loss1, weights1 = unpack(train_step_fn(False, slice_weights(batched_weights, 1), points, labels))
        expected_loss = torch.stack([loss0, loss1])
        expected_weights = tuple(torch.stack([w0, w1]) for w0, w1 in zip(weights0, weights1))

        self.assertEqual(result_loss, expected_loss)
        self.assertEqual(result_weights, expected_weights)

    @parametrize("dropout_layer", [nn.Dropout, nn.AlphaDropout, nn.FeatureAlphaDropout])
    def test_find_learning_rate_ensembling(self, device, dropout_layer):
        # This example mimics what a user might do when trying to find the optimal learning rate. They would
        # want to run a bunch of models with the same behavior (including the same dropout!) and have them
        # each run with different learning rates. Specifically, this is an example of using same randomness with vmap
        points, labels = torch.randn(100, 2, 2, 2, 2, device=device), torch.randint(0, 2, (100,), device=device)

        class MLPClassifier(nn.Module):
            def __init__(self, hidden_dim=32, n_classes=2):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.n_classes = n_classes

                self.dropout = dropout_layer()
                self.fc1 = nn.Linear(16, self.hidden_dim)
                self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)

            def forward(self, x):
                x = self.dropout(x)
                x = torch.flatten(x, start_dim=1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.fc2(x)
                x = F.log_softmax(x, -1)
                return x

        loss_fn = nn.NLLLoss()

        func_model, weights = make_functional(MLPClassifier().to(device))

        def train_step_fn(weights, batch, targets, lr):
            def compute_loss(weights, batch, targets):
                output = func_model(weights, batch)
                loss = loss_fn(output, targets)
                return loss

            grad_weights, loss = grad_and_value(compute_loss)(weights, batch, targets)
            new_weights = []
            with torch.no_grad():
                for grad_weight, weight in zip(grad_weights, weights):
                    new_weights.append(weight - grad_weight * lr)
            # NB: return looks weird because torch.vmap must return Tensors
            return (loss, *new_weights)

        def unpack(train_result):
            return train_result[0], train_result[1:]

        def init_fn(num_models):
            og_model = MLPClassifier().to(device)
            models = tuple(copy.deepcopy(og_model) for _ in range(num_models))  # have same initialization
            weights = tuple(make_functional(model)[1] for model in models)
            weights = tuple(zip(*weights))
            weights = tuple(torch.stack(shards).detach() for shards in weights)
            return weights

        batched_weights = init_fn(num_models=2)
        parallel_train_step_fn = vmap(train_step_fn, in_dims=(0, None, None, 0), randomness="same")

        lrs = torch.tensor([0.2, 0.4], device=device)
        result_loss, result_weights = unpack(parallel_train_step_fn(batched_weights, points, labels, lrs))

        self.assertEqual(result_loss[0], result_loss[1])
        self.assertNotEqual(tuple(weight[0] for weight in result_weights),
                            tuple(weight[1] for weight in result_weights))

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_resnet18_per_sample_grads(self, device):
        import torchvision.models as models
        model = models.__dict__['resnet18'](
            pretrained=False, norm_layer=(lambda c: nn.GroupNorm(min(32, c), c))
        ).to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')  # avoid cross batch reductions for for loop comparison

        func_model, weights = make_functional(model)

        def compute_loss(weights, image, target):
            output = func_model(weights, images)
            loss = criterion(output, targets)
            return loss

        batch_size = 3
        images = torch.randn(batch_size, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        result_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(weights, images, targets)

        expected_grads = [
            torch.autograd.grad(compute_loss(weights, images[i].unsqueeze(0), targets[i].unsqueeze(0)), weights)
            for i in range(batch_size)
        ]
        expected_grads = [torch.stack(shards) for shards in zip(*expected_grads)]

        self.assertEqual(result_grads, expected_grads, atol=1e-3, rtol=1.)


class TestFunctionalize(TestCase):
    def _check_functionalize_correctness(self, f, inpt):
        inpt1 = inpt.clone()
        inpt2 = inpt.clone()
        inpt3 = inpt.clone()

        expected_outputs = f(inpt1)
        actual_outputs = vmap(functionalize(f))(inpt2.unsqueeze(0))[0].squeeze()
        # Right now the flavor of functionalize that also removes view ops
        # isn't being used with vmap
        # That's because {view}_copy ops don't have batching rules yet
        # (although we should probably fix that)
        actual_outputs_view_copy = functionalize(f, remove='mutations_and_views')(inpt3)
        # Check that outputs are the same
        self.assertEqual(actual_outputs, expected_outputs)
        self.assertEqual(actual_outputs_view_copy, expected_outputs)

        # Inputs might have been mutated by f: check that they were mutated properly
        self.assertEqual(inpt1, inpt2)
        self.assertEqual(inpt1, inpt3)

    def test_simple_view(self, device):

        def f(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.ones(2, device=device)
            y = x.view(4, 2)
            y.add_(tmp)
            return x
        self._check_functionalize_correctness(f, torch.zeros(4, 2, device=device))

    def test_multioutput_view(self, device):

        def f(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.ones(2, device=device)
            y1, y2 = x.split(2)
            y1_view = y1.diagonal()
            y1_view.add_(tmp)
            return x
        self._check_functionalize_correctness(f, torch.zeros(4, 2, device=device))


    def test_inplace_view(self, device):

        def f(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.ones(4, device=device)
            y = x + x
            y2 = y.transpose(1, 0)
            z = y2[0]
            z.add_(tmp)
            return y
        self._check_functionalize_correctness(f, torch.zeros(4, 2, device=device))

    # See https://github.com/pytorch/functorch/issues/780
    def test_linear(self, device):

        def f(x, y, z) -> torch.Tensor:
            return torch._C._nn.linear(x, y, z)

        x = torch.randn(14, 1, 384, device=device)
        y = torch.randn(96, 384, device=device)
        z = torch.randn(96, device=device)

        out_expected = f(x, y, z)
        out_actual = functionalize(f)(x, y, z)
        self.assertEqual(out_expected, out_actual)

    def test_multioutput_inplace_slice_view(self, device):

        def f(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.ones(2, 2, device=device)
            y = x.view(8)
            z0 = y.reshape(2, 4)
            z1 = z0.transpose(1, 0)
            z1.unsqueeze_(0)
            z1.squeeze_()
            z2, z3 = z1.split(2)
            z2.add_(tmp)
            return x
        self._check_functionalize_correctness(f, torch.zeros(4, 2, device=device))

    def test_functionalize_fx_simple(self, device):

        def f(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.ones(2, device=device)
            y = x.view(4, 2)
            y.add_(tmp)
            return x
        # There's a copy_ in the graph, because the input (x) was mutated.
        # To preserve semantics, functionalize() needs to propagate the mutation.
        out = make_fx(functionalize(f, remove='mutations_and_views'))(torch.zeros(4, 2, device=device))
        self.assertExpectedInline((out.code), """\



def forward(self, x_1) -> torch.Tensor:
    view_copy = torch.ops.aten.view_copy(x_1, [4, 2])
    _tensor_constant0 = self._tensor_constant0
    add = torch.ops.aten.add(view_copy, _tensor_constant0);  view_copy = _tensor_constant0 = None
    view_copy_1 = torch.ops.aten.view_copy(add, [4, 2]);  add = None
    copy_ = torch.ops.aten.copy_(x_1, view_copy_1);  x_1 = None
    return view_copy_1
    """)

    def test_functionalize_fx_transpose_simple(self, device):

        def f(x: torch.Tensor) -> torch.Tensor:
            return x.transpose(1, 0)
        out = make_fx(functionalize(f, remove='mutations_and_views'))(torch.zeros(4, 2, device=device))
        self.assertExpectedInline(out.code, """\



def forward(self, x_1) -> torch.Tensor:
    transpose_copy = torch.ops.aten.transpose_copy(x_1, 1, 0);  x_1 = None
    return transpose_copy
    """)

    def test_functionalize_fx_out_op(self, device):

        def f(inpt: torch.Tensor) -> torch.Tensor:
            out = torch.empty((), dtype=torch.float32)
            torch.add(inpt, inpt, out=out)
            out_view = out.view(4)
            out_view.add_(1)
            return out

        fn = make_fx(functionalize(f, remove='mutations_and_views'))
        out = fn(torch.arange(4, device=device, dtype=torch.float32))
        self.assertExpectedInline(out.code, """\



def forward(self, inpt_1) -> torch.Tensor:
    add = torch.ops.aten.add(inpt_1, inpt_1);  inpt_1 = None
    view_copy = torch.ops.aten.view_copy(add, [4])
    view_copy_1 = torch.ops.aten.view_copy(add, [4]);  add = None
    add_1 = torch.ops.aten.add(view_copy_1, 1);  view_copy_1 = None
    view_copy_2 = torch.ops.aten.view_copy(add_1, [4]);  add_1 = None
    return view_copy_2
    """)

    def test_functionalize_fx_multi_out_op(self, device):

        def f(inpt: torch.Tensor) -> torch.Tensor:
            mins = torch.empty(4, dtype=torch.float32)
            maxs = torch.empty(2, 2, dtype=torch.float32)
            maxs_view = maxs.view(4)
            inpt_view = inpt.view(2, 4)
            torch.aminmax(inpt_view, dim=0, out=(mins, maxs_view))
            return (maxs, mins)

        fn = make_fx(functionalize(f, remove='mutations_and_views'))
        out = fn(torch.arange(8, device=device, dtype=torch.float32))
        self.assertExpectedInline(out.code, """\



def forward(self, inpt_1) -> torch.Tensor:
    view_copy = torch.ops.aten.view_copy(inpt_1, [2, 4]);  inpt_1 = None
    aminmax = torch.ops.aten.aminmax(view_copy, dim = 0);  view_copy = None
    getitem = aminmax[0]
    getitem_1 = aminmax[1];  aminmax = None
    view_copy_1 = torch.ops.aten.view_copy(getitem_1, [2, 2]);  getitem_1 = None
    return (view_copy_1, getitem)
    """)

    def test_functionalize_fx_reapply_views_simple(self, device):

        def f(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.ones(2, device=device)
            y = x.view(4, 2)
            y.add_(tmp)
            return x

        out = make_fx(functionalize(f))(torch.zeros(4, 2, device=device))
        self.assertExpectedInline(out.code, """\



def forward(self, x_1) -> torch.Tensor:
    view = torch.ops.aten.view(x_1, [4, 2])
    _tensor_constant0 = self._tensor_constant0
    add = torch.ops.aten.add(view, _tensor_constant0);  view = _tensor_constant0 = None
    view_1 = torch.ops.aten.view(add, [4, 2]);  add = None
    copy_ = torch.ops.aten.copy_(x_1, view_1);  x_1 = None
    return view_1
    """)

    def test_functionalize_nonfunctional_output(self, device):

        global_out = torch.ones(2, device=device)

        def f() -> torch.Tensor:
            return global_out

        out = make_fx(functionalize(f))()
        self.assertExpectedInline(out.code, """\



def forward(self) -> torch.Tensor:
    _tensor_constant0 = self._tensor_constant0
    return _tensor_constant0
    """)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(
    TestGradTransform,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestVmapOfGrad,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestJac,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestJvp,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestHessian,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestComposability,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestExamplesCorrectness,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestCustomFunction,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(
    TestFunctionalize,
    globals(),
    only_for=only_for,
)
instantiate_parametrized_tests(
    TestMakeFunctional,
)

if __name__ == '__main__':
    run_tests()
