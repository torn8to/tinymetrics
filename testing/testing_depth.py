import unittest
from tinygrad.tensor import Tensor, dtypes
from tinygrad.tensor import _to_np_dtype
import torch
import numpy as np
from torchmetrics.regression import MeanSquaredError
import tinymetrics
import time


def prepare_test_op(low, high, shps, vals, forward_only=True):
    if shps is None:
        ts = [torch.tensor(x, requires_grad=(not forward_only)) for x in vals]
    else:
        np.random.seed(0)
        np_data = [np.random.uniform(low=low, high=high, size=size).astype(
            _to_np_dtype(dtypes.default_float)) for size in shps]
        ts = [torch.tensor(data, requires_grad=(not forward_only))
              for data in np_data]
    tst = [Tensor(x.detach().numpy(), requires_grad=(
        not forward_only)) for x in ts]
    return ts, tst


def helper_test_op(shps, torch_fxn, tinygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3,
                   forward_only=False, vals=None, low=-2, high=2):
    ''' This is taken from https://github.com/tinygrad/tinygrad/blob/master/test/test_ops.py'''
    if tinygrad_fxn is None:
        tinygrad_fxn = torch_fxn
    ts, tst = prepare_test_op(low, high, shps, vals, forward_only)

    st = time.monotonic()
    out = torch_fxn(*ts)
    torch_fp = time.monotonic() - st

    st = time.monotonic()
    ret = tinygrad_fxn(*tst).realize()
    tinygrad_fp = time.monotonic() - st

    # move inputs to a different device, test the device of intermediate tensors are correct
    def compare(s, tinygrad_output, torch_output, atol, rtol):
        try:
            assert tinygrad_output.shape == torch_output.shape, f"shape mismatch: tinygrad={
                tinygrad_output.shape} | torch={torch_output.shape}"
            assert tinygrad_output.dtype == torch_output.dtype, f"dtype mismatch: tinygrad={
                tinygrad_output.dtype} | torch={torch_output.dtype}"
            np.testing.assert_allclose(
                tinygrad_output, torch_output, atol=atol, rtol=rtol)
        except Exception as e:
            raise Exception(f"{s} failed shape {tinygrad_output.shape}: {e}")
    compare("forward pass", ret.numpy(),
            out.detach().numpy(), atol=atol, rtol=rtol)


class RMSETest(unittest.TestCase):
    def test_RMSE(self):
        torch_mse = MeanSquaredError(squared=True)
        helper_test_op(torch_mse, tinymetrics.losses.depth_loss.RMSE,
                       vals=np.array([1, 7, 5]))


if __name__ == "__main__":
    unittest.main()
