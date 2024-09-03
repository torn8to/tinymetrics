import unittest
from tinygrad.tensor import Tensor, dtypes
from tinygrad.tensor import _to_np_dtype
import torch
import numpy as np
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogError
from tinymetrics.losses import regression
import time


def metric_helper_test(prediction: np.ndarray, target: np.ndarray, torch_fxn, tiny_fxn,
                       atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3, forwrd_only=True):
    torch_pred, torch_test = torch.from_numpy(
        prediction), torch.from_numpy(target)
    tiny_pred, tiny_test = Tensor(prediction), Tensor(target)

    torch_output = torch_fxn(torch_pred, torch_test)
    tiny_output = tiny_fxn(tiny_pred, tiny_test)

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

    compare("regression forward pass", tiny_output.numpy(),
            torch_output.detach().numpy(), atol, rtol)


class RegressionTests(unittest.TestCase):
    def test_RMSE(self):
        torch_RMSE, tiny_RMSE = MeanSquaredError(squared=False), regression.MeanSquareError(rooted=True)  # noqa 501
        pred_1d, target_1d = np.array([1, 5, 7]), np.array([2, 5, 6])
        pred_2d, target_2d = np.array(
            [[1, 5, 7], [1, 9, 5]]), np.array([[2, 5, 6], [1, 4, 5]])
        pred_3d, target_3d = np.array([[[1, 5, 7], [1, 9, 5]], [[3, 5, 4], [5, 3, 6]]]), np.array(
            [[[2, 5, 6], [1, 4, 5]], [[3, 5, 4], [5, 3, 6]]])
        metric_helper_test(pred_1d, target_1d, lambda x,
                           y: torch_RMSE(x, y), lambda x, y: tiny_RMSE(x, y))
        metric_helper_test(pred_2d, target_2d, lambda x,
                           y: torch_RMSE(x, y), lambda x, y: tiny_RMSE(x, y))
        metric_helper_test(pred_3d, target_3d, lambda x,
                           y: torch_RMSE(x, y), lambda x, y: tiny_RMSE(x, y))

    def test_MSE(self):
        torch_MSE, tiny_MSE = MeanSquaredError(squared=True), regression.MeanSquareError(rooted=False)  # noqa 501
        pred_1d, target_1d = np.array([1, 5, 7]), np.array([2, 5, 6])
        pred_2d, target_2d = np.array(
            [[1, 5, 7], [1, 9, 5]]), np.array([[2, 5, 6], [1, 4, 5]])
        pred_3d, target_3d = np.array([[[1, 5, 7], [1, 9, 5]], [[3, 5, 4], [5, 3, 6]]]), np.array(
            [[[2, 5, 6], [1, 4, 5]], [[3, 5, 4], [5, 3, 6]]])
        metric_helper_test(pred_1d, target_1d, lambda x,
                           y: torch_MSE(x, y), lambda x, y: tiny_MSE(x, y))
        metric_helper_test(pred_2d, target_2d, lambda x,
                           y: torch_MSE(x, y), lambda x, y: tiny_MSE(x, y))
        metric_helper_test(pred_3d, target_3d, lambda x,
                           y: torch_MSE(x, y), lambda x, y: tiny_MSE(x, y))

    def test_MSLE(self):
        torch_MSLE, tiny_MSLE = MeanSquaredLogError(), regression.MeanSquareLogError()  # noqa 501
        pred_1d, target_1d = np.array([1, 5, 7]), np.array([2, 5, 6])
        pred_2d, target_2d = np.array(
            [[1, 5, 7], [1, 9, 5]]), np.array([[2, 5, 6], [1, 4, 5]])
        pred_3d, target_3d = np.array([[[1, 5, 7], [1, 9, 5]], [[3, 5, 4], [5, 3, 6]]]), np.array(
            [[[2, 5, 6], [1, 4, 5]], [[3, 5, 4], [5, 3, 6]]])
        metric_helper_test(pred_1d, target_1d, lambda x,
                           y: torch_MSLE(x, y), lambda x, y: tiny_MSLE(x, y))
        metric_helper_test(pred_2d, target_2d, lambda x,
                           y: torch_MSLE(x, y), lambda x, y: tiny_MSLE(x, y))
        metric_helper_test(pred_3d, target_3d, lambda x,
                           y: torch_MSLE(x, y), lambda x, y: tiny_MSLE(x, y))

    def test_MAE(self):
        torch_MAE, tiny_MAE = MeanAbsoluteError(), regression.MeanAbsoluteError()  # noqa 501
        pred_1d, target_1d = np.array([1, 5, 7]), np.array([2, 5, 6])
        pred_2d, target_2d = np.array(
            [[1, 5, 7], [1, 9, 5]]), np.array([[2, 5, 6], [1, 4, 5]])
        pred_3d, target_3d = np.array([[[1, 5, 7], [1, 9, 5]], [[3, 5, 4], [5, 3, 6]]]), np.array(
            [[[2, 5, 6], [1, 4, 5]], [[3, 5, 4], [5, 3, 6]]])
        metric_helper_test(pred_1d, target_1d, lambda x,
                           y: torch_MAE(x, y), lambda x, y: tiny_MAE(x, y))
        metric_helper_test(pred_2d, target_2d, lambda x,
                           y: torch_MAE(x, y), lambda x, y: tiny_MAE(x, y))
        metric_helper_test(pred_3d, target_3d, lambda x,
                           y: torch_MAE(x, y), lambda x, y: tiny_MAE(x, y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
