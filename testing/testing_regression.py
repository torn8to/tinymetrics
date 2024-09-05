import unittest
from tinygrad.tensor import Tensor, dtypes
import torch
import numpy as np
from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogError
from tinymetrics.losses import regression
from typing import Sequence, Optional
from testing_helpers import metric_helper

class RegressionTests(unittest.TestCase):
    def test_RMSE(self):
        torch_RMSE, tiny_RMSE = MeanSquaredError(squared=False), regression.MeanSquareError(rooted=True)
        metric_helper([21], lambda x,y: torch_RMSE(x, y), lambda x, y: tiny_RMSE(x, y))
        metric_helper([45,62], lambda x,y: torch_RMSE(x, y), lambda x, y: tiny_RMSE(x, y))
        metric_helper([15,83,62,], lambda x,y: torch_RMSE(x, y), lambda x, y: tiny_RMSE(x, y))
        metric_helper([15,83,62,20], lambda x,y: torch_RMSE(x, y), lambda x, y: tiny_RMSE(x, y))

    def test_MSE(self):
        torch_MSE, tiny_MSE = MeanSquaredError(squared=True), regression.MeanSquareError(rooted=False)
        metric_helper([21], lambda x,y: torch_MSE(x, y), lambda x, y: tiny_MSE(x, y))
        metric_helper([45,62], lambda x,y: torch_MSE(x, y), lambda x, y: tiny_MSE(x, y))
        metric_helper([15,83,62], lambda x,y: torch_MSE(x, y), lambda x, y: tiny_MSE(x, y))
        metric_helper([15,83,62,20], lambda x,y: torch_MSE(x, y), lambda x, y: tiny_MSE(x, y))

    def test_MSLE(self):
        torch_MSLE, tiny_MSLE = MeanSquaredLogError(), regression.MeanSquareLogError()
        metric_helper([45], lambda x, y: torch_MSLE(x, y), lambda x, y: tiny_MSLE(x, y))
        metric_helper([30,30], lambda x, y: torch_MSLE(x, y), lambda x, y: tiny_MSLE(x, y))
        metric_helper([25,62,40], lambda x, y: torch_MSLE(x, y), lambda x, y: tiny_MSLE(x, y))
        metric_helper([25,62,40,20], lambda x, y: torch_MSLE(x, y), lambda x, y: tiny_MSLE(x, y))

    def test_MAE(self):
        torch_MAE, tiny_MAE = MeanAbsoluteError(), regression.MeanAbsoluteError()
        metric_helper([45], lambda x, y: torch_MAE(x, y), lambda x, y: tiny_MAE(x, y))
        metric_helper([30,30], lambda x, y: torch_MAE(x, y), lambda x, y: tiny_MAE(x, y))
        metric_helper([25,62,40], lambda x, y: torch_MAE(x, y), lambda x, y: tiny_MAE(x, y))
        metric_helper([25,62,40,20], lambda x, y: torch_MAE(x, y), lambda x, y: tiny_MAE(x, y))


if __name__ == "__main__":
    unittest.main(verbosity=2)
