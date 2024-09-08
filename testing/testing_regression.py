import unittest
from tinygrad.tensor import Tensor, dtypes
import torch
import numpy as np
import numpy.linalg as LA
from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, MeanSquaredLogError, CosineSimilarity, MinkowskiDistance
from tinymetrics.losses import regression
from typing import Sequence, Optional,Union
from testing_helpers import metric_helper, error_helper

# WARN: Equivalent Tensor Sahpes are not currently tested in this file


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
    
    def test_CosineSimilarity(self):
        torch_CS, tiny_CS =  CosineSimilarity(), regression.CosineSimilarity()
        # no 1d testing as 1 dimensional testing does not need to be compared 
        with self.assertRaises(AssertionError) as cm: error_helper([45], lambda x, y: tiny_CS(x, y))
        with self.assertRaises(AssertionError): error_helper([32,24], lambda x, y: tiny_CS(x, y.T))
        def numpyCS(x:np.ndarray,y:np.ndarray):
            p, t = x.numpy(), y.numpy()
            np_upper = np.dot(p,t.transpose()).sum()
            norm_a, norm_b = LA.matrix_norm(p), LA.matrix_norm(t)
            value = np_upper/(norm_a*norm_b)
            return torch.from_numpy(np.array(value))
            
        metric_helper([32,24],numpyCS, lambda x, y: tiny_CS(x,y))
        with self.assertRaises(AssertionError) as cm: error_helper([45], lambda x, y: tiny_CS(x, y))
        with self.assertRaises(AssertionError) as cm: error_helper([45,32,24], lambda x, y: tiny_CS(x,y))
        with self.assertRaises(AssertionError) as cm: error_helper([42,32,24,12], lambda x, y: tiny_CS(x,y))

    def test_MinkowskiDistance(self):
        torch_MD, tiny_MD =  MinkowskiDistance(3), regression.MinkowskiDistance(p=3)
        wrong_mink =  regression.MinkowskiDistance(p=-0.1)
        with self.assertRaises(AssertionError) as cm: error_helper([42], lambda x, y: wrong_mink(x,y))
        metric_helper([45], lambda x, y: torch_MD(x, y), lambda x, y: tiny_MD(x,y))
        metric_helper([16,24], lambda x, y: torch_MD(x, y), lambda x, y: tiny_MD(x,y))
        metric_helper([45,32,24], lambda x, y: torch_MD(x, y), lambda x, y: tiny_MD(x,y))
        metric_helper([42,32,24,12], lambda x, y: torch_MD(x, y), lambda x, y: tiny_MD(x,y))


    def test_frobenius_norm(self):
        p,t = np.random.rand(86,24), np.random.rand(86,24)
        p_t, t_t = Tensor(p),Tensor(t)
        np_upper = np.dot(p,t.transpose()).sum()
        tiny_upper = p_t.matmul(t_t.transpose()).sum()
        np_mat_norms = LA.matrix_norm(p),LA.matrix_norm(t)
        tiny_mat_norm =  (p_t.abs().pow(2).sum().sqrt()) * t_t.abs().pow(2).sum().sqrt() 
        np.testing.assert_allclose((tiny_upper/tiny_mat_norm).numpy(), np_upper/(np_mat_norms[0]*np_mat_norms[1]))
        #np.testing.assert_allclose(tiny_mat_norm.numpy(),np_mat_norm,atol=1e-6,rtol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
