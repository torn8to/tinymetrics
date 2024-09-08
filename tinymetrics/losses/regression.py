from tinygrad.tensor import Tensor
from tinymetrics.metrics import Metric



class MeanSquareError(Metric):
    """
    runs both MSE and RMSE 
    """
    def __init__(self, rooted=False):
        """
            
        Arguments
            rooted -- set to True change to RMSE default v. MSE (default False)
        """
        self.rooted = rooted

    def __call__(self, prediction: Tensor, truth: Tensor) -> Tensor:
        assert prediction.shape == truth.shape, f"ouput shape {
            prediction.shape}  and truth shape {truth.shape} not equivalent shape"
        return prediction.sub(truth).square().mean() if not self.rooted else prediction.sub(truth).square().mean().sqrt()


class MeanSquareLogError(Metric):
    def __init__(self):
        pass

    def __call__(self, prediction: Tensor, truth: Tensor) -> Tensor:
        assert prediction.shape == truth.shape, f"ouput shape {
            prediction.shape}  and truth shape {truth.shape} not equivalent shape"
        return (prediction+1).log().sub((truth+1).log()).square().mean()


class MeanAbsoluteError(Metric):
    def __init__(self):
        pass

    def __call__(self, prediction: Tensor, truth: Tensor) -> Tensor:
        assert prediction.shape == truth.shape, f"ouput shape {
            prediction.shape}  and truth shape {truth.shape} not equivalent shape"
        return prediction.sub(truth).abs().mean()


class CosineSimilarity(Metric):
    def __init__(self, matrix_norm=None):
        self.matrix_norm = None

    def __call__(self, prediction:Tensor, truth:Tensor) -> Tensor:
        assert prediction.shape == truth.shape, f"ouput shape {prediction.shape}  and truth shape {truth.shape} not equivalent shape"
        assert len(prediction.shape) ==2, f"minkowski distance does not support anything other than 2d tensors"
        return (prediction.matmul(truth.T)).sum().div(((truth.abs().pow(2).sum().sqrt()) * (prediction.abs().pow(2).sum().sqrt())))

class MinkowskiDistance(Metric):
    def __init__(self, p=2):
        self.p = p

    def __call__(self,prediction:Tensor, truth:Tensor, p = None) -> Tensor:
        if not p == None: self.p = p
        assert prediction.shape == truth.shape, f"ouput shape {prediction.shape}  and truth shape {truth.shape} not equivalent shape"
        assert isinstance(self.p,(float,int, Tensor)) and self.p >= 1, f"p is not of type float int or tensor or > 1 the value of p is {self.p} and type is {type(self.p)} "
        return prediction.sub(truth).abs().pow(self.p).sum().pow(1/self.p)




