from tinygrad.tensor import Tensor
from tinymetrics import Metric

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
        print(prediction.sub(truth).mean())
