from tinygrad.tensor import Tensor


class MeanSquareError:
    def __init__(self, rooted=False):
        self.rooted = rooted

    def __call__(self, prediction: Tensor, truth: Tensor) -> Tensor:
        assert prediction.shape == truth.shape, f"ouput shape {
            prediction.shape}  and truth shape {truth.shape} not equivalent shape"
        return prediction.sub(truth).square().mean() if not self.rooted else prediction.sub(truth).square().mean().sqrt()


class MeanSquareLogError:
    def __init__(self):
        pass

    def __call__(self, prediction: Tensor, truth: Tensor) -> Tensor:
        assert prediction.shape == truth.shape, f"ouput shape {
            prediction.shape}  and truth shape {truth.shape} not equivalent shape"
        return prediction.log().sub(truth.log()).square().mean() if not self.rooted else prediction.log().sub(truth.log()).square().mean().sqrt()


class MeanAbsoluteError:
    def __init__(self):
        pass

    def __call__(self, prediction: Tensor, truth: Tensor) -> Tensor:
        assert prediction.shape == truth.shape, f"ouput shape {
            prediction.shape}  and truth shape {truth.shape} not equivalent shape"
        return prediction.sub(truth).mean()
