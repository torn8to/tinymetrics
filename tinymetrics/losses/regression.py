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
        return (prediction+1).log().sub((truth+1).log()).square().mean()


class MeanAbsoluteError:
    def __init__(self):
        pass

    def __call__(self, prediction: Tensor, truth: Tensor) -> Tensor:
        assert prediction.shape == truth.shape, f"ouput shape {
            prediction.shape}  and truth shape {truth.shape} not equivalent shape"
        return prediction.sub(truth).abs().mean()
