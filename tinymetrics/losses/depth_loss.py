from tinygrad.tensor import Tensor


def RMSE(output: Tensor, truth: Tensor) -> Tensor:
    assert output.shape == truth.shape, "ouput and truth not equivalent shape"
    return output.subtract(truth).mean().sqrt()


def ABSOLUTE_ERROR(preds: Tensor, truth: Tensor):
    assert preds.shape == truth.shape, "ouput and ground_truth not equivalent shape"
    return preds.subtract(truth).sum()
