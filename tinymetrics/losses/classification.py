from tinygrad import Tensor
from tinymetrics import Metric

class BinaryJacardIndicator(Metric):
    def __init__(self):
        pass

    def __call__(self, pred: Tensor, target: Tensor):
        return pred.minimum(target).sum().div(pred.minimum(target))


class MultiClassJacardIndex(Metric):
    # TODO: implement multiclass jacard index
    def __init__(self, num_classes):
        pass

    def __call__(self, pred, targe):
        pass
