# tinymetrics
torch metrics but for tinygrad that all there is to it. I found the interface really convienents for logging metrics and creating balanced loss functions whihc i found convienient for rapid expirementation and eeking out some extra performance with the same models

## direct install
    python3 -m pip install git+https://github.com/torn8to/tinymetrics.git

or

    pip install git+https://github.com/torn8to/tinymetrics.git

## usage

    from tinymetrics.losses import regression
    from tinygrad import Tensor
    from numpy as np
    p,gt = Tensor(np.array([1,5,7])), Tensor(np.array([1,5,3]))
    rmse = regression.MeanSquaredError(rooted=True)
    value = rmse(p,gt)


## goals
