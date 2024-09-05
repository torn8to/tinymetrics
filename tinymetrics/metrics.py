from abc import ABC,abstractmethod

class Metric(ABC):
    """The base class for all metrics mostly for typing.  
        __init__ and __call__ are are the only overides needed to
        implement your own metric

    Arguments:
        ABC -- _description_
    """    
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self,prediction,target):
        pass
