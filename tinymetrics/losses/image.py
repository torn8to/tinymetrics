from tinymetrics import Metric
from tinygrad import Tensor
from typing import Union, Sequence

def _gaussian_kernel(kernels_size:int,sigma:float):
    dist = Tensor.arange(star=1-kernel_size,end=1+kernel_size)
    gauss = Tensor.exp(-Tensor.pow(dist.div(sigma),2).div(2))
    return gauss.div(gauss.sum()).unsqueeze(dim=0)

def _gaussian_kernel_2d(channel:int, kernels_size:Union[int, Sequence[int]], sigma: Union[float,Sequence[float]]):
    kernel_x: Tensor
    kernel_y: Tensor
    kernel_x, kernel_y = _gaussian_kernel(kernel_size[0],sigma[0]),_gaussian_kernel(kernel_size[1],sigma[1])
    return kernel_x.transpose().matmul(kernel_y)

class StructuredSimilarityIndexMeasure2d(Metric):
    def __init__(self, kernel_size:int=11, sigma=1.5, k1= 0.01, k2 =0.03, gaussian:bool = True):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2

    def __call__(self, pred: Tensor, target: Tensor):
        #TODO: complete the implemeentation of 2d structured similarity
        assert prediction.shape == target.shape, f"prediction and target tensors have different shape prediction: {prediction.shape} target: {target.shape}"
        assert len(target.shape) >= 5, f"tensors of this shape are unsupported tensor must have only 4 dimensions tensor yours has {len(target.shape)}"
        assert self.kernel_size%2 == 1 and self.kernel_size > 0, f"kernel size is notan odd positive number kernel size is {self.kernel_size}"
        assert self.sigma > 0, f"sigma needs to be greater than 0 sigma: {self.sigma}"
        kernel_size = 2*[kernel_size] if not gaussian else  2*[2*int(3.5*self.sigma +.5)*2+1]
        sigma = 2* [self.sigma] if isinstance(self.sigma,int) else self.sigma
        data_range =  max(preds.max() - preds.min(), target.max() - target.min())
        c1 = (data_range * k1)**2
        c2 = (data_range * k2)**2
        
        pred_pad = preds.pad2d((pad_w,pad_w,pad_h,pad_h))
        target_pad = target.pad2d((pad_w,pad_w,pad_h,pad_h))
        kernel = _gaussian_kernel_2d(preds.size(1),sigma)
        output_list = torch.conv2d( preds.cat(target,preds*preds, targets*targets,preds*targets),kernel,groupps=channel).split([0])
        mu_pred_sq , mu_target_sq, mu_pred_target = output_list[0].pow(2), output_list[1].pow(2), output_list[0] * output_list[1]
        sigma_pred_sq, sigma_target_sq, sigma_pred_target = Tensor.clamp(output_list[2] - mu_pred_sq), Tensor.clamp(output_list[3] - mu_pred_target), output_list[4] - mu_pred_target
        upper, lower  = 2 * sigma_pred_target + c2, sigma_pred_sq + sigma_target_sq + c2
        ssim_idx_padded = ((2 * mu_pred_target + c1)*upper)/((mu_pred_sq + mu_target_sq+ c1) * lower)
        ssim_idx_unpadded = ssim_idx_padded[..., pad_h:-padh, pad_w:-pad_w]
        return ssim_idx_unpadded.reshape(ssim_idx.shape[0],-1).mean()


