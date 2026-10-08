from torchmetrics import Metric
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import xarray as xr
import properscoring as ps
from einops import rearrange


class EarthnetX_Metric(torch.nn.Module):
    def __init__(
        self, rgbn=True
    ):
        super().__init__()
        self.metrics = {
            'num_valid_pixel': 0, 
            'sum_squared_error': 0,
            'num_batch_ssim': 0,
            'batch_ssim': 0,
        }
        self.ssim_rgbn_module = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ssim_index_module = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ssim_scale_factor = 1

        self.rgbn = rgbn


    def reset(self):
        for k in self.metrics.keys():
            self.metrics[k] = 0

    def update(self, pred, target, mask):
        results_metrics = self.compute_batch(pred, target, mask)
        for k in self.metrics.keys():
            self.metrics[k] += results_metrics[k]
        results = {
            'rmse': (results_metrics['sum_squared_error']/(results_metrics['num_valid_pixel']+1e-8))**0.5,
            'ssim': results_metrics['batch_ssim']/(results_metrics['num_batch_ssim']+1e-8),
        }

        return results
    
    def calculate_epoch_metrics(self):
        results = {
            'rmse': (self.metrics['sum_squared_error']/(self.metrics['num_valid_pixel']+1e-8))**0.5,
            'ssim': self.metrics['batch_ssim']/(self.metrics['num_batch_ssim']+1e-8),
        }
        return results

    def compute_batch(self, pred, target, mask):
        assert pred.numel() == mask.numel(), 'size of pred not equal to mask'
        assert target.numel() == mask.numel(), 'size of target not equal to mask'
        
        prediction = pred * mask
        target = target * mask
        
        num_valid_pixel = torch.sum(mask).item()
        sum_squared_error = torch.sum((prediction-target)**2).item()

        if self.rgbn:
            prediction = rearrange(prediction, "b t c h w -> (b t) c h w").contiguous()
            target = rearrange(target, "b t c h w -> (b t) c h w").contiguous()
            batch_ssim = self.ssim_rgbn_module(prediction, target)**self.ssim_scale_factor  # b*t
            batch_ssim = torch.sum(batch_ssim).item()
        else:
            prediction = rearrange(prediction, "b t h w -> (b t) h w").contiguous().unsqueeze(1)
            target = rearrange(target, "b t h w -> (b t) h w").contiguous().unsqueeze(1)  # b*t,1,h,w
            batch_ssim = self.ssim_index_module(prediction, target)**self.ssim_scale_factor  # b*t
            batch_ssim = torch.sum(batch_ssim).item()

        metrics = {
            'num_valid_pixel': num_valid_pixel, 
            'sum_squared_error': sum_squared_error,
            'num_batch_ssim': 1,
            'batch_ssim': batch_ssim,
        }

        return metrics




