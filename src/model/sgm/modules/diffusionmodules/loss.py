from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from .metric import EarthnetX_Metric
import numpy as np

from ...util import append_dims, instantiate_from_config
from einops import rearrange



class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        noise_image_num,
        past_weight,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)
        self.noise_image_num = noise_image_num
        self.past_weight = past_weight

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            from taming.modules.losses.lpips import LPIPS

            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

        self.metric_rgbn = EarthnetX_Metric()
        self.metric_ndvi = EarthnetX_Metric()
        self.metric_upper_limit = EarthnetX_Metric()



    def __call__(self, network, denoiser, conditioner, input, batch, input_future, x_past_mean, decode_first_stage,
                 valid_mask=None):
        """
        valid_mask: None = the loss runs on every pixel. Otherwise (b, t, 1, H, W) with
        1 = valid pixel of the target frames (the cloud mask of the ENS): only the
        valid pixels count in the loss.
        """

        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device, dtype=torch.bfloat16)
        input = input.to(input.device, dtype=torch.bfloat16)
        # print(f'sigma:{sigmas}')
        without_noise_input, with_noise_input = input[:,:-self.noise_image_num, ...], input[:,-self.noise_image_num:, ...]
        latent_space_future = with_noise_input.clone()  # b,t,c,h,w


        x_past_mean = x_past_mean.unsqueeze(1).repeat(1,self.noise_image_num,1,1,1)
        # with_noise_input = with_noise_input - x_past_mean  # b,t2,c,h,w
        # without_noise_input = without_noise_input - x_past_mean[:,:without_noise_input.shape[1],...]  # b,t1,c,h,w

        noise = torch.randn_like(with_noise_input).to(input.device, dtype=torch.bfloat16)
        noise = self.past_weight*x_past_mean + (1 - self.past_weight)*noise

        # noise = torch.randn_like(with_noise_input).to(input.device, dtype=torch.bfloat16)
        # x_past_mean = x_past_mean.unsqueeze(1)
        # without_noise_input = torch.cat([without_noise_input, x_past_mean.repeat(1,without_noise_input.shape[1],1,1,1)], dim=2)  # b,t,c,h,w




        # offset_noise_level= 0.0
        if self.offset_noise_level > 0.0:
            noise = noise + self.offset_noise_level * append_dims(
                torch.randn(input.shape[0], device=input.device), input.ndim
            )
        with_noise_input_added = with_noise_input + noise * append_dims(sigmas, with_noise_input.ndim)

        # with_noise_input_added = torch.cat([with_noise_input_added, x_past_mean.repeat(1,with_noise_input_added.shape[1],1,1,1)], dim=2)  # b,t,c,h,w

        model_output = denoiser(
            network, without_noise_input, with_noise_input_added, sigmas, cond, **additional_model_inputs
        ).to(dtype=torch.bfloat16)  # b,t2,c,h,w
        # model_output = rearrange(model_output.to(dtype=torch.bfloat16), "b (t c) h w -> b t c h w", t=self.noise_image_num).contiguous()  
        # print(f'model_output: {model_output[0][0][0]}')

        # # bm,tm,cm,hm,wm = x_past_mean.shape
        # # x_past_mean = x_past_mean.view(bm,tm*cm,hm,wm)
        # model_output = model_output + x_past_mean  # b,t2,c,h,w 

        noise_added_output = with_noise_input_added
        w = append_dims(denoiser.w(sigmas), input.ndim)


        # print(f'model_output:{model_output}')
        
        latent_mask = self.latent_valid_mask(valid_mask, latent_space_future)
        loss, true_latent_rmse = self.get_latent_space_loss(model_output, latent_space_future, w, latent_mask)
        loss_mean, true_latent_rmse_mean = self.get_latent_space_loss(x_past_mean, latent_space_future, w, latent_mask)

        # loss_latent_space, true_latent_rmse, min_latent_rmse, max_latent_rmse = self.get_latent_space_loss(model_output, latent_space_future, w)
        # mean_loss_latent_space, mean_true_latent_rmse, mean_min_latent_rmse, mean_max_latent_rmse = self.get_latent_space_loss(x_past_mean, latent_space_future, w)
        # loss_latent_space = self.get_latent_space_loss(model_output, x_past_mean, w)

        # loss = torch.mean(loss_latent_space)

        # model_output_decode = decode_first_stage(rearrange(model_output, "b t c h w -> (b t) c h w").contiguous())
        # model_output_decode = rearrange(model_output_decode, "(b t) c h w -> b t c h w", t=self.noise_image_num).contiguous()

        # model_output_decode = torch.clamp(model_output_decode, -1.0, 1.0)
        # input_future = torch.clamp(input_future, -1.0, 1.0)
        # mask = batch["mask"][:, -self.noise_image_num:, ...].repeat(1,1,input_future.shape[2],1,1)  # b,t,c,h,w  # b,t,4,h,w
        # loss_decode_space, true_decode_rmse = self.get_loss(model_output_decode, input_future, w, mask)
        # true_latent_rmse = torch.sqrt(torch.mean((model_output-latent_space_future)**2))  # 记录latent space的rmse，方便和在latent space计算损失的实验对比
        # loss = torch.mean(loss_decode_space)
        # print(f'true_decode_rmse:{true_decode_rmse.item()}')

        model_output_sample = model_output[:2,0,...]  # b,c,h,w
        latent_space_future_sample = latent_space_future[:2,0,...]  # b,c,h,w
        noise_added_output_sample = noise_added_output[:2,0,...]
        with torch.no_grad():
            model_output_sample = decode_first_stage(model_output_sample)[0]  # c,h,w
            latent_space_future_sample = decode_first_stage(latent_space_future_sample)[0]  # c,h,w
            noise_added_output_sample = decode_first_stage(noise_added_output_sample)[0]  # c,h,w


        model_output_sample = torch.clamp(model_output_sample, -1.0, 1.0)
        latent_space_future_sample = torch.clamp(latent_space_future_sample, -1.0, 1.0)
        noise_added_output_sample = torch.clamp(noise_added_output_sample, -1.0, 1.0)

                
        def denormalize(array):
            return (array+1.0)/2
        model_output_sample = denormalize(model_output_sample).detach().to(dtype=torch.float).cpu().numpy()
        latent_space_future_sample = denormalize(latent_space_future_sample).detach().to(dtype=torch.float).cpu().numpy()
        noise_added_output_sample = denormalize(noise_added_output_sample).detach().to(dtype=torch.float).cpu().numpy()
        
        return loss, true_latent_rmse, true_latent_rmse_mean, model_output_sample, latent_space_future_sample, noise_added_output_sample
        # return loss, true_latent_rmse, min_latent_rmse, max_latent_rmse, model_output_sample, latent_space_future_sample, noise_added_output_sample, w, mean_true_latent_rmse, mean_min_latent_rmse, mean_max_latent_rmse



    def get_loss(self, pred, target, w, mask):
        # the shape of pred, target, mask and w should be b,t,c,h,w
        assert pred.numel() == mask.numel(), 'size of pred not equal to mask'
        assert target.numel() == mask.numel(), 'size of target not equal to mask'
        
        # 仅考虑掩膜图像值为1的部分
        prediction = pred * mask
        target = target * mask
        
        # 计算均方根误差
        num_valid = torch.sum(mask.reshape(target.shape[0], -1), dim=-1)
        mse = torch.sum((w*((prediction-target)**2)).reshape(target.shape[0], -1), dim=-1)/(num_valid + 1e-8)
        true_rmse = torch.sqrt(torch.sum(((prediction-target)**2))/(torch.sum(mask) + 1e-8))
        # mse = torch.mean(mse)

        # 默认是l2, mse
        return mse, true_rmse
    
    @staticmethod
    def latent_valid_mask(valid_mask, latent):
        """
        Pixel mask (b,t,1,H,W) -> weights (b,t,1,h,w) at the resolution of the latent
        (b,t,c,h,w): every latent cell weighs the fraction of its pixels that are valid
        (area interpolation, as the images are resized). None stays None.
        """
        if valid_mask is None:
            return None
        b, t, _, h, w = latent.shape
        mask = valid_mask.to(device=latent.device, dtype=torch.float32)
        assert mask.shape[:2] == (b, t), f'mask {tuple(mask.shape)} does not match the latent {tuple(latent.shape)}'
        if tuple(mask.shape[-2:]) != (h, w):
            mask = F.interpolate(mask.flatten(0, 1), size=(h, w), mode="area").unflatten(0, (b, t))
        return mask

    def get_latent_space_loss(self, pred, target, w, mask=None):
        # the shape of pred, target and w should be b,t,c,h,w
        # the first shape of all inputs are b
        # mask: None (every pixel) or (b,t,1,h,w) weights, see latent_valid_mask
        assert pred.numel() == target.numel(), f'size of pred not equal to target: pred:{pred.numel()}, target:{target.numel()}'     

        if mask is None:
            mse = torch.mean(
                (w * (pred - target) ** 2).reshape(target.shape[0], -1), 1
                )  # b
            mse_loss = torch.mean(mse)
            true_rmse = torch.sqrt(torch.mean((pred - target) ** 2))
            angle_loss = self.latent_vector_angle_difference(pred, target)
        else:
            m = mask.float().expand_as(target)
            sq_err = ((pred - target) ** 2).float()
            # per cube: weighted mean over its valid cells (a cube without any gives 0)
            mse = torch.sum((w * m * sq_err).reshape(target.shape[0], -1), 1) \
                / (torch.sum(m.reshape(target.shape[0], -1), 1) + 1e-8)  # b
            mse_loss = torch.mean(mse)
            true_rmse = torch.sqrt(torch.sum(m * sq_err) / (torch.sum(m) + 1e-8))
            angle_loss = self.latent_vector_angle_difference(pred, target, mask)
        loss = mse_loss + angle_loss
        
        # # 计算均方根误差
        # mse = torch.mean(w*((prediction-target)**2), dim=1)  # b
        # true_mse = torch.mean(w*((prediction-target)**2), dim=1)  # b
        # min_latent_rmse = torch.sqrt(torch.min(true_mse))
        # max_latent_rmse = torch.sqrt(torch.max(true_mse))
        # # mse = torch.mean(mse)
        # true_rmse = torch.sqrt(torch.mean((prediction-target)**2))
        # # mean_square = (prediction-target)**2
        # # if torch.max(w)>1000:
        # #     print(mean_square)
        # #     print(w)
        # #     print(torch.mean(w*((prediction-target)**2), dim=1))

        # 默认是l2, mse
        # return mse, true_rmse, min_latent_rmse, max_latent_rmse
        return loss, true_rmse


    def latent_vector_angle_difference(self, pred, target, mask=None):
        # 输入图像形状应该为b,t,c,h,w; mask: None or (b,t,1,h,w) weights
        pred_flat = rearrange(pred, "b t c h w -> (b t h w) c").contiguous()
        target_flat = rearrange(target, "b t c h w -> (b t h w) c").contiguous()

        theta = torch.nn.functional.cosine_similarity(pred_flat, target_flat)  # 取值范围为-1到1
        theta = 1 - theta  # 取值范围为0到2
        if mask is None:
            theta = torch.mean(theta)
        else:
            # in float32: the sum of the weights runs up to b*t*h*w (~1e4 cells) and in bf16
            # (8 bits of mantissa) it would be rounded by up to ~0.4%
            m = rearrange(mask, "b t c h w -> (b t h w c)").float()
            theta = torch.sum(theta.float() * m) / (torch.sum(m) + 1e-8)
        
        return theta

