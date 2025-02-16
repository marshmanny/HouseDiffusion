import torch
import torch.nn.functional as F
import collections
import sys
import os

import hdif.utils.trainer_utils as trainer_utils
from . import LOSSES_REGISTRY

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from hdif.utils.bicubic import BicubicDownSample

MIN_FLOAT = -1e10
normalize = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
from collections import namedtuple
from torch.nn import (
    Conv2d,
    BatchNorm2d,
    PReLU,
    ReLU,
    Sigmoid,
    MaxPool2d,
    AdaptiveAvgPool2d,
    Sequential,
    Module,
    Dropout,
    Linear,
    BatchNorm1d,
)

from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from collections import OrderedDict

import torch

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class DilatedMask:
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size
        
        cords_x = torch.arange(0, kernel_size).view(1, -1).expand(kernel_size, -1) - kernel_size // 2
        cords_y = cords_x.clone().permute(1, 0)
        self.kernel = torch.as_tensor((cords_x ** 2 + cords_y ** 2) <= (kernel_size // 2) ** 2, dtype=torch.float).view(1, 1, kernel_size, kernel_size).cuda()
        self.kernel /= self.kernel.sum()
    
    def __call__(self, mask):
        smooth_mask = F.conv2d(mask, self.kernel, padding=self.kernel_size // 2)
        return smooth_mask ** 0.25

downsample_512 = BicubicDownSample(factor=2)
downsample_256 = BicubicDownSample(factor=4)
    
def scale_to_256(image):
    if not hasattr(image, 'shape') or not (\
        len(image.shape) == 4 and \
            tuple(image.shape[1:]) in [(3, 256, 256),
                                       (3, 512, 512),
                                       (3, 1024, 1024)]):
        return image
    
    if image.shape[-1] == 1024:
        return downsample_256(image)
    elif image.shape[-1] == 512:
        return downsample_512(image)
    else:
        return image
    
def batch_scale_to_256(args, kwargs):
    new_args = list(map(scale_to_256, args))
    new_kwargs = {}
    
    for k, v in kwargs.items():
        new_kwargs[k] = scale_to_256(v)
    
    return new_args, new_kwargs

def scale_input(func):
    def wrapper(*args, **kwargs):
        new_args, new_kwargs = batch_scale_to_256(args, kwargs)
        return func(*new_args, **new_kwargs)
    return wrapper


@LOSSES_REGISTRY.add_to_registry("clip")
@scale_input
def clip_loss(x, y, clip_model, mask=None):
    if mask is not None:
        x = x * mask
        y = y * mask
    gen_embed = clip_model.get_image_embed(x)
    gt_embed = clip_model.get_image_embed(y)
    loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()
    return loss

@LOSSES_REGISTRY.add_to_registry("l2_loss")
def l2_loss(x, y):
    l2_loss = F.mse_loss
    return l2_loss(x, y)

@LOSSES_REGISTRY.add_to_registry("l2_pad_w_loss")
def l2_pad_w_loss(x, y, padding_mask, weight1, weight2):
    with open("exptxt/PAD_MASK.txt", "w") as log_file:
        log_file.write(f"shape: {padding_mask.shape}\n")
            
    out = (x - y) ** 2  
    out *= weight1
    out = out * padding_mask.unsqueeze(1)
    out = out.mean(dim=list(range(1, len(out.shape))))/torch.sum(padding_mask, dim=1)
    out *= weight2
    return out.mean()

@LOSSES_REGISTRY.add_to_registry("cos_loss")
def cos_loss(x, y):
    cos_loss = F.cosine_similarity
    return 1 - torch.mean(cos_loss(x, y, dim=-1))

# @LOSSES_REGISTRY.add_to_registry("argface_class")
# class ArgFace(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.arc_face = iresnet100()
#         self.arc_face.load_state_dict(torch.load("pretrained_models/ArcFace/backbone_r100.pth"))
#         self.arc_face.eval().cuda()
#         self.toArcface = T.Compose([
#             T.Resize((112, 112)),
#             T.Normalize(0.5, 0.5)
#         ])
#         trainer_utils.toggle_grad(self.arc_face, False)
    
#     def to_pil_norm(self, I):
#         return ((I + 1) / 2).clip(0, 1)

#     def forward(self, x, y, to_pil_norm=True):
#         if to_pil_norm:
#             x = self.to_pil_norm(x)
#             y = self.to_pil_norm(y)
        
#         x_embed = self.arc_face(self.toArcface(x))
#         y_embed = self.arc_face(self.toArcface(y))
#         arc_face_loss = (1 - F.cosine_similarity(x_embed, y_embed)).mean()
#         return arc_face_loss

# @LOSSES_REGISTRY.add_to_registry("argface", init=True)
# def argface_loss():
#     argface_model = LOSSES_REGISTRY["argface_class"]().cuda().eval()
    
#     @scale_input
#     def argface_loss_f(*args, **kwargs):
#         return argface_model(*args, **kwargs)
#     return argface_loss_f


class CombinedLoss:
    def __init__(self, losses):
        self.losses = losses
        self.loss_funcs = set()
        self.ema_coefs = dict()
        for g_name, g_losses in self.losses.items():
            for l_name, l_params in g_losses.items():
                loss_name = f"{g_name}_{l_name}"
                if l_params.get('ema_coef', False):
                    self.ema_coefs[loss_name] = l_params['ema_coef']
        
        self.ema = trainer_utils.EMALoss(self.ema_coefs)

    def __call__(
        self, data_dict, train_iter, models, stage
    ):
        losses = collections.defaultdict(float)
        for g_name, g_losses in self.losses.items():
            for l_name, l_params in g_losses.items():
                if train_iter % l_params.get("every_iter", 1) != 0 \
                    or (train_iter < l_params.get("after_iter", MIN_FLOAT)):
                    continue
                
                if l_params.func in LOSSES_REGISTRY.classes:
                    self.loss_funcs.add(l_params.func)
                    loss_name = f"{g_name}_{l_name}"
                    inputs = {k: data_dict.get(v, getattr(models, v, v)) for k, v in l_params.input_map.items()}
                    with open("exptxt/inputs_shapes.txt", "w") as log_file:
                        log_file.write("INPUTS SHAPES:\n")
                        for key, value in inputs.items():
                            if isinstance(value, torch.Tensor):
                                log_file.write(f"{key} shape: {value.shape}\n")
                            else:
                                log_file.write(f"{key} is not a tensor (type: {type(value)})\n")
                    losses[loss_name] = LOSSES_REGISTRY.classes[l_params.func](**inputs)
                    losses[loss_name] = l_params.coef * losses[loss_name]
                    
                    if l_params.get('ema_coef', False) and stage == "train":
                        losses[loss_name] = self.ema.calc_loss(loss_name, losses[loss_name])
                        self.ema.update(loss_name, losses[loss_name].item())
                    
                    losses[f"agg/{l_params.func}"] = losses.get(f"agg/{l_params.func}", 0) + losses[loss_name]
                    losses["total"] += losses[loss_name]
                
                else:
                    raise NotImplementedError(f"{l_params.func} loss is not implemented!")
        return losses
