import collections
import torch
import inspect
import omegaconf
import dataclasses
import typing
from PIL import Image, ImageDraw, ImageFont


class EMALoss:
    def __init__(self, weights: dict, alpha=0.02):
        self.alpha = alpha
        self.weights = weights
        self.vals = {}

    def update(self, key, val):
        self.vals[key] = self.alpha * val + (1 - self.alpha) * self.vals.get(key, val)

    def calc_loss(self, key, val):
        loss = self.weights.get(key, 1) * val / self.vals.get(key, 1)
        return loss


def add_caption(image, captions, image_size, font, font_size, type="header"):
    SHAPE = image_size
    grid = image
    font = ImageFont.truetype(font, font_size) 

    draw = ImageDraw.Draw(grid)
    W_, H_ = grid.size
    ws, hw = zip(*[draw.textbbox((0, 0), name, font=font)[-2:] for name in captions])
    
    if type == "header":
        H_TEXT = max(hw) + 5 // (1024 // SHAPE)
        new_img = Image.new('RGB', (W_, H_ + H_TEXT), (255, 255, 255))

        draw = ImageDraw.Draw(new_img)
        W_, H_ = new_img.size
        new_img.paste(grid, box=(0, H_TEXT))

        for i, name in enumerate(captions):
            draw.text((SHAPE * i + (SHAPE-ws[i])/2, 0), name, font=font, fill='black')
    elif type == "image":
        new_img = Image.new('RGB', (W_, H_), (255, 255, 255))

        draw = ImageDraw.Draw(new_img)
        W_, H_ = new_img.size
        new_img.paste(grid, box=(0, 0))

        for i, name in enumerate(captions):
            draw.text((SHAPE * i + SHAPE-ws[i], 0), name, font=font, fill='yellow')
    return new_img


def toggle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()


def move_to_device(x, device):
    if isinstance(x, (dict)):
        x =  {k: v.to(device) if k != "index" else v for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        x = x.__class__(move_to_device(t, device) for t in x)
    else:
        x = x.to(device)
    return x


def torch_cov(m, rowvar=False):
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def torch_sqrtm_newton(a, num_iters=50, max_iters=10, eps=1e-6):
    """
    Credits:
        https://github.com/msubhransu/matrix-sqrt
    """
    a_sqrt = None
    for _ in range(max_iters):
        a_norm = torch.norm(a)
        y = a / a_norm
        eye = torch.eye(*a.size(), out=torch.empty_like(a))
        z = torch.eye(*a.size(), out=torch.empty_like(a))
        for i in range(num_iters):
            t = 0.5 * (3.0 * eye - z.mm(y))
            y = y.mm(t)
            z = t.mm(z)
        a_sqrt = y * torch.sqrt(a_norm)
        if torch.isfinite(a_sqrt).all():
            break
        a = a + eps * torch.eye(*a.size(), out=torch.empty_like(a))
        eps *= 10
    return a_sqrt


def accuracy(real_logits=None, fake_logits=None):
    if real_logits is None and fake_logits is None:
        raise ValueError("at least one of the logits should be not None")

    real_acc = real_logits.ge(0).float().mean() if real_logits is not None else 0.0
    fake_acc = fake_logits.le(0).float().mean() if fake_logits is not None else 0.0
    if real_logits is None or fake_logits is None:
        return real_acc + fake_acc
    return (real_acc + fake_acc) / 2


class ClassRegistry:
    def __init__(self):
        self.classes = dict()
        self.args = dict()
        self.arg_keys = None

    def __getitem__(self, item):
        return self.classes[item]

    def make_dataclass_from_init(self, func, name, arg_keys, stop_args):
        args = inspect.signature(func).parameters
        args = [
            (k, typing.Any, omegaconf.MISSING)
            if v.default is inspect.Parameter.empty
            else (k, typing.Optional[typing.Any], None)
            if v.default is None
            else (
                k,
                type(v.default),
                dataclasses.field(default=v.default),
            )
            for k, v in args.items()
        ]
        args = [arg for arg in args if arg[0] not in stop_args]
        if arg_keys:
            self.arg_keys = arg_keys
            arg_classes = dict()
            for key in arg_keys:
                arg_classes[key] = dataclasses.make_dataclass(key, args)
            return dataclasses.make_dataclass(
                name,
                [
                    (k, v, dataclasses.field(default=v()))
                    for k, v in arg_classes.items()
                ],
            )
        return dataclasses.make_dataclass(name, args)

    def make_dataclass_from_classes(self, name):
        return dataclasses.make_dataclass(
            name,
            [(k, v, dataclasses.field(default=v())) for k, v in self.classes.items()],
        )

    def make_dataclass_from_args(self, name):
        return dataclasses.make_dataclass(
            name,
            [(k, v, dataclasses.field(default=v())) for k, v in self.args.items()],
        )

    def add_to_registry(
        self, name, arg_keys=None, init=False, stop_args=("self", "args", "kwargs")
    ):
        def add_class_by_name(cls):
            if init:
                self.classes[name] = cls()
            else:
                self.classes[name] = cls
            
            if name not in self.args:
                self.args[name] = self.make_dataclass_from_init(
                    cls.__init__, name, arg_keys, stop_args
                )
            return cls

        return add_class_by_name

import numpy as np
import torch as th

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def dict_to_device(dict: dict, device) -> dict:
    return {key: value.to(device) for key, value in dict.items()}

