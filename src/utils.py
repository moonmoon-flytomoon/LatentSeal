import math
from itertools import repeat
import collections.abc
import importlib
import torch
import torch.nn as nn
from torchvision import transforms
from augly.image import functional as aug_functional
import random
from torchvision.transforms import functional

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

class pixcel_Loss(nn.Module):
    def __init__(self):
        super(pixcel_Loss, self).__init__()

    def forward(self, I_clean, I_wm):
        # 验证输入的维度是否符合预期（Batch Size, Channels, Height, Width）
        if I_clean.dim() != 4 or I_wm.dim() != 4:
            raise ValueError("Input tensors must have 4 dimensions (B, C, H, W)")
        I_clean = (I_clean+ 1.0) / 2.0
        I_wm = (I_wm+ 1.0) / 2.0
        # 计算损失
        loss = torch.abs(I_clean - I_wm) / (I_clean + 1.0)
        return loss.mean()
class WarmupCosineLambda:
    def __init__(self, warmup_steps: int, cycle_steps: int, decay_scale: float, exponential_warmup: bool = False):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.decay_scale = decay_scale
        self.warmup_scale = decay_scale
        self.exponential_warmup = exponential_warmup

    def __call__(self, step: int):
        if step < self.warmup_steps:
            if self.exponential_warmup:
                return self.warmup_scale * pow(self.warmup_scale, -step / self.warmup_steps)
            ratio = step / self.warmup_steps
            return self.warmup_scale + (1 - self.warmup_scale) * ratio
        else:
            ratio = (1 + math.cos(math.pi * (step - self.warmup_steps) / self.cycle_steps)) / 2
            return self.decay_scale + (1 - self.decay_scale) * ratio

### Load LDM models

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


image_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

def normalize_img(x):
    """ Normalize image to approx. [-1,1] """
    return (x - image_mean.to(x.device)) / image_std.to(x.device)

def unnormalize_img(x):
    """ Unnormalize image to [0,1] """
    return (x * image_std.to(x.device)) + image_mean.to(x.device)

def clamp_pixel(x):
    """
    Clamp pixel values to 0 255.
    Args:
        x: Image tensor with values approx. between [-1,1]
    Returns:
        y: Rounded image tensor with values approx. between [-1,1]
    """
    # x_pixel = 255 * unnormalize_img(x)
    x_pixel = 255 * x
    y = x_pixel.clamp(0, 255)
    y = normalize_img(y/255.0)
    return y

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: Tensor image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    x = unnormalize_img(x)
    for ii,img in enumerate(x):
        pil_img = to_pil(img)
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    # return normalize_img(img_aug)
    return img_aug

class DiffJPEG(nn.Module):
    def __init__(self,p=0.5, quality=50):
        super().__init__()
        self.quality = quality
        self.p = p

    def forward(self, x):
        random_number = random.random()
        if random_number <= self.p:
            with torch.no_grad():
                img_clip = clamp_pixel(x)
                img_jpeg = jpeg_compress(img_clip, self.quality)
                img_gap = img_jpeg - x
                img_gap = img_gap.detach()
            img_aug = x + img_gap
            return img_aug
        else:
            return x

class RandomSaturation(nn.Module):
    def __init__(self,p=0.5, saturation=(0.5,1.5)):
        super().__init__()
        self.saturation = saturation
        self.p = p

    def forward(self, x):
        for i in range(x.shape[0]):
            random_number = random.random()
            if random_number < self.p:
                random_saturation = random.uniform(self.saturation[0],self.saturation[1])
                i_aug = functional.adjust_contrast(x[i], random_saturation)
                x[i] = i_aug
        return x