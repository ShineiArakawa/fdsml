"""DeiT models from timm with pretrained weights from the original DeiT repository.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

This code is a derivative work based on code from the nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch repository, whose original code is licensed under the MIT License. The original code can be found at https://github.com/nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch.
"""

# autopep8: off
# isort: skip_file
import warnings
from functools import partial

# Mute warnings from timm library about registry overwriting
warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm\.models\.registry")
warnings.filterwarnings("ignore", category=UserWarning, message=r"Overwriting .* in registry .*")

import torch
import torch.nn as nn
from timm.models._registry import register_model
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg, VisionTransformer
# autopep8: on


# --------------------------------------------------------------------------------------------------------
# DeiT models from timm with pretrained weights from the original DeiT repository


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    drop_path_rate = kwargs.pop('drop_path_rate', 0.1)
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    drop_path_rate = kwargs.pop('drop_path_rate', 0.1)
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    drop_path_rate = kwargs.pop('drop_path_rate', 0.1)
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=drop_path_rate, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

# --------------------------------------------------------------------------------------------------------
