from torchvision import transforms
import random
import torch
import numpy as np
import torch.nn as nn


def get_img_trans(img_size:tuple, norm_type:str=None):
    assert norm_type in ["imagenet", "standard", None], "Invalid norm_type"

    if norm_type == "imagenet":
        norm_trans = transforms.Normalize((0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(img_size),
                                    norm_trans])
    elif norm_type == "standard":
        norm_trans = transforms.Normalize((0.5, 0.5, 0.5),
                                          (0.5, 0.5, 0.5))
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(img_size),
                                    norm_trans])
    else:
        trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(img_size)])

    return trans


def set_seed(random_seed:int):
    random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)


def load_gt_stylef_dist(gt_stylef_dict_path:str, device:str):
    gt_stylef_dict = torch.load(gt_stylef_dict_path)
    gt_stylef_mean = gt_stylef_dict['mean']
    gt_stylef_std = gt_stylef_dict['std']
    gt_stylef_mean = gt_stylef_mean.to(device)
    gt_stylef_std = gt_stylef_std.to(device)

    gt_stylef_dist = torch.distributions.Normal(gt_stylef_mean, gt_stylef_std)
    return gt_stylef_dist


def set_grad_flag(module: nn.Module, flag):
    for p in module.parameters():
        p.requires_grad = flag