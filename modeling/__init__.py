# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .osnet import osnet_x1_0 #,Baseline_osnet
import torch


def build_model(cfg, num_classes):
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model

def build_omni_model(cfg, num_classes):
    model = osnet_x1_0(num_classes=num_classes, loss='softmax')
    print(model)
    model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH), strict=False)
    return model