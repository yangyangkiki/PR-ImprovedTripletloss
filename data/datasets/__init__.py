# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dataset_loader import ImageDataset
from .prw_2_stage_gt_bbox import PRW_2_Stage_prw_gt_bbox_train
from .prw_2_stage_gt_predict_bbox import  PRW_2_Stage_prw_gt_predict_bbox_train
from .prw_2_stage_predict_bbox import PRW_2_Stage_prw_predict_bbox_train

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'prw2stage-gt-bbox': PRW_2_Stage_prw_gt_bbox_train,
    'prw2stage-gt-predict-bbox': PRW_2_Stage_prw_gt_predict_bbox_train,
    'prw2stage-predict-bbox': PRW_2_Stage_prw_predict_bbox_train
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
