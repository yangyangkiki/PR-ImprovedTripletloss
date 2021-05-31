# Learning Deep Part-Aware Embedding for Person Retrieval

# Requirements:
Python=3.6 and Pytorch=1.0.0

download pretrained osnet from https://github.com/KaiyangZhou/deep-person-reid 

Thanks to bag of tricks reid (https://github.com/michuanhaohao/reid-strong-baseline) and deep-person-reid (https://github.com/KaiyangZhou/deep-person-reid).

# Training
e.g. for duke

python ./tools/train.py --config_file=configs/softmax_triplet_osnet.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('dukemtmc')" OUTPUT_DIR "('./results_kiki/dukemtmc/softmax_triplet/model1')" MODEL.PRETRAIN_PATH '../pretrained_models/osnet_x1_0_imagenet.pth' DATASETS.ROOT_DIR "('../Datasets')" MODEL.NAME osnet_x1_0 CUHK03.LABELED False

# Citation
If you use our code in your research or wish to refer to the baseline results, please cite our paper:
```
@article{zhao2021learning,
  title={Learning deep part-aware embedding for person retrieval},
  author={Zhao, Yang and Shen, Chunhua and Yu, Xiaohan and Chen, Hao and Gao, Yongsheng and Xiong, Shengwu},
  journal={Pattern Recognition},
  volume={116},
  pages={107938},
  year={2021},
  publisher={Elsevier}
}
```
