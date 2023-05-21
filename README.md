# Syntax-Aware Network for Handwritten Mathematical Expression Recognition

This is the paddle implementation of [SAN](https://arxiv.org/abs/2203.01601) (CVPR'2022).
![SAN Overview](overview.png)

### Train

single gpu

```
python train.py --config path_to_config_yaml --gpu 0
```

multi gpu

```
python -m paddle.distributed.launch --gpus='0,1,2,3' train.py --multi_gpu
```

### Inference
```
python inference.py --config path_to_config_yaml --image_path path_to_image_folder --label_path path_to_label_folder
```

```
Example:
python inference.py --config 14.yaml --image_path data/14_test_images --label_path data/test_caption.txt
```

### Dataset

CROHME: 
```
Download the dataset from: https://github.com/JianshuZhang/WAP/tree/master/data
```

HME100K
```
Download the dataset from the official website: https://ai.100tal.com/dataset
```

### Pretrain Model
checkpoints/best/best.pth

Results on CROHME 2014
| Model  | ExpRate |  ≤1  |  ≤2  |
|--------|---------|------|------|
| Paper  | 56.2    | 72.6 | 79.2 |
| Paddle | 55.3    | 72.6 | 79.7 |


### Citation

If you find this dataset helpful for your research, please cite the following paper:

```
@inproceedings{yuan2022syntax,
  title={Syntax-Aware Network for Handwritten Mathematical Expression Recognition},
  author={Yuan, Ye and Liu, Xiao and Dikubab, Wondimu and Liu, Hui and Ji, Zhilong and Wu, Zhongqin and Bai, Xiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4553--4562},
  year={2022}
}
```