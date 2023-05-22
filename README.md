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

数据集放入`data/CROHME/`后，依次执行预处理代码：
```python
python data/gen_hybrid_data.py
python data/gen_pkl.py
python data/gen_symbols_struct_dict.py
```


### Pretrain Model
checkpoints/best/best.pth

Results on CROHME 2014
| Model  | ExpRate |  ≤1  |  ≤2  |
|--------|---------|------|------|
| Paper  | 56.2    | 72.6 | 79.2 |
| Paddle | 55.3    | 72.6 | 79.7 |

因为算力点数限制，只调参到与论文中给出的指标相同水平，可以更加精细地调参。

### Test
前向、反向测试均在test文件夹中。

forward
```python
python test/test_forward.py
```

backword

```python
python test/test_backward.py
```

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
