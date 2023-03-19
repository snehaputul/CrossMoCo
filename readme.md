# CrossMoCo

Official implementation of the paper [CrossMoCo: Multi-modal Momentum Contrastive Learning for Point Cloud](https:).


# Dataset
For data preparation, please refer to [CrossPoint](https://github.com/MohamedAfham/CrossPoint

# Training

### ScanObjectNN
```bash
python train_crosspoint_moco_pretrained_scanobjectNN.py --model dgcnn --epochs 125 --lr 0.001 --gpu 3 --output_dim 256 --batch_size 20 --print_freq 200 --k 15 --K 4000 --exp_name moco_crossspoint_scanobject --img_model resnet50 --m 0.9999
```


### ModelNet40
```bash 
python train_crosspoint_moco_pretrained.py --model dgcnn --epochs 125 --lr 0.001 --gpu 2 --output_dim 256 --batch_size 20 --print_freq 200 --k 15 --K 4000 --exp_name moco_crossspoint_resnet50_epoch125_m_0.9999 --img_model resnet50 --m 0.9999
```

# Acknowledgement
Our code is based on [CrossPoint](https://github.com/MohamedAfham/CrossPoint)