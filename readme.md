# CrossMoCo

Official implementation of the paper [CrossMoCo: Multi-modal Momentum Contrastive Learning for Point Cloud](https://ieeexplore.ieee.org/abstract/document/10229841).

**Few-shot 3D Point Cloud Classification**
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/few-shot-3d-point-cloud-classification-on-6)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-6?p=crossmoco-multi-modal-momentum-contrastive)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/few-shot-3d-point-cloud-classification-on-7)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-7?p=crossmoco-multi-modal-momentum-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/few-shot-3d-point-cloud-classification-on-8)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-8?p=crossmoco-multi-modal-momentum-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/few-shot-3d-point-cloud-classification-on-9)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-9?p=crossmoco-multi-modal-momentum-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/few-shot-3d-point-cloud-classification-on-2)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-2?p=crossmoco-multi-modal-momentum-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/few-shot-3d-point-cloud-classification-on-1)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-1?p=crossmoco-multi-modal-momentum-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/few-shot-3d-point-cloud-classification-on-3)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-3?p=crossmoco-multi-modal-momentum-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/few-shot-3d-point-cloud-classification-on-4)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-4?p=crossmoco-multi-modal-momentum-contrastive)

**3D Point Cloud Classification**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/3d-point-cloud-linear-classification-on-1)](https://paperswithcode.com/sota/3d-point-cloud-linear-classification-on-1?p=crossmoco-multi-modal-momentum-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=crossmoco-multi-modal-momentum-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/3d-object-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-object-classification-on-modelnet40?p=crossmoco-multi-modal-momentum-contrastive)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/crossmoco-multi-modal-momentum-contrastive/3d-point-cloud-linear-classification-on)](https://paperswithcode.com/sota/3d-point-cloud-linear-classification-on?p=crossmoco-multi-modal-momentum-contrastive)


# Dataset
For data preparation, please refer to [CrossPoint](https://github.com/MohamedAfham/CrossPoint)

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
