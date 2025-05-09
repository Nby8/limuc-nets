# **limuc-nets** 
This is an official Pytorch implementation of paper "Ulcerative severity Estimation based on advanced CNN-transformer hybrid models"

# Image Classification
## 1. Requirements
For model CoAtNet, Maxvit, Deit please use the dependencies in [requirements.txt](/requirements.txt) document.

For model OverLoCK，we hightly suggested creating a new conda environment following the steps below on the basis of the existing env above:
* create a new environment
  ```
  conda create -n overlock_env python=3.9
  ```
* install the dependencies:
  ```
  pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
  pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
  pip install timm==0.6.12
  pip install mmengine==0.2.0
  ```

>[!note]
>The timm version used for other models is 1.0.15

## 2. Get LIMUC dataset
You can get the LIMUC dataset [here](https://zenodo.org/record/5827695#.Yi8GJ3pByUk) 

## 3. Train、 validation and test
To train、 validation and test the models on LIMUC, run the following coomand according to different models:

For CoAtNet(0~3), run:
```
python main.py --loss='CDW' --CV --finetune pretrained --data-set='LIMUC' --model coatnet_2_rw_224 --model-name coatnet_2 -alpha=5.0
```
For DeiT, run:
```
python main.py --loss='CDW' --CV --finetune pretrained --data-set='LIMUC' --model deit_base_distilled_patch16_224 --model-name deit -alpha=x
```
For MaxVit, run:
```
python main.py --loss='CDW' --CV --finetune pretrained --data-set='LIMUC' --model maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k --model-name maxvit -alpha=x
```
For OverLoCK, run:
```
python main.py --loss='CDW' --CV --finetune pretrained --data-set='LIMUC' --model overlock_b --model-name overlock_b -alpha=1.0
```
## Citation

## Acknowledgment
Our implementation is mainly based on the following codebases and we sincerely appreciate the outstanding contributions of their authors.
> [timm](https://github.com/rwightman/pytorch-image-models), [natten](https://github.com/SHI-Labs/NATTEN), [unireplknet](https://github.com/AILab-CVC/UniRepLKNet), [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection), [mmseg](https://github.com/open-mmlab/mmsegmentation)
> [coatnet](https://github.com/chinhsuanwu/coatnet-pytorch), [DeiT](https://github.com/facebookresearch/deit), [overlock](https://github.com/LMMMEng/OverLoCK)

# Contact
Any Question please feel free to [contact me](brinnie253@gmail.com)
