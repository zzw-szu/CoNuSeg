# CoNuSeg

Implementation for ***"Continual Nuclei Segmentation via Prototype-wise Relation Distillation and Contrastive Learning"***.

## Key Fuctions

- `utils/loss.py - PrototypeWiseRelationDistillationLoss`: Prototype-wise Relation Distillation
- `utils/loss.py - PrototypeWiseContrastiveLoss`: Prototype-wise Contrastive Learning

## Getting Started

```bash
python  run.py \
--logdir $logdir \ # path to save log and checkpoint
--dataset monusac \ # dataset name
--name $name \ # experiment name
--task 1-1 \ # incremental task type
--step 0 \ # incremental step
--step_ckpt $step_ckpt \ # path to load checkpoint, invalid for step==0
--gpu $gpu # CUDA_VISIBLE_DEVICES=$gpu
```

## Citation

```bibtex
@article{wu2023continual,
  title={Continual Nuclei Segmentation via Prototype-wise Relation Distillation and Contrastive Learning},
  author={Wu, Huisi and Wang, Zhaoze and Zhao, Zebin and Chen, Cheng and Qin, Jing},
  journal={IEEE transactions on medical imaging},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgement

This repo is based on [SDR](https://github.com/LTTM/SDR)
