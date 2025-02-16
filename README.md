## About
This repository presents a method for generating floor plans using diffusion. This is an improved method of [House Diffusion](https://github.com/aminshabani/house_diffusion) and [MSD](https://github.com/caspervanengelenburg/msd).

## Training
To reproduce this model, run the following command:

```
$ python training/train.py root=./ exp.config_dir=./ exp.config=configs/train/03_model_hd_data_msd.yaml
```
[Weights & Biases](https://wandb.ai/home) is used to track experiments. You shoud insert your W&B API key in the  `wandb_key` environment variable.
