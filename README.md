# Knowledge Distillation via conditional generative modeling to improve image quality of specific region.
使用 conditional StyleGAN2 進行知識蒸餾，用來改善特定區域的圖片品質。

## Train
本專案以 Pytorch Distributed Data Parallel 支援多 GPU 訓練，在 config 檔案(.yml)設定訓練相關的參數，以 comand line arguments 決定有關 performance(e.g. mixed precision training) 及 monitoring(以 weight & bias 追蹤訊練數據)。
* basic training command
```bash
python train.py --cfg <config-file.yml>
```

* Multi-gpu training
```bash
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=<num-gpus> train.py --cfg <config-file.yml>
```
When using subset gpus to training (e.g. only use 4 gpus on 8 gpus server). Please set `CUDA_VISIBLE_DEVICES` to control which gpus should be used.

* resume training or finetuning
    * training iteration in config file means **target iteration** instead of additional iteration
    * When resuming training with wandb enabled, please refer weight & bias chpater

### Training Configuration
I design configuration with `yacs`(same style as Detectron). Please refer `config.py` to default configuration.

### Command line Arguments
* `--cfg=<config-file.yml>`: path to configuration file.
* `--amp`: enable mixed-precision training (via pytorch autocast). It might cause overflow during training. Use it carefully.
* `--no-wandb`: disable tracking training stats with *weight & bias*
* `--debug`: debug mode
* `--nobench`: diable CUDNN benchmarking. CUDNN benchmarking will best algorithm in start of training
* `--seed=<seed>`: random seed for reproducibility. default 0

### About weight & bias
This repo. monitoring training stats with `weight & bias`(wandb). User should use `wandb login` at first time.
It will kepp synchronizing to wandb server during training. However, it is possible to store training stats in local and sync to server later. For that case, please refer `wandb offline`. When resuming training with wandb enabled, please set `WANDB_RESUME=must` and `WANDB_RUN_ID=<your-previous-wandb-ID>` to record training stats to same experiment run.

## Evaluation
Currently only support **FID** and **KID** separately.
* FID
```bash
python -m metrics.fid --cfg <config-file.yml> --ckpt <single checkpoint or folder> --out_dir <output folder> --gpus <num_gpus> --save
```
When calc. metrics, configuration file only be used to decide **classess**, **DATASET**, **MODEL**, **EVAL.FID**

## Details
1. 設定 `OMP_NUM_THREADS=1` 為避免 `scipy` 在計算 **FID** 時的 bug