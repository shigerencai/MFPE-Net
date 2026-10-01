# MFPE-Net

Official PyTorch implementation of **MFPE-Net: Multi-factor Precipitation Evolution Network for Radar Precipitation Nowcasting**.

This repository provides the model implementation, dataset loaders, training scripts, evaluation scripts, metric computation, checkpoint utilities, and visualization utilities for experiments on the CIKM AnalytiCup 2017 and Shanghai radar echo datasets.

## Repository Structure

```text
MFPE-Net/
├── datasets/
│   ├── cikm_folder.py
│   └── shanghai_folder.py
├── models/
│   └── mfpe_net.py
├── utils/
│   ├── checkpoint.py
│   ├── metrics.py
│   └── visualize.py
├── configs/
│   ├── cikm.json
│   └── shanghai.json
├── checkpoints/
│   └── README.md
├── train_cikm.py
├── test_cikm.py
├── train_shanghai.py
├── test_shanghai.py
├── requirements.txt
└── README.md
```

## Environment

The experiments reported in the manuscript were conducted with:

```text
Python 3.13
PyTorch 2.9.1
CUDA 13.0
NVIDIA GeForce RTX 5060
```

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

`thop` is included for optional FLOPs profiling. If FLOPs profiling is not required, the main training and evaluation pipeline does not depend on it.

## Dataset Preparation

### CIKM AnalytiCup 2017

The CIKM dataset should be organized as follows:

```text
/path/to/CIKM/
├── train/
│   ├── sample_1/
│   │   ├── img_1.png
│   │   ├── img_2.png
│   │   ├── ...
│   │   └── img_15.png
│   └── ...
├── validation/
│   ├── sample_1/
│   └── ...
└── test/
    ├── sample_1/
    └── ...
```

Each sequence contains 15 consecutive radar frames. The first 5 frames are used as historical input and the following 10 frames are used as prediction targets.

The loader reads grayscale images, normalizes pixel values to `[0,1]`, and applies center cropping/padding to the requested spatial resolution. The default image size is `128 x 128`.

For threshold-based evaluation, the implementation uses:

```text
PIXEL_SCALE = 80
THRESHOLDS = [20, 30, 35, 40]
```

### Shanghai Radar Dataset

The Shanghai dataset should be organized as follows:

```text
/path/to/Shanghai/
├── train/
│   ├── sample_1/
│   └── ...
├── validation/
│   ├── sample_1/
│   └── ...
└── test/
    ├── sample_1/
    └── ...
```

Each sequence contains 20 radar frames. The first 10 frames are used as historical input and the following 10 frames are used as prediction targets.

The loader reads grayscale images, normalizes pixel values to `[0,1]`, and resizes them to `128 x 128` by default.

For threshold-based evaluation, the implementation uses:

```text
PIXEL_SCALE = 90
THRESHOLDS = [20, 30, 35, 40]
```

# Reproduction

The commands below follow the released training and evaluation scripts. The exact command-line defaults used by the repository are also recorded in `configs/cikm.json` and `configs/shanghai.json`.

## 1. Train MFPE-Net on CIKM

```bash
python train_cikm.py \
    --data_root /path/to/CIKM \
    --output_dir ./outputs \
    --exp_name mfpe_cikm \
    --img_size 128 \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --seed 2026
```

The training script uses `AdamW` and a cosine annealing learning-rate scheduler. The released script currently uses a default `weight_decay` of `0.0`.

For each run, the script automatically stores the complete command-line configuration in:

```text
outputs/mfpe_cikm/args.json
```

Training outputs are organized as:

```text
outputs/mfpe_cikm/
├── args.json
├── metrics_log.jsonl
├── metrics_val_log.jsonl
└── checkpoints/
    ├── best.pt
    └── last.pt
```

`best.pt` is selected according to the highest validation CSI-M and should be used for test-set evaluation.

## 2. Evaluate MFPE-Net on CIKM

```bash
python test_cikm.py \
    --data_root /path/to/CIKM \
    --ckpt ./outputs/mfpe_cikm/checkpoints/best.pt \
    --split test \
    --output_dir ./results/cikm
```

The evaluation script reports CSI, POD, HSS, SSIM, and MSE. The main numerical output is saved to:

```text
results/cikm/metrics.json
```

Per-sample CSI40 statistics are additionally stored as:

```text
results/cikm/sample_csi40_scores.json
results/cikm/sample_csi40_scores.csv
results/cikm/csi40_distribution.json
```

To save individual input, ground-truth, and prediction frames for qualitative inspection, run:

```bash
python test_cikm.py \
    --data_root /path/to/CIKM \
    --ckpt ./outputs/mfpe_cikm/checkpoints/best.pt \
    --split test \
    --output_dir ./results/cikm_vis \
    --save_individual \
    --save_input
```

## 3. Train MFPE-Net on Shanghai

```bash
python train_shanghai.py \
    --data_root /path/to/Shanghai \
    --output_dir ./outputs \
    --exp_name mfpe_shanghai \
    --img_size 128 \
    --seq_len 20 \
    --frames_in 10 \
    --frames_out 10 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --seed 2026
```

The complete command-line configuration is saved to:

```text
outputs/mfpe_shanghai/args.json
```

The best checkpoint is selected according to validation CSI-M and saved to:

```text
outputs/mfpe_shanghai/checkpoints/best.pt
```

## 4. Evaluate MFPE-Net on Shanghai

```bash
python test_shanghai.py \
    --data_root /path/to/Shanghai \
    --ckpt ./outputs/mfpe_shanghai/checkpoints/best.pt \
    --split test \
    --output_dir ./results/shanghai
```

The main numerical results are saved to:

```text
results/shanghai/metrics.json
```

To save individual input, ground-truth, and prediction frames, run:

```bash
python test_shanghai.py \
    --data_root /path/to/Shanghai \
    --ckpt ./outputs/mfpe_shanghai/checkpoints/best.pt \
    --split test \
    --output_dir ./results/shanghai_vis \
    --save_individual \
    --save_input
```

## 5. Main Model Configuration

The principal defaults shared by the released training scripts are:

| Setting | Value |
| --- | ---: |
| Image size | 128 x 128 |
| Batch size | 4 |
| Epochs | 100 |
| Learning rate | 1e-4 |
| Random seed | 2026 |
| Hidden dimension | 128 |
| Number of layers | 4 |
| `spec_num` | 20 |
| Physical branch base channels | 48 |
| Maximum motion | 8.0 |
| Residual scale | 0.25 |
| Refiner hidden dimension | 96 |
| Refiner blocks | 4 |
| Strong-echo threshold | 35 |
| Gate sharpness | 1.0 |
| Maximum refinement residual | 0.15 |
| Refiner loss weight | 0.05 |
| Lead-time loss weight | 0.10 |
| Lead-time weight start/end | 1.0 / 2.0 |
| Temporal-consistency loss weight | 0.05 |
| Phase loss weight | 0.01 |
| Amplitude loss weight | 0.01 |
| Intensity-branch loss weight | 0.10 |
| Physical-branch loss weight | 0.10 |
| Gradient clipping | 1.0 |

Dataset-specific sequence settings are:

| Dataset | Sequence length | Input | Output | Evaluation scale |
| --- | ---: | ---: | ---: | ---: |
| CIKM | 15 | 5 | 10 | 80 |
| Shanghai | 20 | 10 | 10 | 90 |

The complete released defaults are listed in the JSON files under `configs/`.

## 6. Metric Computation

Threshold-based metrics are evaluated at:

```text
20, 30, 35, and 40
```

The evaluator accumulates hits, misses, false alarms, and correct negatives over the complete test set before computing CSI, POD, and HSS.

CSI-M, POD-M, and HSS-M are obtained by averaging the corresponding scores over the four thresholds.

MSE is accumulated over all predicted pixels after conversion to the dataset-specific evaluation scale.

SSIM is computed using an `11 x 11` Gaussian window with standard deviation `1.5` and is averaged over all evaluated prediction frames and sequences.

The implementation is provided in:

```text
utils/metrics.py
```

## 7. Checkpoints

Each training run automatically produces:

```text
checkpoints/best.pt
checkpoints/last.pt
```

The checkpoint used for evaluation should be `best.pt`, which corresponds to the highest validation CSI-M during training.

To distribute the exact paper checkpoints, place the CIKM and Shanghai `best.pt` files in a GitHub Release and record the release links in `checkpoints/README.md`.

## 8. Mapping Repository Files to Paper Experiments

| Paper component | Repository entry |
| --- | --- |
| MFPE-Net architecture | `models/mfpe_net.py` |
| CIKM training | `train_cikm.py` |
| CIKM evaluation | `test_cikm.py` |
| Shanghai training | `train_shanghai.py` |
| Shanghai evaluation | `test_shanghai.py` |
| CSI/POD/HSS/MSE/SSIM | `utils/metrics.py` |
| Checkpoint loading/saving | `utils/checkpoint.py` |
| Prediction visualization | `utils/visualize.py`, `test_cikm.py`, `test_shanghai.py` |

The current repository directly reproduces MFPE-Net training, validation, test metrics, and model prediction visualizations. Comparative baseline results and paper-level multi-model plotting require the corresponding baseline outputs under the same evaluation protocol.

## Citation

If you find this repository useful, please cite the corresponding paper.

```bibtex
@article{mfpenet,
  title   = {MFPE-Net: Multi-factor Precipitation Evolution Network for Radar Precipitation Nowcasting},
  author  = {Shiyuan Gao and co-authors},
  journal = {Information Sciences},
  year    = {2026}
}
```
