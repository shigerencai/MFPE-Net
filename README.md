# MFPE-Net

Official PyTorch implementation of **MFPE-Net: Multi-factor Precipitation Evolution Network for Radar Precipitation Nowcasting**.

This repository contains the training and evaluation code for the CIKM AnalytiCup 2017 and Shanghai radar echo datasets.


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
├── train_cikm.py
├── test_cikm.py
├── train_shanghai.py
└── test_shanghai.py
