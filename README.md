# RobustDF-Detv0

Research code for vehicle fire detection, with configuration files and utilities for installation, training, and evaluation.

[Installation](#installation) | [Data and weights](#data-and-weights) | [Training](#training) | [Evaluation](#evaluation)

## Installation

### Recorded experimental environment

| Component | Setting |
|---|---|
| Operating system | Ubuntu 24.04 LTS |
| GPU | NVIDIA GeForce RTX 4090 D, 24 GB |
| CPU / system memory | Intel Core i9-14900KF / approximately 64 GB |
| Python | 3.11.14 |
| PyTorch / TorchVision | 2.1.2+cu118 / 0.16.2+cu118 |
| PyTorch CUDA runtime | 11.8 |
| Installed CUDA toolkit | 12.1 |
| MMCV / MMEngine | 2.1.0 / 0.10.7 |
| MMDetection | 3.3.0 with the included RDF-Det overlay |

The commands below target Ubuntu with an NVIDIA GPU. Check that `nvidia-smi` works before creating the environment.

```bash
git clone https://github.com/luckyLuWen/RobustDF-Detv0.git
cd RobustDF-Detv0

conda create -n rdfdet python=3.11 -y
conda activate rdfdet
python -m pip install pip==23.3.2 setuptools==60.2.0 wheel==0.46.3
python -m pip install numpy==1.26.4
python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install mmcv==2.1.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
python -m pip install -r requirements.txt
python tools/setup_mmdet.py
```

`setup_mmdet.py` downloads MMDetection at commit `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a` (`v3.3.0`), installs the custom modules and the original benchmark compatibility patch, and performs an editable installation. The dependency checkout is excluded from this repository. A small, source-preserving DINOv3 ConvNeXt runtime is included, so no separate DINOv3 installation is needed.

## Data and weights

The image dataset, annotations, and trained model weights (`.pth` files) are available on request from the corresponding author.

Place the required data and pretrained backbone weights as follows:

```text
RobustDF-Detv0/
├── datasets/fire2/coco_4c/
│   ├── annotations/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── images/
├── HAZ-ViT/weight_files/
│   └── dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
└── pretrained/torchvision/
    └── resnet50-0676ba61.pth
```

The dataset uses COCO bounding-box annotations and this class order:

| ID | Annotation label | Meaning |
|---:|---|---|
| 0 | `car_fire` | Ordinary vehicle with visible fire |
| 1 | `car_nofire` | Ordinary vehicle without visible fire |
| 2 | `lkyw_fire` | Coach or oil tanker with visible fire |
| 3 | `lkyw_nofire` | Coach or oil tanker without visible fire |

Here, ordinary vehicles exclude coaches and oil tankers. Smoke without visible fire is labeled `NoFire`.

After placing the data and weights, check the environment and configuration:

```bash
python tools/check_environment.py
```

## Training

Run commands from the repository root with the `rdfdet` environment activated. Configuration paths, dataset paths, and output paths are resolved from the checkout location.

### Main RDF-Det model

```bash
python tools/train.py --seed 0
```

This selects [`rtmdet_l_fire_4c_innov_dualpath_globalgate_dinov3.py`](HAZ-ViT/configs/rtmdet_l_fire_4c_innov_dualpath_globalgate_dinov3.py). Checkpoints and logs are written under `work_dirs/<configuration>/seed_0/`.

### Baselines and routing controls

Pass a configuration filename to select another experiment:

```bash
# RTMDet baseline
python tools/train.py rtmdet_l_fire_4c_std.py --seed 0

# Level-wise learned routing
python tools/train.py rtmdet_l_fire_4c_innov_gated_dualneck_dinov3.py --seed 0

# Fixed uniform fusion
python tools/train.py rtmdet_l_fire_4c_innov_dualpath_nogate_dinov3.py --seed 0

# Parameter-matched frozen routing
python tools/train.py rtmdet_l_fire_4c_innov_dualpath_frozengate_parammatch_dinov3.py --seed 0
```

| Detector family | Input size | Epochs | Training batch |
|---|---|---:|---|
| RTMDet | 640 × 640 | 300 | 8 |
| YOLOX | 640 × 640 | 300 | 8; DINO variants use 4 with two-step accumulation |
| Faster R-CNN | 896 × 896 | 70 | 2 |

The configurations define the optimizer, augmentation, scheduler, and checkpoint settings.

### All five-seed experiments

The reported seeds are `0, 1, 42, 1234, 3407`. The two manifests contain 17 main configurations and 3 additional routing controls, giving 100 training runs.

```bash
# Inspect the 100 jobs without launching training
python tools/train_seeds.py --group all --list-only

# Run the 85 main experiments
python tools/train_seeds.py --group main

# Run the 15 routing-control experiments
python tools/train_seeds.py --group core
```

The runner stores one `train.log` per configuration and seed. Completed runs have a `done.ok` marker and are skipped when the command is repeated. Individual runs can be resumed with the standard MMDetection option:

```bash
python tools/train.py --seed 0 --resume
```

## Evaluation

Evaluate a trained checkpoint on the configured test split:

```bash
python tools/test.py /path/to/trained_checkpoint.pth
```

For a different detector, supply its matching configuration:

```bash
python tools/test.py /path/to/trained_checkpoint.pth \
  --config yolox_l_fire_4c_innov_gated_dualneck_dinov3.py
```

Measure inference throughput with the supplied benchmark utility:

```bash
python tools/benchmark.py /path/to/trained_checkpoint.pth
```

## Summarizing training logs

Extract validation records from training logs and compute numerical summaries:

```bash
python tools/collect_metrics.py work_dirs --output outputs/new_seed_metrics.csv
python tools/summarize.py --metrics outputs/new_seed_metrics.csv \
  --output-dir outputs/new_results
```

## Repository layout

```text
HAZ-ViT/configs/          Original training configurations
HAZ-ViT/configs_seeds5/   Main and routing-control manifests
HAZ-ViT/dinov3_repo/      Minimal DINOv3 ConvNeXt runtime
projects/ViTDet/vitdet/   RDF-Neck, backbone adapter, and focal-loss adapter
patches/mmdetection/     Archived benchmark compatibility changes
tools/                   Setup, training, testing, and numerical summaries
results/                 Compact per-seed metrics and reproduced tables
```

The `projects/ViTDet` import path is retained for compatibility with the archived configurations. The released DINO configurations use the CNN-based ConvNeXt-B backbone. Dataset images, model weights, historical Git objects, full training logs, and generated paper assets are excluded from this code release.

## Acknowledgments and licenses

This implementation builds on [MMDetection](https://github.com/open-mmlab/mmdetection), [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine), and [DINOv3](https://github.com/facebookresearch/dinov3). Third-party source notices and license terms are retained; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
