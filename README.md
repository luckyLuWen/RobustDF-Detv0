# RDF-Det

**Robust Dual-Path Fusion for CNN-Based Vehicle Fire Detection in Ground-Based RGB Imagery**

Wenyi Lu, Xiang Zhang, Boyu Zhang, Zhiqing Li, Zeqiang Chen, and Nengcheng Chen<br>
China University of Geosciences, Wuhan

[Installation](#installation) | [Data and weights](#data-and-weights) | [Training](#training) | [Evaluation](#evaluation) | [Results](#results) | [Result reproduction](#result-reproduction)

RDF-Det detects vehicles and their visible fire states in road-scene RGB images. Its RDF-Neck combines a top-down Semantic Cascade Path, a bidirectional Spatial Reinforcement Path, a fusion operator, and a post-fusion alignment block. The main model uses a DINOv3 **ConvNeXt-B** backbone and the RTMDet detection head.

```text
RGB image -> CNN backbone -> parallel FPN / CSPNeXtPAFPN paths
          -> learned or fixed fusion -> post-fusion alignment -> detection head
```

This release includes the 20 experimental configurations, the custom model modules, training/evaluation tools, and compact metrics from the 100 reported runs. The main configuration, **RTM-RDF-GLR-DINO**, obtains **0.6352 AP** and **0.9290 AP50**, an improvement of **7.24 AP points** over RTM-Base. Its recorded batch-one throughput is **42.8 FPS** on an RTX 4090 D.

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

### Downloads

Google Drive links are reserved below and will be filled after the files are uploaded.

| Asset | Used for | Google Drive |
|---|---|---|
| TFR images and COCO annotations | Training and evaluation | |
| DINOv3 ConvNeXt-B pretrained weights | DINO-based configurations | |
| ResNet-50 pretrained weights | Faster R-CNN baseline and neck control | |
| Trained RDF-Det checkpoints | Evaluation and inference benchmarking | |

Place the downloaded files as follows:

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

Here, ordinary vehicles exclude coaches and oil tankers. Smoke without visible fire is labeled `NoFire`. TFR was specifically curated to include scarce fire imagery involving coaches and oil tankers.

| Split | Images | Annotated objects |
|---|---:|---:|
| Train | 968 | 988 |
| Validation | 277 | 283 |
| Test | 141 | 142 |

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

The pretrained DINOv3 ConvNeXt-B backbone is jointly fine-tuned with the neck and head. The SCP mixing coefficient starts at 0.85 for LLR and 0.5 for GLR; FUF and PMFR retain 0.5. The released configurations preserve the original optimizer, augmentation, scheduler, and checkpoint settings.

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

The archived throughput command used PyTorch eager execution, batch size 1, `--max-iter 100`, and `--num-warmup 20` on the RTX 4090 D. MMDetection counts warm-up iterations within `max-iter`, leaving 80 CUDA-synchronized timed iterations. The main model has approximately 118M parameters and 182G FLOPs at 640 × 640.

```bash
python tools/benchmark.py /path/to/trained_checkpoint.pth
```

## Results

The table reports mean **best validation AP** and **best validation AP50** over five seeds, selected independently within each run. AP is averaged over IoU thresholds 0.50:0.05:0.95. Scores remain on the 0–1 scale; differences are expressed in AP points relative to the baseline of the same detector family.

<!-- RESULTS_TABLE_START -->
| Model | AP | AP50 | Delta AP (points) |
|---|---:|---:|---:|
| **[RTM-RDF-GLR-DINO](HAZ-ViT/configs/rtmdet_l_fire_4c_innov_dualpath_globalgate_dinov3.py)** | 0.6352 | 0.9290 | +7.24 |
| [RTM-RDF-FUF-DINO](HAZ-ViT/configs/rtmdet_l_fire_4c_innov_dualpath_nogate_dinov3.py) | 0.6348 | 0.9288 | +7.20 |
| [RTM-RDF-PMFR-DINO](HAZ-ViT/configs/rtmdet_l_fire_4c_innov_dualpath_frozengate_parammatch_dinov3.py) | 0.6314 | 0.9312 | +6.86 |
| [RTM-DINO-Opt](HAZ-ViT/configs/rtmdet_l_fire_4c_var_backbone_dinov3_opt_backbone_lr01.py) | 0.6312 | 0.9214 | +6.84 |
| [RTM-RDF-LLR-DINO](HAZ-ViT/configs/rtmdet_l_fire_4c_innov_gated_dualneck_dinov3.py) | 0.6294 | 0.9214 | +6.66 |
| [RTM-RDF-LLR-CSP](HAZ-ViT/configs/rtmdet_l_fire_4c_innov_gated_dualneck_cspnext.py) | 0.5840 | 0.8912 | +2.12 |
| [RTM-FPN](HAZ-ViT/configs/rtmdet_l_fire_4c_var_neck_fpn.py) | 0.5784 | 0.8898 | +1.56 |
| [RTM-DINO](HAZ-ViT/configs/rtmdet_l_fire_4c_var_backbone_dinov3.py) | 0.5742 | 0.8630 | +1.14 |
| [RTM-DINO-Opt-WD0](HAZ-ViT/configs/rtmdet_l_fire_4c_var_backbone_dinov3_opt_backbone_wd0.py) | 0.5730 | 0.8628 | +1.02 |
| [RTM-Base](HAZ-ViT/configs/rtmdet_l_fire_4c_std.py) | 0.5628 | 0.8608 | +0.00 |
| [RTM-Focal](HAZ-ViT/configs/rtmdet_l_fire_4c_var_loss_focal.py) | 0.5506 | 0.8470 | -1.22 |
| [YOLOX-RDF-LLR-DINO](HAZ-ViT/configs/yolox_l_fire_4c_innov_gated_dualneck_dinov3.py) | 0.6240 | 0.9314 | +15.64 |
| [YOLOX-DINO](HAZ-ViT/configs/yolox_l_fire_4c_var_backbone_dinov3.py) | 0.6218 | 0.9308 | +15.42 |
| [YOLOX-CSPPAFPN](HAZ-ViT/configs/yolox_l_fire_4c_var_neck_cspnextpafpn.py) | 0.5098 | 0.8196 | +4.22 |
| [YOLOX-Base](HAZ-ViT/configs/yolox_l_fire_4c_std.py) | 0.4676 | 0.8046 | +0.00 |
| [YOLOX-GIoU](HAZ-ViT/configs/yolox_l_fire_4c_var_loss_giou.py) | 0.4610 | 0.8086 | -0.66 |
| [FRCNN-DINO](HAZ-ViT/configs/faster_rcnn_fire_4c_plus_dino.py) | 0.5864 | 0.9072 | +20.00 |
| [FRCNN-DINO-Neck](HAZ-ViT/configs/faster_rcnn_fire_4c_plus_both.py) | 0.5772 | 0.9098 | +19.08 |
| [FRCNN-Base](HAZ-ViT/configs/faster_rcnn_fire_4c_baseline.py) | 0.3864 | 0.7368 | +0.00 |
| [FRCNN-Neck](HAZ-ViT/configs/faster_rcnn_fire_4c_plus_neck.py) | 0.2792 | 0.6092 | -10.72 |
<!-- RESULTS_TABLE_END -->

Full-precision means, standard deviations, confidence intervals, final scores, and best–final gaps are provided in [`results/main_results.csv`](results/main_results.csv). Paired baseline comparisons and family-wise Holm corrections are in [`results/baseline_comparisons.csv`](results/baseline_comparisons.csv).

## Result reproduction

The compact [`seed_metrics.csv`](results/seed_metrics.csv) contains the 100 archived run records used for the table. Recomputing these numerical summaries does not require a GPU, images, or model weights:

```bash
python -m pip install numpy==1.26.4 scipy==1.17.0
python tools/summarize.py
```

Outputs are written to `outputs/results/`. After running new training experiments, extract their validation records and summarize them with the same selection rule:

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

## Citation

```bibtex
@misc{lu2026rdfdet,
  title  = {RDF-Det: Robust Dual-Path Fusion for CNN-Based Vehicle Fire Detection in Ground-Based RGB Imagery},
  author = {Lu, Wenyi and Zhang, Xiang and Zhang, Boyu and Li, Zhiqing and Chen, Zeqiang and Chen, Nengcheng},
  year   = {2026},
  note   = {Manuscript and research code},
  url    = {https://github.com/luckyLuWen/RobustDF-Detv0}
}
```

## Acknowledgments and licenses

This implementation builds on [MMDetection](https://github.com/open-mmlab/mmdetection), [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine), and [DINOv3](https://github.com/facebookresearch/dinov3). Third-party source notices and license terms are retained; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
