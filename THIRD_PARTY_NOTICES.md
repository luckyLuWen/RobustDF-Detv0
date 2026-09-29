# Third-party notices

## OpenMMLab

The model adapters and benchmark files retain their OpenMMLab source notices. MMDetection is installed from release v3.3.0, commit `44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`.

The original Apache License 2.0 is provided in [LICENSES/MMDetection-Apache-2.0.txt](LICENSES/MMDetection-Apache-2.0.txt). The installation helper overlays the released project modules and the two benchmark compatibility files onto that checkout.

## DINOv3

The runtime under `HAZ-ViT/dinov3_repo/` is derived from the DINOv3 source snapshot included with the recorded RDF-Det experiments. It is distributed under the original [DINOv3 License Agreement](HAZ-ViT/dinov3_repo/LICENSE.md).

The ConvNeXt implementation is preserved verbatim. The ConvNeXt-Base factory functions retain their original bodies. Package initialization is reduced to the ConvNeXt runtime used by the released configurations; unused transformer, segmentation, depth, and text-model entry points are omitted. Pretrained weights are not redistributed in this repository.

## RDF-Det additions

The repository makes the RDF-Det research implementation available for inspection and reproduction. No separate project-wide license is declared for project-specific additions in this release. The licenses above continue to apply to their respective third-party components.
