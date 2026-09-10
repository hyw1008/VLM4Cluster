# Dataset Notes

This document records the benchmark's datasets, recommended acquisition methods, and source
differences to consider when preparing them.

Originally recorded: `2026-04-20`

## Acquisition categories

- `native auto-download`: Download directly through common Python dataset APIs, preferably `torchvision`.
- `external scripted`: Download through external scripts, TFDS, or resources from official repositories, generally outside the `torchvision download=True` workflow.
- `manual prepared`: Download and organize files manually, or extract a subset from a larger source dataset before placing it in `data/`.

## Overview

| Benchmark Group | Dataset | Recommended Mode | Notes |
| --- | --- | --- | --- |
| Classic | CIFAR-10 | `native auto-download` | `torchvision.datasets.CIFAR10` supports `download=True` |
| Classic | CIFAR-20 | `native auto-download` | Download `CIFAR-100` and map its coarse labels to 20 classes |
| Classic | STL-10 | `native auto-download` | `torchvision.datasets.STL10` supports `download=True` |
| Classic | ImageNet-10 | `manual prepared` | ImageNet subset common in deep clustering research; generated from prepared ImageNet data rather than downloaded as a separate official dataset |
| Classic | ImageNet-Dogs | `manual prepared` | ImageNet subset common in deep clustering research; generated from prepared ImageNet data rather than downloaded as a separate official dataset |
| Challenging | DTD | `native auto-download` | `torchvision.datasets.DTD` supports `download=True` |
| Challenging | UCF101 | `manual prepared` | Originally a video dataset; download videos and split annotations manually, then prepare frame images |
| Challenging | CIFAR-100 | `native auto-download` | `torchvision.datasets.CIFAR100` supports `download=True` |
| Large-scale | Places365-Standard | `native auto-download` | `torchvision.datasets.Places365` supports downloading official components |
| Large-scale | ImageNet-1K | `manual prepared` | Download ILSVRC2012 files manually and prepare the required directories |
| Fine-grained | Aircraft | `native auto-download` | `torchvision.datasets.FGVCAircraft` supports `download=True` |
| Fine-grained | Flowers | `native auto-download` | `torchvision.datasets.Flowers102` supports `download=True` |
| Fine-grained | Food | `native auto-download` | `torchvision.datasets.Food101` supports `download=True` |
| Fine-grained | Cars | `manual prepared` | The benchmark prefers the official `Stanford_Cars.tar` layout, `train/test/<class_name>`, and also supports torchvision's older metadata layout |
| Fine-grained | Pets | `native auto-download` | `torchvision.datasets.OxfordIIITPet` supports `download=True` |
| Domain-shift | ImageNet-A | `external scripted` | Acquire through TFDS or scripts using resources from the official repository |
| Domain-shift | ImageNet-Sketch | `external scripted` | Acquire through TFDS or scripts using resources from the official repository |
| Domain-shift | ImageNet-R | `external scripted` | Optional automatic downloading and directory preparation are built into the benchmark |
| Domain-shift | ImageNet-V2 | `external scripted` | Optional automatic downloading and directory preparation are built in; the default variant is `matched-frequency` |
| Domain-shift | ImageNet-C (gaussian_noise, level 1) | `manual prepared` | Download the full ImageNet-C dataset manually, then use the `gaussian_noise/1` portion |

## Shared directory convention

The code uses the `data/<dataset_registry_name>/...` directory convention.
Here, `dataset_registry_name` is the registry name listed by `python3 main.py list-datasets`.

### Directories for automatically downloaded datasets

Automatically downloaded datasets are written to these paths by default:

- `data/cifar10/`
- `data/cifar20/`
- `data/cifar100/`
- `data/stl10/`
- `data/dtd/`
- `data/places365_standard/`
- `data/aircraft/`
- `data/flowers/`
- `data/food/`
- `data/pets/`

### Directories for manually prepared datasets

Datasets prepared manually or through external scripts use these paths:

- `data/imagenet/<split>/<class_name>/*`
- `data/imagenet10/<split>/<class_name>/*`
- `data/imagenet_dogs/<split>/<class_name>/*`
- `data/ucf101/<split>/<class_name>/*`
- `data/cars/train/<class_name>/*` and `data/cars/test/<class_name>/*`
- `data/imagenet_a/test/<class_name>/*`
- `data/imagenet_sketch/test/<class_name>/*`
- `data/imagenet_r/test/<class_name>/*`
- `data/imagenet_v2/test/<class_name>/*`
- `data/imagenet_c/gaussian_noise/1/test/<class_name>/*`

The shared text resource uses:

- `data/WordNetNouns.csv`

Neither `imagenet10` nor `imagenet_dogs` requires a separate download. Once the full
`data/imagenet/<split>/<wnid>` layout is ready, running either dataset automatically generates
its subset directories using `data/imagenet_subsets/imagenet10_wnids.txt` or
`data/imagenet_subsets/imagenet_dogs_wnids.txt`. Symlinks are used by default.
If your cluster does not support symlinks, set the dataset parameter `imagenet_subset_mode = "copy"`.

The loader supports automatic preparation of `imagenet_v2`:

- If `data/imagenet_v2/test` is already prepared, the loader reads it directly.
- If the directory is missing and `image_dataset.download` is not explicitly set to `false`,
  the code downloads the `matched-frequency` archive by default and maps the numeric class
  directories `0..999` to `wnid` directories.
- This mapping requires a locally prepared `data/imagenet/val` directory, falling back to
  `data/imagenet/train` if needed.
- Set `image_dataset.params.imagenet_v2_variant` to select from
  `matched-frequency`, `threshold0.7`, or `top-images`.
- Set `image_dataset.params.imagenet_v2_materialize_mode = "symlink" | "copy"` to control materialization.
- If you have downloaded the archive manually, reuse it by setting
  `image_dataset.params.imagenet_v2_archive_path = "/path/to/archive.tar.gz"`.

The loader also supports automatic preparation of `imagenet_r`:

- If `data/imagenet_r/test` is already prepared, the loader reads it directly.
- If the directory is missing and `image_dataset.download` is not explicitly set to `false`,
  the code downloads the official `imagenet-r.tar` archive by default and organizes it as
  `data/imagenet_r/test/<wnid>/*`.
- Set `image_dataset.params.imagenet_r_materialize_mode = "symlink" | "copy"` to control materialization.
- If you have downloaded the archive manually, reuse it by setting
  `image_dataset.params.imagenet_r_archive_path = "/path/to/imagenet-r.tar"`.

## Dataset groups

### Classic datasets

#### CIFAR-10

- Recommended acquisition: `native auto-download`
- Recommended source: `torchvision.datasets.CIFAR10`
- Notes: A standard small-scale image classification dataset that can be downloaded directly through code.
- Source:
  [Torchvision CIFAR10](https://docs.pytorch.org/vision/0.18/generated/torchvision.datasets.CIFAR10.html)

#### CIFAR-20

- Recommended acquisition: `native auto-download`
- Recommended source: The original `CIFAR-100` archive with coarse-label mapping.
- Notes: `CIFAR-20` is generally not a separate download; its 20 superclasses are derived from `CIFAR-100`.
- Sources:
  [Torchvision CIFAR100](https://docs.pytorch.org/vision/0.23/generated/torchvision.datasets.CIFAR100.html),
  [TFDS CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100)

#### STL-10

- Recommended acquisition: `native auto-download`
- Recommended source: `torchvision.datasets.STL10`
- Notes: Upstream torchvision supports `train`, `test`, `unlabeled`, and `train+unlabeled`. This benchmark exposes `train`, `test`, and `train+unlabeled`; the standalone `unlabeled` split is not supported by its loader.
- Source:
  [Torchvision STL10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.STL10.html)

#### ImageNet-10

- Recommended acquisition: `manual prepared`
- Recommended source: ImageNet subsets commonly used in deep clustering papers and their accompanying implementations.
- Notes: This is a commonly used subset of `ImageNet-1K`, rather than a general-purpose dataset released independently. The benchmark generates missing subset directories from prepared ImageNet data using its bundled wnid list, as described above.
- The description of common practice is an inference from public clustering repositories and papers.
- Sources:
  [ProPos README](https://github.com/Hzzone/ProPos),
  [ProPos paper mirror](https://www.researchgate.net/publication/364639634_Learning_Representation_for_Clustering_Via_Prototype_Scattering_and_Positive_Sampling)

#### ImageNet-Dogs

- Recommended acquisition: `manual prepared`
- Recommended source: ImageNet subsets commonly used in deep clustering papers and their accompanying implementations.
- Notes: Like `ImageNet-10`, this subset is typically prepared from `ImageNet-1K` using the class selection rules of the original paper or repository. The benchmark generates missing subset directories using its bundled wnid list, as described above.
- The description of common practice is an inference from public clustering repositories and papers.
- Sources:
  [ProPos README](https://github.com/Hzzone/ProPos),
  [ProPos paper mirror](https://www.researchgate.net/publication/364639634_Learning_Representation_for_Clustering_Via_Prototype_Scattering_and_Positive_Sampling)

### Challenging datasets

#### DTD

- Recommended acquisition: `native auto-download`
- Recommended source: `torchvision.datasets.DTD`
- Notes: Supports `download=True` and provides a dataset with varied textures.
- Source:
  [Torchvision DTD](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.DTD.html)

#### UCF101

- Recommended acquisition: `manual prepared`
- Recommended source: `torchvision.datasets.UCF101` and the official videos and annotations.
- Notes: The original data consists of videos rather than ordinary image classification directories. `torchvision` provides a reader, not a one-step download of all videos and annotations. The benchmark currently reads extracted frame images organized into class directories, as described in the data layout documentation.
- Source:
  [Torchvision UCF101](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.UCF101.html)

#### CIFAR-100

- Recommended acquisition: `native auto-download`
- Recommended source: `torchvision.datasets.CIFAR100`
- Notes: Supports `download=True`.
- Source:
  [Torchvision CIFAR100](https://docs.pytorch.org/vision/0.23/generated/torchvision.datasets.CIFAR100.html)

### Large-scale datasets

#### Places365-Standard

- Recommended acquisition: `native auto-download`
- Recommended source: `torchvision.datasets.Places365`
- Notes: The documentation describes downloading official data components. Be aware of the distinction between `train-standard` and other splits when preparing the data.
- Source:
  [Torchvision Places365](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Places365.html)

#### ImageNet-1K

- Recommended acquisition: `manual prepared`
- Recommended source: `torchvision.datasets.ImageNet`
- Notes: `torchvision` requires the relevant `ILSVRC2012` files to be downloaded manually and placed in the root directory before its dataset class can parse them. The benchmark itself reads prepared class directories through ImageFolder, as described in the data layout documentation.
- Source:
  [Torchvision ImageNet](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.ImageNet.html)

### Fine-grained datasets

#### Aircraft

- Recommended acquisition: `native auto-download`
- Recommended source: `torchvision.datasets.FGVCAircraft`
- Notes: Supports `download=True` and can be used directly as a fine-grained benchmark dataset.
- Source:
  [Torchvision FGVCAircraft](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.FGVCAircraft.html)

#### Flowers

- Recommended acquisition: `native auto-download`
- Recommended source: `torchvision.datasets.Flowers102`
- Notes: Supports `download=True`. The documentation also specifies that `scipy` is required to read labels.
- Source:
  [Torchvision Flowers102](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html)

#### Food

- Recommended acquisition: `native auto-download`
- Recommended source: `torchvision.datasets.Food101`
- Notes: Supports `download=True`.
- Source:
  [Torchvision Food101](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html)

#### Cars

- Recommended acquisition: `manual prepared`
- Recommended source: The official `Stanford_Cars.tar` archive.
- Notes: The benchmark first looks for the commonly used extracted layout,
  `data/cars/train/<class_name>/*` and `data/cars/test/<class_name>/*`. It also supports the
  older metadata layout, `data/cars/stanford_cars/`, compatible with
  `torchvision.datasets.StanfordCars`, as a fallback.
- Source:
  [Torchvision StanfordCars](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html)

#### Pets

- Recommended acquisition: `native auto-download`
- Recommended source: `torchvision.datasets.OxfordIIITPet`
- Notes: Supports `download=True`.
- Source:
  [Torchvision OxfordIIITPet](https://docs.pytorch.org/vision/0.26/generated/torchvision.datasets.OxfordIIITPet.html)

### Domain-shift datasets

#### ImageNet-A

- Recommended acquisition: `external scripted`
- Recommended source: TFDS `imagenet_a` or the official repository.
- Notes: This does not use the native `torchvision download=True` workflow. It can be acquired through TFDS or scripts using official project resources.
- Sources:
  [TFDS ImageNet-A](https://www.tensorflow.org/datasets/catalog/imagenet_a),
  [Natural Adversarial Examples repo](https://github.com/hendrycks/natural-adv-examples)

#### ImageNet-Sketch

- Recommended acquisition: `external scripted`
- Recommended source: TFDS `imagenet_sketch` or the official repository.
- Notes: Suitable for evaluating domain shift to sketches. Acquisition uses external resources rather than native `torchvision` downloading.
- Sources:
  [TFDS ImageNet-Sketch](https://www.tensorflow.org/datasets/catalog/imagenet_sketch),
  [ImageNet-Sketch repo](https://github.com/HaohanWang/ImageNet-Sketch)

#### ImageNet-R

- Recommended acquisition: `external scripted`
- Recommended source: TFDS `imagenet_r` or the official repository.
- Notes: Like `ImageNet-A`, this is categorized as acquisition through external scripts. The benchmark also provides optional automatic downloading and directory preparation, as described above.
- Sources:
  [TFDS ImageNet-R](https://www.tensorflow.org/datasets/catalog/imagenet_r),
  [ImageNet-R repo](https://github.com/hendrycks/imagenet-r)

#### ImageNet-V2

- Recommended acquisition: `external scripted`
- Recommended source: TFDS `imagenet_v2` or the official repository.
- Notes: The TFDS documentation lists several variant configurations. The benchmark defaults to the matched-frequency variant; the parameter and supported alternatives are listed above.
- Sources:
  [TFDS ImageNet-V2](https://www.tensorflow.org/datasets/catalog/imagenet_v2),
  [ImageNetV2 repo](https://github.com/modestyachts/ImageNetV2)

#### ImageNet-C (gaussian_noise, level 1)

- Recommended acquisition: `manual prepared`
- Recommended source: The official Hendrycks robustness page or repository.
- Notes: Download the full `ImageNet-C` dataset, then use the `severity=1` portion of the `gaussian_noise` corruption. To match published experimental settings, use the officially released corrupted images rather than generating them dynamically at runtime.
- Sources:
  [Hendrycks robustness page](https://danhendrycks.com/robustness/),
  [hendrycks/robustness](https://github.com/hendrycks/robustness)

## Acquisition summary

- `native auto-download`:
  `CIFAR-10`, `CIFAR-20`, `STL-10`, `DTD`, `CIFAR-100`, `Places365-Standard`, `Aircraft`, `Flowers`, `Food`, `Pets`
- `external scripted`:
  `ImageNet-A`, `ImageNet-Sketch`, `ImageNet-R`, `ImageNet-V2`
- `manual prepared`:
  `ImageNet-10`, `ImageNet-Dogs`, `UCF101`, `ImageNet-1K`, `Cars`, `ImageNet-C (gaussian_noise, level 1)`

## Shared text resource

The 20 dataset count above refers to image datasets. Language-assisted methods
also use the bundled [WordNetNouns.csv](../data/WordNetNouns.csv), copied unchanged
from the official TAC repository. It contains 146,347 rows with `word` and
`definition` columns; the benchmark consumes the noun column. This shared lexical
resource is not an additional image benchmark dataset.

See the [text-data instructions](../data/README.md#text-data) for the exact source
commit, checksum verification, WordNet license notice, and custom-path settings.

## Notes

- This document is a reference for dataset acquisition strategies. It does not define the code interfaces exhaustively.
- The classification of `ImageNet-10` and `ImageNet-Dogs` partly reflects inferences from papers and public repository practices. The benchmark uses its bundled wnid lists; check these against the original method repository or paper when comparing experimental settings.
- `UCF101` currently uses the benchmark's shared image interface with manually prepared frame images. The original video dataset still requires frame extraction and directory preparation before use.
