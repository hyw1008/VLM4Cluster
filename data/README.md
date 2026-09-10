# Data Layout

The benchmark uses the `data/<dataset_registry_name>/...` directory convention.
Except for text resources stored as a single file, such as `WordNetNouns.csv`, directory names
match the dataset registry names shown by `python3 main.py list-datasets` by default.

## Automatically downloaded datasets

The code downloads the following datasets to `data/<dataset_registry_name>/` by default:

```text
data/
├── aircraft/
├── cifar10/
├── cifar20/
├── cifar100/
├── dtd/
├── flowers/
├── food/
├── pets/
├── places365_standard/
└── stl10/
```

For `cifar20`, the code downloads `CIFAR-100` and maps its coarse labels to 20 superclasses.

## Manually prepared datasets

Organize manually prepared datasets as shown below. All ImageFolder-based datasets use:

```text
data/<dataset_name>/<split>/<class_name>/*
```

## ImageNet

```text
data/
└── imagenet/
    ├── train/
    │   ├── class_000/
    │   └── class_001/
    └── val/
        ├── class_000/
        └── class_001/
```

The framework uses `torchvision.datasets.ImageFolder` to read manually prepared data.
It requires the directory layout shown above and does not depend on the official devkit.

## ImageNet-10 and ImageNet-Dogs

The benchmark treats both datasets as standard subsets extracted from `ImageNet-1K`.

- The standard wnid list for `ImageNet-10` is in [imagenet10_wnids.txt](imagenet_subsets/imagenet10_wnids.txt).
- The standard wnid list for `ImageNet-Dogs` is in [imagenet_dogs_wnids.txt](imagenet_subsets/imagenet_dogs_wnids.txt).

The benchmark uses these two lists to define subset extraction. They come from the following
files in the official Contrastive Clustering repository:

- `datasets/ImageNet-10.txt`
- `datasets/ImageNet-dogs.txt`

The recommended approach is to prepare the full `data/imagenet/` directory first, then let
the benchmark generate the subset directories when running `imagenet10` or `imagenet_dogs`.
By default, it creates symlinks without duplicating image files.

When you run `image_dataset.name = "imagenet10"` or `image_dataset.name = "imagenet_dogs"`,
the code checks the subset directories below. If a directory or any required wnids are missing,
it fills them from `data/imagenet/<split>/<wnid>` using the lists in `data/imagenet_subsets/*.txt`:

```text
data/
├── imagenet10/
│   ├── train/
│   │   ├── <wnid>/
│   │   └── ...
│   └── val/
│       ├── <wnid>/
│       └── ...
└── imagenet_dogs/
    ├── train/
    │   ├── <wnid>/
    │   └── ...
    └── val/
        ├── <wnid>/
        └── ...
```

The source and destination paths are:

- `data/imagenet/train/<wnid>/...` -> symlink or copy to `data/imagenet10/train/<wnid>/...` or `data/imagenet_dogs/train/<wnid>/...`
- `data/imagenet/val/<wnid>/...` -> symlink or copy to `data/imagenet10/val/<wnid>/...` or `data/imagenet_dogs/val/<wnid>/...`

Symlinks are the default. If your cluster does not allow them, set the dataset parameter
`imagenet_subset_mode = "copy"` to copy directories instead. If you have already prepared
complete subset directories, the code uses them directly without requiring the full
`data/imagenet/` directory.

## Domain-shift datasets

ImageNet variants serve as target evaluation datasets and use the following paths:

```text
data/
├── imagenet_a/
│   └── test/
│       ├── <class_name>/
│       └── ...
├── imagenet_sketch/
│   └── test/
│       ├── <class_name>/
│       └── ...
├── imagenet_r/
│   └── test/
│       ├── <class_name>/
│       └── ...
├── imagenet_v2/
│   └── test/
│       ├── <class_name>/
│       └── ...
└── imagenet_c/
    └── gaussian_noise/
        └── 1/
            └── test/
                ├── <class_name>/
                └── ...
```

The code also supports automatic preparation of `imagenet_v2`:

- If `data/imagenet_v2/test/<wnid>/*` already exists, the code uses the existing directory.
- If the directory is missing and `image_dataset.download` is not explicitly set to `false`,
  the loader downloads the `matched-frequency` ImageNet-V2 archive by default.
- After downloading, it maps the archive's numeric class directories to `wnid` directories
  and organizes them as:
  `data/imagenet_v2/test/<wnid>/*`
- This mapping requires a locally prepared `data/imagenet/val` directory, falling back to
  `data/imagenet/train` if needed. The original ImageNet-V2 archive uses numeric class indices
  `0..999`, whereas the benchmark organizes classes by `wnid`.

Optional parameters:

- `image_dataset.params.imagenet_v2_variant = "matched-frequency" | "threshold0.7" | "top-images"`
- `image_dataset.params.imagenet_v2_materialize_mode = "symlink" | "copy"`
- `image_dataset.params.imagenet_v2_archive_path = "/path/to/local/archive.tar.gz"`
- `image_dataset.params.imagenet_v2_download_url = "https://..."`
- `image_dataset.params.imagenet_v2_reference_root = "data/imagenet/val"`

The default variant is `matched-frequency`, and the default materialization mode is symlinks.

The code also supports automatic preparation of `imagenet_r`:

- If `data/imagenet_r/test/<wnid>/*` already exists, the code uses the existing directory.
- If the directory is missing and `image_dataset.download` is not explicitly set to `false`,
  the loader downloads the official `imagenet-r.tar` archive by default and organizes it under
  `data/imagenet_r/test/<wnid>/*`.
- Optional parameters:
  - `image_dataset.params.imagenet_r_archive_path = "/path/to/imagenet-r.tar"`
  - `image_dataset.params.imagenet_r_download_url = "https://..."`
  - `image_dataset.params.imagenet_r_materialize_mode = "symlink" | "copy"`

## UCF101 frame images

The benchmark reads `ucf101` as image classification directories containing extracted frames:

```text
data/
└── ucf101/
    ├── train/
    │   ├── <class_name>/
    │   └── ...
    ├── val/
    │   ├── <class_name>/
    │   └── ...
    └── test/
        ├── <class_name>/
        └── ...
```

## Cars

For `cars`, the loader first looks for the ImageFolder-style layout commonly used by the
official `Stanford_Cars.tar` archive:

```text
data/
└── cars/
    ├── train/
    │   ├── <class_name>/
    │   └── ...
    └── test/
        ├── <class_name>/
        └── ...
```

The code also supports the older layout compatible with `torchvision.datasets.StanfordCars`
as a fallback:

```text
data/cars/stanford_cars/
├── cars_train/
├── cars_test/
├── cars_test_annos_withlabels.mat
└── devkit/
```

## Text data

The shared `wordnet` resource is included in the repository:

```text
data/WordNetNouns.csv
```

All language-assisted methods that use a WordNet noun pool read this shared
vocabulary. It is an unmodified copy of the file shipped in the
[official TAC repository](https://github.com/XLearning-SCU/2024-ICML-TAC/blob/830e02c625d98784dfe97bb57226e94bf53413de/data/WordNetNouns.csv)
for [Image Clustering with External Guidance](https://arxiv.org/abs/2310.11989).

| Property | Value |
| --- | --- |
| Upstream commit | `830e02c625d98784dfe97bb57226e94bf53413de` |
| Upstream path | `data/WordNetNouns.csv` |
| Size | 12,294,872 bytes |
| CSV schema | UTF-8; `word,definition` |
| Data rows | 146,347 |
| SHA-256 | `deff9dec810a2ff5c8012c61b5e052e7ba5e62476da9acb6ce8e68df01fd8e5d` |

The benchmark reads the `word` column. The definitions, original row order, and
file bytes are retained; this resource is not regenerated from NLTK, deduplicated,
or replaced with another noun list. On case-sensitive filesystems, use the exact
filename `WordNetNouns.csv`.

The [source manifest](WordNetNouns.source.json) records the immutable download URL
and checksums. The accompanying [WordNet license notice](WordNet-LICENSE.txt)
reproduces the license published by
[Princeton University](https://wordnet.princeton.edu/license-and-commercial-use).
The TAC repository does not specify which WordNet release generated this CSV and
has no separate repository-level license at the recorded commit. The WordNet
notice concerns the lexical resource; it does not establish a license for TAC's
program code.

If the file needs to be restored, download the exact `source_url` from the
manifest to `data/WordNetNouns.csv`, then verify it from the repository root:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path("data/WordNetNouns.source.json").read_text())
payload = Path(manifest["path"]).read_bytes()
assert len(payload) == manifest["size_bytes"], "WordNet file size mismatch"
assert hashlib.sha256(payload).hexdigest() == manifest["sha256"], "WordNet checksum mismatch"
print("Verified TAC WordNetNouns.csv")
PY
```

With a custom image-data root, point language-assisted methods to the bundled
file explicitly using `method.wordnet_csv=data/WordNetNouns.csv`. A separately
configured runner text dataset uses `text_dataset.params.wordnet_csv` instead.

## AnyAttack checkpoint

For adversarial evaluation, download `coco_cos.pt` from the authors'
[checkpoint repository](https://huggingface.co/jiamingzz/anyattack/blob/main/checkpoints/coco_cos.pt),
linked by the [official AnyAttack repository](https://github.com/jiamingzhang94/AnyAttack),
and place it at:

```text
data/anyattack/checkpoints/coco_cos.pt
```

This checkpoint and image datasets are not bundled. To use a different local
checkpoint path, set `attack.decoder_path`. See the
[reproduction guide](../docs/reproduction.md#adversarial-evaluation) for a complete
command and the clean-training/adversarial-evaluation protocol.
