# Reproduction Guide

This guide connects the required resources, the existing `main.py` entry point,
method variants, and output artifacts. Commands run the current VLM4Cluster
implementations with their existing defaults and fixed profiles. Reproducing a
particular reported score additionally requires that result's code revision,
configuration, seed, data split, and environment; a generic command alone does
not establish those details.

See [methods and sources](methods.md) for the core 16-method scope and adaptations,
and [method hyperparameters](method_hyperparameters.md) for all parameters. No
per-method upstream checkout or private training script is needed.

## Environment and resources

Run all commands from the repository root. Install and activate the environment:

```bash
bash setup_env.sh
conda activate vlm4cluster
```

This setup targets the CUDA environment described in the
[installation instructions](../README.md#installation). For CPU installation and
FAISS compatibility, use the [troubleshooting guide](troubleshooting.md), then pass
`runtime.device=cpu`. A CPU fallback does not make full benchmark runs inexpensive:
feature extraction, training, and graph construction can require substantial time
and memory.

| Resource | Required by | Preparation |
| --- | --- | --- |
| Image datasets | Every method | Use the [data layout](../data/README.md) and [dataset sources](datasets.md). The CIFAR-10 examples below download through the loader when missing. |
| OpenCLIP checkpoint and tokenizer | Every core method | Loaded through the shared model loader; the first run needs access to uncached weights. See the exact checkpoint table below. |
| `data/WordNetNouns.csv` | SIC, TAC, SAC, GradNorm, NTK-SC, SEIC, MAGIC | Included in this repository from official TAC. [Source, hash verification, and license](../data/README.md#text-data). |
| `data/anyattack/checkpoints/coco_cos.pt` | AnyAttack evaluation only | Download the author-provided checkpoint following the [resource instructions](../data/README.md#anyattack-checkpoint). |
| SPAMS | EnSC only when `method.algorithm=spams` is selected | Optional external solver. The existing default is `lasso_lars`, which uses scikit-learn. |

Training methods create their heads within this codebase. The core examples do
not require upstream IDC/TCL checkpoints, PRO-DSC `.pt` feature exports, BLIP-2 or
Qwen caption files, or SBERT embeddings. The reasons for these differences are in
the [implementation notes](methods.md#what-the-benchmark-implementation-runs).

For image datasets stored elsewhere, pass `image_dataset.root=/path/to/data`.
Keep the bundled vocabulary available to language-assisted methods with
`method.wordnet_csv=data/WordNetNouns.csv` (or an absolute path). This method
parameter selects the noun pool without requiring a separate runner-level
`text_dataset` or `text_features` configuration.

## Inspect a run before execution

```bash
python3 main.py list-methods
python3 main.py list-datasets
python3 main.py dataset-stats cifar10 --root data
```

Dataset statistics inspect local files unless `--download` is supplied. To parse
a run configuration and write its plan without downloading data/weights or
running a method:

```bash
python3 main.py run --config configs/experiments/base.toml --dry-run \
  dataset=cifar10 method=tac seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv method.train_cluster_heads=true
```

A dry run checks configuration and planning; it does not verify that model
weights, datasets, optional solvers, or GPU memory are available. Remove
`--dry-run` to execute. The framework creates `plan.json` and a trial record even
for a dry run, so the command is not an artifact-free validation step.

## Run the core methods

The following commands select **CIFAR-10**, **LAION400M + ViT-B/32**, and **seed 42**
explicitly. They retain existing method defaults and dataset/model profiles.
Training methods use the source training split and evaluate on the test split;
the four classical baselines cluster evaluation features directly. Selection and
supervision differences are listed [below](#evaluation-and-variant-choices).

Run the commands individually as needed; the blocks are not a lightweight smoke
test or a requirement to run every method consecutively.

### Classical methods

```bash
# K-means
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=clip_kmeans seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32

# Spectral Clustering
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=clip_sc seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32

# SSC-OMP
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=ssc_omp seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32

# EnSC
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=ensc seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32
```

### Deep methods

```bash
# IDC (training labels simulate interactive oracle feedback)
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=idc seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32

# SCAN
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=scan seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32

# CPP (default test-metric epoch selection; see the evaluation notes)
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=cpp seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32

# TEMI
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=temi seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32

# PRO-DSC
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=pro_dsc seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32
```

### Language-assisted methods

```bash
# SIC
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=sic seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv

# TAC: training-free
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=tac seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv method.train_cluster_heads=false

# TAC*: train cluster heads
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=tac seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv method.train_cluster_heads=true

# SAC
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=sac seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv

# GradNorm
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=gradnorm seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv

# NTK-SC
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=ntk_sc seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv

# SEIC: Stage 1, the current default benchmark variant
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=seic seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv method.stage2=false

# MAGIC
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=magic seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv
```

The extra TAC command is a variant, not a seventeenth method.

## Select a dataset, model, and seed

Replace `dataset=cifar10` with a registered dataset name from
[the dataset guide](datasets.md). Do not concatenate training and test data to
match an upstream script: use the benchmark split definitions. Profiles resolve
from the existing dataset/pretraining/backbone rules; explicit
`method.<parameter>=<value>` overrides take precedence.

The shared [model registry](../vlm4cluster/models/openclip.py) maps seven supported
combinations to these exact OpenCLIP identifiers:

| `openclip_pretraining` | `openclip_backbone` | OpenCLIP `model_name` | OpenCLIP `pretrained` |
| --- | --- | --- | --- |
| `LAION400M` | `ViT-B/32` | `ViT-B-32-quickgelu` | `laion400m_e32` |
| `LAION400M` | `ViT-B/16` | `ViT-B-16` | `laion400m_e32` |
| `LAION400M` | `ViT-L/14` | `ViT-L-14` | `laion400m_e32` |
| `LAION2B` | `ViT-B/32` | `ViT-B-32` | `laion2b_s34b_b79k` |
| `LAION2B` | `ViT-B/16` | `ViT-B-16` | `laion2b_s34b_b88k` |
| `LAION2B` | `ViT-L/14` | `ViT-L-14` | `laion2b_s32b_b82k` |
| `SigLIP` | `ViT-B/16` | `ViT-B-16-SigLIP` | `webli` |

These are seven registered pairs, not a Cartesian product. Image and text towers
come from the same selected checkpoint. The loader's preprocessing determines
image preparation; avoid silently changing resolution or transforms. LAION and
SigLIP differ in data and training recipe as well as loss, so this comparison does
not isolate the causal effect of the loss alone.

For example, run the trained TAC variant with SigLIP B/16:

```bash
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=tac seed=42 \
  method.openclip_pretraining=SigLIP method.openclip_backbone=ViT-B/16 \
  method.wordnet_csv=data/WordNetNouns.csv method.train_cluster_heads=true
```

Use `seed=<integer>` for an additional repetition. Report the actual seed list,
per-seed results, and aggregation rule; one seed-42 run is not a mean over seeds.
The seed is propagated to methods unless a method-level seed/random-state
override is supplied. Equal seeds do not guarantee bitwise equality across
different library versions or CPU/GPU backends.

## Evaluation and variant choices

These are properties of the current implementations, not changes introduced by
the reproduction guide:

| Method or option | Interpretation |
| --- | --- |
| IDC | Training labels simulate oracle feedback; account for its interactive supervision budget when comparing methods. |
| SCAN | Default head/epoch selection uses SCAN loss on evaluation features. It is label-free selection, but not selection isolated from the evaluation split. |
| CPP | Defaults to two heads and `select_best_epoch=true`, selecting predictions by test/evaluation ACC. Setting `method.select_best_epoch=false` is an explicit alternative protocol and must be reported as such. |
| TAC / TAC* | Preserve the explicit `train_cluster_heads` flag with the result. |
| SEIC | `stage2=false` runs Stage 1 only. `method.stage2=true` enables self-enhancement; it is a separate variant that updates visual LoRA parameters. |
| Spectral Clustering | The default graph is dense. `method.graph_k=30 method.affinity_tau=0.04` selects the documented sparse setting and changes the experiment. |
| NTK-SC | Record the actual diffusion backend and graph parameters selected for the run, especially when comparing time and memory. |
| `internal_metrics_only=true` | Reports SIL/DBI/CHI in the final evaluator. It does not alter the above training or selection choices, or remove the assumed cluster count. |

Small sample limits can be useful for debugging, but should be labeled as such.
They change the evaluation sample set and can also affect clustering dimensions
and data-dependent profiles. In particular, PRO-DSC requires a valid effective
training batch relative to the cluster count. Do not add sample limits or reduce
epochs to a score-reproduction command without recording that change.

## ImageNet distribution shifts

Prepare both ImageNet and the target dataset following the
[data layout](../data/README.md). Keep ImageNet as the configured source dataset
and select the target explicitly. For trained TAC:

```bash
# Optional first run: save an ImageNet-trained cluster-head checkpoint.
python3 main.py run --config configs/experiments/base.toml \
  dataset=imagenet method=tac seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv method.train_cluster_heads=true \
  method.save_checkpoint=true

# Evaluate on ImageNet-A using the same source-side setup.
python3 main.py run --config configs/experiments/base.toml \
  dataset=imagenet method=tac seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv method.train_cluster_heads=true \
  method.target_dataset_name=imagenet_a
```

Replace the target with `imagenet_c`, `imagenet_r`, `imagenet_v2`, or
`imagenet_sketch` for the other variants. ImageNet-C uses Gaussian noise severity
1, and ImageNet-V2 defaults to `matched-frequency`.

Noun selection and training use ImageNet training data. Source-side profiles and
the default 1,000-way trained heads remain tied to ImageNet; target class counts
must not silently redefine them. If a matching ImageNet checkpoint exists,
training methods attempt to reuse it by default. Otherwise, they train. Saving is
opt-in and checkpoint IO is restricted to a full ImageNet source. Use
`method.force_retrain=true` or `method.load_checkpoint=false` for a fresh run.

The four classical baselines currently cluster the evaluation features directly,
including when a target dataset is selected. That execution is not a
source-trained transfer experiment. Also retain each training method's documented
selection protocol when interpreting distribution-shift results.

## Adversarial evaluation

Prepare the [AnyAttack checkpoint](../data/README.md#anyattack-checkpoint), then:

```bash
python3 main.py run --config configs/experiments/base.toml \
  dataset=cifar10 method=tac seed=42 \
  method.openclip_pretraining=LAION400M method.openclip_backbone=ViT-B/32 \
  method.wordnet_csv=data/WordNetNouns.csv method.train_cluster_heads=true \
  attack=anyattack attack.decoder_path=data/anyattack/checkpoints/coco_cos.pt \
  attack.eps=0.03137254901960784
```

By default, AnyAttack modifies only test/validation images and uses an
`epsilon=8/255` budget, matching the recorded leaderboard setting and the explicit
decimal value in the command above. The training split stays clean. Generated
images live under `data/adversarial/anyattack/`, and attack settings are included in feature
and method cache identities. Preserve the checkpoint identity, attack parameters,
target-selection policy, and seed with the reported result. Attack target
selection uses the configured `avoid_same_label` policy, which defaults to true;
this attack configuration should not be described as entirely label-agnostic.

## Read and preserve the result

The CLI prints the completed `report.json` path. Under the default output layout,
each trial has its own fingerprint and evaluation-mode directory:

```text
outputs/<method>/<dataset>/<openclip>/seed_<seed>/
└── trials/trial_<fingerprint>/
    ├── trial.json
    └── external/                  # internal/ for internal_metrics_only=true
        ├── plan.json
        ├── predictions.npz
        └── report.json
```

`report.json` contains the final metrics, prediction configuration, method result metadata,
and artifact paths. Retain `trial.json`, `plan.json`, predictions, and the report
together. A dry-run plan is not a completed measurement. Never fill a missing
result with zero or with a number copied from an original paper.

To make a reported score independently reproducible, preserve:

1. The VLM4Cluster commit, exact command, and base TOML used for that run.
2. The environment/package versions, CUDA/driver versions, and hardware.
3. Dataset versions, splits, class/subset mappings, sample limits, and image
   preprocessing, including the ImageNet-C/V2 choices where relevant.
4. OpenCLIP identifiers, vocabulary checksum, actual resolved profiles, method
   variants, seed, and any checkpoint reuse or selection procedure.
5. The completed report and prediction artifacts, plus any aggregation rule used
   to turn multiple runs into one leaderboard entry.

The source-audit snapshots in [methods.md](methods.md) identify external references;
they are not substitutes for the VLM4Cluster commit that generated a result.

Only ImageNet-1K runs produce the framework's `efficiency` measurements; other
datasets report `null`. A checkpoint-reuse run must not be presented as full
training-and-evaluation timing. See the
[measurement definitions](../README.md#efficiency-measurements) for timing and
memory scopes.
