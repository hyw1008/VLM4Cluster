# Method Hyperparameters

This document collects hyperparameter information for the methods currently integrated into the benchmark, to support consistent organization, experiment configuration checks, and further documentation.

For the core method list, papers, upstream repositories, checked licenses, and
adaptation summary, see [methods and sources](methods.md). For executable commands
and required resources, see the [reproduction guide](reproduction.md).

## Global seed rules

- The top-level `seed` in the experiment configuration is used by the runner to set the Python / NumPy / PyTorch / CUDA random seeds consistently.
- The runner injects the top-level `seed` as the default for each method's `method.params.seed` and `method.params.random_state`; explicitly provided method-level `seed` or `random_state` values take precedence.
- For standard experiments, setting `seed = 42` at the top level of the TOML configuration is sufficient; there is no need to repeat `method.seed=42` or `method.random_state=42` for each method.

## Model combinations and fixed hyperparameter profile fallback

- `openclip_pretraining="SigLIP"` with `openclip_backbone="ViT-B/16"` selects SigLIP v1 B/16-224.
- Model-specific fixed hyperparameter profiles require an exact match on the full `pretraining + backbone + dataset` combination;
  the currently registered entries cover only the three `LAION400M` backbones. Therefore, `LAION2B` does not inherit
  the backbone-specific tuned overrides for `LAION400M`, but still uses dataset-only profiles and
  paper or official default profiles that are independent of the pretrained model; explicit CLI parameters always take precedence.
- No new method-level fixed hyperparameter profiles are currently added for `SigLIP + ViT-B/16`. For `SCAN`, `EnSC`, `SIC`, `CPP`, `PRO-DSC`, and `TEMI`, SigLIP explicitly skips dataset/model profiles added after initial integration,
  preserving each method's current generic fallback. This isolation rule applies only to SigLIP and must not be extended to all
  `pretraining != LAION400M` settings, as that would incorrectly remove the dataset defaults that LAION2B should use.
- An audit on 2026-08-15 confirmed that the methods above had previously grouped LAION2B and SigLIP under the same generic
  fallback. This has been corrected so that LAION2B uses dataset profiles while SigLIP remains isolated; the other methods with model
  profiles already used exact matching on the full key and were unaffected.
- Dataset defaults that were present at initial integration and belong to the original method logic are retained, including the
  semantic cluster counts used by TAC and methods reusing its logic, and the original GradNorm defaults.
  These defaults do not constitute leakage from LAION/benchmark profiles added later.
- Among the methods affected by fallback isolation that reuse training checkpoints, only SigLIP uses the
  `generic-openclip-defaults-v1` artifact version; SCAN's SigLIP fallback uses
  `generic-openclip-defaults-v2`. LAION2B has returned to the normal dataset-profile artifact paths, preventing accidental
  loading of LAION2B checkpoints saved in generic version directories before the fix; LAION400M paths are unchanged.

## Training checkpoint saving rules

- All methods that require training expose `save_checkpoint` (with the compatibility alias `save_checkpoints`), defaulting to `False`; this switch only takes effect when the source/training dataset is full `ImageNet`.
- When the source/training dataset is not full `ImageNet`, model checkpoints are neither saved nor loaded; existing frequency parameters such as `save_every`, `checkpoint_every`, `save_checkpoint_frequency`, and `saveckp_freq` are treated as disabled.
- With `save_checkpoint=False`, no model checkpoints are saved; existing frequency parameters are also treated as disabled.
- With `save_checkpoint=True`, methods with periodic saving logic (CPP, PRO-DSC, and TEMI) continue to save at the corresponding frequency; if no frequency is specified, the enabled-state default documented for each method is used.
- With `save_checkpoint=True`, training methods without existing periodic checkpointing, such as TAC, SAC, SIC, SEIC, SCAN, and IDC, save a checkpoint of the final training state.
- When the source/training dataset is full `ImageNet`, training methods attempt to reuse existing checkpoints by default: `load_checkpoint=True` (with the compatibility aliases `reuse_checkpoint`/`reuse_checkpoints`). If a checkpoint matching the current source-side training settings is found, training is skipped and only inference and evaluation on the current evaluation dataset are performed.
- To force retraining, set `force_retrain=True` or `load_checkpoint=False`.
- In domain-shift experiments, checkpoint reuse still follows the source-side training settings: first look for a checkpoint matching the full current settings, then, where possible, fall back to a checkpoint saved when evaluating on the same source dataset itself; the target dataset affects only the final evaluation.

## CLIP-KMeans

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `test_split`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the default evaluation split of the target dataset
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Clustering parameters:
  `image_batch_size` defaults to `256`
  `kmeans_niter` defaults to `300`
  `kmeans_nredo` defaults to `20`
- Notes:
  `CLIP-KMeans` is the most direct evaluation-only baseline: extract the shared OpenCLIP image features for the evaluation split, then apply K-Means directly.

## CLIP-SC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `test_split`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the default evaluation split of the target dataset
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Spectral clustering parameters:
  `image_batch_size` defaults to `256`
  `affinity_kernel` defaults to `dot_product`: build the full Gram matrix of the normalized evaluation features and set its diagonal to zero. Set `affinity_kernel=rbf` (or the legacy alias `affinity=rbf`) to select the previous RBF construction
  `graph_k`, with the compatibility aliases `n_neighbors`/`q`, defaults to `M-1` (`M` is the number of evaluation samples). Smaller neighbor counts require `affinity_kernel=rbf`; the dot-product setting uses all sample pairs
  `affinity_tau`, with the compatibility alias `tau`, defaults to `1.0` for the RBF kernel only; it is not used by the dot-product kernel
  For the RBF kernel, the legacy parameter `gamma` is interpreted as `affinity_tau = 1 / gamma`; if `affinity_tau`/`tau` is also provided, the temperature parameter takes precedence
  `assign_labels` defaults to `discretize`
  `n_init` defaults to `10`
  `force_recompute_affinity` defaults to `False`; setting it to `True` ignores the existing CLIP-SC affinity graph cache
  `spectral_random_state`, or its alias `sc_random_state`, defaults to the runner-injected `random_state`/`seed`, which is `42` in standard experiments
- Notes:
  `CLIP-SC` is an evaluation-only baseline. With features stored as rows in `X`, the default affinity is `A = X @ X.T`, followed by `A[i,i]=0`. Features remain L2-normalized, so off-diagonal values are cosine similarities. There is no exponential transform, neighbor pruning, clipping, or shift. CUDA runs compute the Gram matrix in row blocks with a NumPy fallback; the spectral solver remains on CPU.
  Raw dot products can be negative. The standard normalized-cut interpretation assumes nonnegative affinities; this implementation preserves the requested raw weights, and rejects negative or non-finite degree sums before Laplacian normalization. The RBF kernel remains available when a nonnegative affinity is required.
  The explicit RBF setting retains an edge only between mutual top-q neighbors, with weight `exp(-||z_i-z_j||^2 / affinity_tau)` and zero diagonal. With `q=M-1`, it is the previous full RBF graph. The earlier sparse setting can be selected with `affinity_kernel=rbf, graph_k=30, affinity_tau=0.04`.
  Dot-product caches use the `full_dot_product_l2_zero_diag_v1` token and cannot reuse old RBF graphs. The resolved kernel also participates in the trial fingerprint, so default dot-product runs write to a different trial directory from previous RBF runs. Shared image features remain reusable. Both kernels use sklearn `SpectralClustering(affinity='precomputed', assign_labels='discretize')`; reports identify the kernel and graph statistics. Historical RBF results and efficiency measurements are not dot-product measurements and require a new run for this setting.

## SCAN

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the default evaluation split of the target dataset
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Default cluster count:
  If `n_clusters` is not explicitly provided, it defaults to the number of classes in the source training dataset; the default for domain shift from full ImageNet is `1000`
- Benchmark workflow:
  The current `scan` freezes the `OpenCLIP` visual backbone by default and reuses the shared OpenCLIP image feature cache;
  it does not run pretext training or self-labeling, and does not apply image augmentation during SCAN training.
  Only the SCAN clustering head module is updated during training.
  The SCAN-specific cache stores only method artifacts such as nearest-neighbor indices by default; with `save_checkpoint=True`, it additionally saves the final selected clustering head checkpoint.
- Neighbor mining and head-only SCAN parameters:
  `num_neighbors` defaults to `20`; for `DTD`, `UCF101`, and `Cars`, it defaults to `50` following the reference feature-only SCAN script.
  `Places365-Standard` uses the large-scale profile default of `20`
  `num_heads` defaults to `1` for `CIFAR10`, `CIFAR20`, `CIFAR100`, `STL10`, `ImageNet10`, and `ImageNet-Dogs`;
  other datasets default to `10` following the reference script's multi-head profile, including the large-scale dataset `Places365-Standard`, which also defaults to `10`
  Override this with `method.num_heads=<N>` or `method.params.num_heads=<N>`
  `selection_num_neighbors` defaults to `5`, matching the official selection logic based on validation-side SCAN loss
  `update_cluster_head_only` is fixed at `True`
  `include_self_neighbors` defaults to `True`, preserving the official behavior of including each sample itself in its nearest-neighbor indices
  `entropy_weight` defaults to `5.0`
  Selection uses `consistency - entropy` following the official `scan_evaluate`, without applying the training-time `entropy_weight` multiplier
  `scan_epochs` defaults to `100`
  small/subset profile (`CIFAR10`, `CIFAR20`, `CIFAR100`, `STL10`, `ImageNet10`, `ImageNet-Dogs`):
  `scan_optimizer=adam`, `scan_learning_rate=1e-4`, `scan_weight_decay=1e-4`, `scan_batch_size=128`
  Fine-grained profile: the defaults are `adam`, `1e-4`, and `1e-4`; `DTD` defaults to `scan_learning_rate=3e-3` based on tuning in this benchmark
  Batch sizes are `32` for `DTD` and `UCF101`, `128` for `Flowers` and `Pets`, `256` for `Aircraft` and `Cars`, and `512` for `Food`
  large-scale profile (`ImageNet`, `ImageNet` variants, `Places365-Standard`):
  `scan_optimizer=sgd`, `scan_learning_rate=30.0`, `scan_weight_decay=0.0`, `scan_momentum=0.9`, `scan_batch_size=4096`
- LAION2B and SigLIP:
  All three `LAION2B` backbones use the source-dataset profiles above, but do not match any backbone-specific tuned overrides registered only for
  `LAION400M`. For example, STL10 uses `num_heads=1`, DTD uses
  `knn=50`, `scan_batch_size=32`, and `scan_learning_rate=3e-3`, while ImageNet uses
  `SGD / lr=30 / batch=4096`.
  `SigLIP + ViT-B/16` continues to skip dataset and LAION400M profiles, retaining the generic defaults of
  `knn=20`, `num_heads=10`, `scan_epochs=100`, `entropy_weight=5.0`.
  Source datasets that are not large-scale continue to use `scan_batch_size=128`, `scan_optimizer=adam`,
  `scan_learning_rate=1e-4`, `scan_weight_decay=1e-4`, `scan_momentum=0`; ImageNet,
  ImageNet variants, and Places365-Standard use optimizer defaults suitable for large-scale data:
  `scan_batch_size=4096`, `scan_optimizer=sgd`, `scan_learning_rate=30.0`,
  `scan_weight_decay=0.0`, `scan_momentum=0.9`. Explicit parameters still take precedence.
- Currently confirmed benchmark parameters for `LAION400M + ViT-B/32`; `default` in the table means that the reported value has been checked against the current implementation and matches its fallback:

  | dataset | `knn` | `head` | `entropy_weight` | `lr` |
  | --- | ---: | ---: | ---: | ---: |
  | CIFAR-10 | default | default | default | 3e-3 |
  | CIFAR-20 | default | default | default | default |
  | CIFAR-100 | default | 10 | 10.0 | default |
  | STL-10 | default | default | default | default |
  | ImageNet-10 | default | default | default | default |
  | ImageNet-Dogs | default | default | default | default |
  | DTD | default | default | default | default |
  | UCF101 | 10 | default | default | default |
  | ImageNet | default | default | default | default |
  | Places365-Standard | default | default | default | default |
  | Aircraft | default | default | default | default |
  | Cars | default | default | default | 1e-2 |
  | Flowers | default | default | default | 3e-3 |
  | Food | default | default | default | default |
  | Pets | default | default | default | 5e-3 |
  | ImageNet-Variants | default | default | default | default |

  Currently confirmed benchmark parameters for `LAION400M + ViT-B/16`; `default` likewise means that the reported value has been checked against the current implementation and matches its fallback:

  | dataset | `knn` | `head` | `entropy_weight` | `lr` |
  | --- | ---: | ---: | ---: | ---: |
  | CIFAR-10 | default | default | default | default |
  | CIFAR-20 | default | default | default | 3e-3 |
  | CIFAR-100 | default | 10 | 10.0 | default |
  | STL-10 | default | default | default | 1e-3 |
  | ImageNet-10 | default | default | default | default |
  | ImageNet-Dogs | default | default | default | 1e-3 |
  | DTD | default | default | default | default |
  | UCF101 | 10 | default | default | default |
  | ImageNet | default | default | default | default |
  | Places365-Standard | default | default | default | default |
  | Aircraft | default | default | default | default |
  | Cars | default | default | default | 1e-2 |
  | Flowers | default | default | default | 3e-3 |
  | Food | default | default | default | default |
  | Pets | default | default | default | 5e-3 |
  | ImageNet-Variants | default | default | default | default |

  Currently confirmed benchmark parameters for `LAION400M + ViT-L/14`; `default` likewise means that the reported value has been checked against the current implementation and matches its fallback:

  | dataset | `knn` | `head` | `entropy_weight` | `lr` |
  | --- | ---: | ---: | ---: | ---: |
  | CIFAR-10 | default | default | default | default |
  | CIFAR-20 | default | default | default | default |
  | CIFAR-100 | default | 10 | 10.0 | default |
  | STL-10 | default | default | default | 1e-3 |
  | ImageNet-10 | default | default | default | default |
  | ImageNet-Dogs | default | default | default | default |
  | DTD | default | default | default | 1e-3 |
  | UCF101 | 10 | default | default | default |
  | ImageNet | default | default | default | default |
  | Places365-Standard | default | default | default | default |
  | Aircraft | default | default | default | default |
  | Cars | default | default | default | 1e-2 |
  | Flowers | default | default | default | 3e-3 |
  | Food | default | default | default | default |
  | Pets | default | default | default | 5e-3 |
  | ImageNet-Variants | default | default | default | default |
- Input size and evaluation parameters:
  Image preprocessing uses the preprocess function from the current `OpenCLIP` bundle, consistent with the shared image feature cache
  `eval_batch_size` defaults to `max(scan_batch_size, 256)`
- Notes:
  This benchmark version retains SCAN's core logic of nearest-neighbor consistency + entropy regularization + optional multi-head selection,
  while replacing the visual backbone with frozen `OpenCLIP` and training only the clustering head module.
  Frozen OpenCLIP image features use the same shared feature cache as other methods;
  changing SCAN training parameters such as `num_heads`, `scan_epochs`, and `scan_learning_rate` does not trigger image feature extraction again.
  To align with official SCAN, the best head and in-memory head state are selected using SCAN loss on the evaluation/validation split by default;
  this differs from a benchmark selection protocol that strictly separates training and test data.

## TAC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `wordnet_csv`/`wordnet_path`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the default evaluation split of the target dataset
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Default cluster count:
  If `n_clusters` is explicitly provided, its explicit value takes precedence
  If `n_clusters` is not explicitly provided, it defaults to the number of classes in the source training dataset; the default for domain shift from full ImageNet is `1000`
  This applies to both `train_cluster_heads=True` and `train_cluster_heads=False`, ensuring that domain shift replaces only the final evaluation dataset
- Text filtering and retrieval parameters:
  `image_batch_size` defaults to `256`
  `text_batch_size` defaults to `2048`
  `retrieval_batch_size` defaults to `8192`
  `selected_nouns_per_center` defaults to `5`
  `retrieval_temperature`, or the official alias `tau`, defaults to `0.005`
  Currently confirmed benchmark parameters for `LAION400M + ViT-B/32`; `default` in the table means that the reported value has been checked against the current implementation and matches its fallback, so it is not explicitly added to the code profile; only non-default values are added to the profile table:

  | dataset | no-train `tau` | train `tau` | train `distill_temperature` |
  | --- | ---: | ---: | ---: |
  | CIFAR-10 | 0.01 | 0.02 | default |
  | CIFAR-20 | 0.02 | default | default |
  | CIFAR-100 | 0.01 | 0.01 | default |
  | STL-10 | 0.01 | 0.01 | default |
  | ImageNet-10 | 0.02 | 0.04 | default |
  | ImageNet-Dogs | 0.01 | 0.01 | default |
  | DTD | 0.01 | 0.01 | default |
  | UCF101 | 0.01 | 0.01 | default |
  | ImageNet | 0.01 | 0.01 | default |
  | Places365-Standard | 0.01 | 0.01 | default |
  | Aircraft | 0.03 | 0.03 | 10.0 |
  | Cars | 0.09 | 0.04 | 9.0 |
  | Flowers | 0.03 | 0.01 | default |
  | Food | 0.01 | 0.01 | default |
  | Pets | 0.01 | 0.01 | default |
  | ImageNet-Variants | 0.01 | 0.01 | default |

  Currently confirmed benchmark parameters for `LAION400M + ViT-B/16`; `default` likewise means that the reported value has been checked against the current implementation and matches its fallback:

  | dataset | no-train `tau` | train `tau` | train `distill_temperature` |
  | --- | ---: | ---: | ---: |
  | CIFAR-10 | 0.02 | 0.02 | default |
  | CIFAR-20 | 0.02 | 0.008 | default |
  | CIFAR-100 | 0.02 | 0.01 | default |
  | STL-10 | 0.1 | 0.05 | default |
  | ImageNet-10 | 0.02 | 0.01 | default |
  | ImageNet-Dogs | 0.02 | 0.01 | default |
  | DTD | 0.02 | 0.05 | 2.0 |
  | UCF101 | 0.03 | 0.03 | default |
  | ImageNet | 0.01 | 0.01 | default |
  | Places365-Standard | 0.01 | 0.01 | default |
  | Aircraft | 0.03 | 0.03 | 3.0 |
  | Cars | 0.04 | 0.05 | 7.0 |
  | Flowers | 0.04 | 0.01 | default |
  | Food | 0.01 | 0.01 | default |
  | Pets | 0.03 | 0.03 | default |
  | ImageNet-Variants | 0.01 | 0.01 | default |

  Currently confirmed benchmark parameters for `LAION400M + ViT-L/14`; `default` likewise means that the reported value has been checked against the current implementation and matches its fallback:

  | dataset | no-train `tau` | train `tau` | train `distill_temperature` |
  | --- | ---: | ---: | ---: |
  | CIFAR-10 | 0.02 | 0.03 | default |
  | CIFAR-20 | 0.08 | 0.09 | default |
  | CIFAR-100 | 0.02 | 0.01 | default |
  | STL-10 | 0.02 | 0.02 | default |
  | ImageNet-10 | 0.03 | 0.01 | default |
  | ImageNet-Dogs | 0.04 | 0.03 | default |
  | DTD | 0.03 | 0.02 | 1.0 |
  | UCF101 | 0.03 | 0.03 | default |
  | ImageNet | 0.01 | 0.01 | default |
  | Places365-Standard | 0.02 | 0.02 | default |
  | Aircraft | 0.03 | 0.05 | 3.0 |
  | Cars | 0.07 | 0.05 | default |
  | Flowers | 0.03 | 0.01 | default |
  | Food | 0.05 | 0.05 | default |
  | Pets | 0.02 | 0.02 | default |
  | ImageNet-Variants | 0.01 | 0.01 | default |

  `semantic_cluster_size` defaults to `300`
  `semantic_clusters` defaults to `None`; for datasets covered by official TAC, the fixed semantic cluster counts in the official `filter_nouns.py` take precedence by default:
  `CIFAR-10=167`, `CIFAR-20=167`, `STL-10=17`, `ImageNet-10=43`, `ImageNet-Dogs=65`, `DTD=141`, `UCF101=303`, `ImageNet=4271`
  Only datasets not covered fall back to inference based on training set size and `semantic_cluster_size`
  `candidate_noun_limit` defaults to `None`
  `force_recompute_nouns` defaults to `False`
- Variant control parameters:
  `train_cluster_heads` defaults to `True`
  When `train_cluster_heads=False`, the following additional parameters are used:
  `kmeans_niter` defaults to `300`
  `kmeans_nredo` defaults to `20`
- Parameters specific to the training variant:
  `epochs` depends on `n_clusters` by default: `100` when `n_clusters >= 100`, otherwise `20`
  `train_batch_size` depends on `n_clusters` by default: `8192` when `n_clusters >= 100`, otherwise `512`
  If the source training split contains fewer samples than `train_batch_size`, the effective training batch size is automatically reduced to the training split size to prevent the cluster head from receiving zero training steps
  `distill_temperature` depends on `n_clusters` by default: `5.0` when `n_clusters >= 100`, otherwise `0.5`
  `learning_rate` defaults to `1e-3`
  `neighbors_topk` defaults to `50`
  `balance_weight` defaults to `5.0`
  `save_checkpoint` defaults to `False`; when enabled, it saves the final TAC cluster-head checkpoint
- Correspondence with the official scripts:
  `selected_nouns_per_center` corresponds to `topK=5` in `filter_nouns.py`
  `retrieval_temperature/tau` corresponds to `tau=0.005` in `retrieve_text.py` and `concat_kmeans.py`
  `neighbors_topk` corresponds to `topK=50` in `train_head.py`
  The official repository uses the name `topK` in both `filter_nouns.py` and `train_head.py` with different meanings; the benchmark separates these into two independent parameters to avoid confusion

## SEIC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `wordnet_csv`/`wordnet_path`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the default evaluation split of the target dataset
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Integration conventions in this benchmark:
  The two-stage main workflow follows the paper `Self-Enhanced Image Clustering with Cross-Modal Semantic Consistency`:
  Stage 1 freezes the OpenCLIP image/text encoders and trains the image/text projection heads and clustering heads;
  Stage 2 injects a local LoRA wrapper into the Q/V paths of OpenCLIP visual self-attention, then uses high-confidence pseudo-labels for self-enhancement by fine-tuning the visual encoder's LoRA parameters and image heads.
  At integration time, an official implementation was not available to this project. Details of `text_weight_temperature`, the fixed cross-modal temperature, and the LoRA wrapper therefore use local interpretations of the paper's equations; the source audit in [methods and sources](methods.md) records the current verification status.
- Text construction parameters:
  `image_batch_size` defaults to `256`
  `text_batch_size` defaults to `2048`
  `candidate_noun_limit` defaults to `None`
  `force_recompute_nouns` defaults to `False`
  `k1`, or the alias `nouns_per_initial_center`, defaults to `200`
  `k2`, or the alias `nouns_per_image`, defaults to `50`
  `text_weight_temperature`, or the aliases `text_temp`/`retrieval_temperature`, defaults to `0.01`
  SEIC uses different noun selection from TAC: it first obtains initial centers through `K`-cluster K-Means on source training image features, selects the top-`k1` WordNet nouns per center to form a candidate set, then constructs a weighted text feature for each image from its top-`k2` nouns in that set.
- Stage 1 alignment parameters:
  `stage1_epochs`, or the alias `alignment_epochs`, defaults to `200`
  `stage1_batch_size`, or the alias `alignment_batch_size`, defaults to `1024`
  `stage1_learning_rate`, or the alias `alignment_lr`, defaults to `0.005`
  `stage1_weight_decay` defaults to `0.0`
  `instance_temperature` defaults to `0.07` and initializes a trainable logit scale
  `assignment_temperature`, or the alias `assign_temp`, defaults to `0.5`
  `center_temperature`, or the alias `center_temp`, defaults to `0.5`
  `balance_momentum` defaults to `0.9`
  `balance_history_floor` defaults to `0.01`
  `balance_max_weight` defaults to `5.0`
  Default loss weights follow the paper's appendix: `alpha=0.5`, `beta=1.0`, `gamma=1.0`, `delta=2.0`
  To prevent `1 / h_j` from becoming numerically unstable when the historical share of a few clusters approaches 0, the implementation floors, normalizes, and clips the dynamic history weights; if cluster collapse occurs, increasing `delta` or `balance_max_weight` is a first option.
  Currently confirmed benchmark parameters for `LAION400M`; `default` in the table means that the reported value has been checked against the current implementation and matches its fallback, so it is not explicitly added to the code profile:

  `ViT-B/32`

  | dataset | `text_temp` | `assign_temp` | `center_temp` |
  | --- | ---: | ---: | ---: |
  | CIFAR-10 | default | default | default |
  | CIFAR-20 | default | default | default |
  | CIFAR-100 | default | default | default |
  | STL-10 | default | default | default |
  | ImageNet-10 | default | default | default |
  | ImageNet-Dogs | default | default | default |
  | DTD | 0.008 | 0.2 | default |
  | UCF101 | 0.02 | default | default |
  | ImageNet | default | default | default |
  | Places365-Standard | default | default | default |
  | Aircraft | 0.02 | default | default |
  | Cars | 0.05 | default | default |
  | Flowers | default | default | default |
  | Food | default | default | default |
  | Pets | 0.008 | default | 0.4 |
  | ImageNet-Variants | default | default | default |

  `ViT-B/16`

  | dataset | `text_temp` | `assign_temp` | `center_temp` |
  | --- | ---: | ---: | ---: |
  | CIFAR-10 | 0.04 | default | default |
  | CIFAR-20 | default | default | default |
  | CIFAR-100 | default | default | default |
  | STL-10 | default | default | default |
  | ImageNet-10 | default | default | default |
  | ImageNet-Dogs | default | default | default |
  | DTD | 0.008 | 0.2 | default |
  | UCF101 | 0.03 | default | default |
  | ImageNet | default | default | default |
  | Places365-Standard | default | default | default |
  | Aircraft | 0.04 | default | 0.6 |
  | Cars | 0.05 | 0.4 | default |
  | Flowers | 0.02 | default | default |
  | Food | 0.02 | default | default |
  | Pets | default | default | default |
  | ImageNet-Variants | default | default | default |

  `ViT-L/14`

  | dataset | `text_temp` | `assign_temp` | `center_temp` |
  | --- | ---: | ---: | ---: |
  | CIFAR-10 | default | default | default |
  | CIFAR-20 | 0.02 | default | default |
  | CIFAR-100 | default | default | default |
  | STL-10 | 0.02 | default | default |
  | ImageNet-10 | default | default | default |
  | ImageNet-Dogs | default | default | default |
  | DTD | 0.008 | 0.2 | default |
  | UCF101 | 0.02 | default | default |
  | ImageNet | default | default | default |
  | Places365-Standard | default | default | default |
  | Aircraft | 0.03 | default | default |
  | Cars | 0.04 | default | default |
  | Flowers | default | default | default |
  | Food | 0.04 | default | default |
  | Pets | default | default | default |
  | ImageNet-Variants | default | default | default |
- Stage 2 self-enhancement parameters:
  `stage2` defaults to `False`, running only Stage 1 and returning SEIC† / Stage-1 image-head predictions by default
  Pass `method.stage2=true` on the command line to explicitly enable Stage 2
  `stage1_only=True` forces Stage 2 to be skipped, even when `stage2=True` is also provided
  `stage2_epochs`, or the alias `self_enhance_epochs`, defaults to `40`
  `stage2_batch_size`, or the alias `self_enhance_batch_size`, defaults to `128`
  `stage2_learning_rate`, or the alias `self_enhance_lr`, defaults to `5e-5`
  `stage2_weight_decay` defaults to `0.0`
  `lora_rank` defaults to `128`
  `lora_alpha` defaults to `lora_rank`
  `lora_dropout` defaults to `0.0`
  `confidence_momentum` defaults to `0.999`
  Strong augmentation includes `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, and `RandomGrayscale` by default, with the following parameters:
  `strong_crop_scale_min=0.5`, `strong_crop_scale_max=1.0`, `horizontal_flip_p=0.5`,
  `color_jitter_p=0.8`, `color_jitter_strength=0.4`, `grayscale_p=0.2`
- Recorded outputs:
  In `report.json`, `result_metadata.extra_predictions.stage1` stores Stage 1 predictions; for the full SEIC method, the main `predictions` are the results after Stage 2.
  `result_metadata.stage1.training_history` and `result_metadata.stage2.training_history` record the loss/confidence curves for the two stages, respectively, to support tuning.

## SAC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `wordnet_csv`/`wordnet_path`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the default evaluation split of the target dataset
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Default cluster count:
  If `n_clusters` is not explicitly provided, it defaults to the number of classes in the source training dataset; the default for domain shift from full ImageNet is `1000`
- Integration conventions in this benchmark:
  This integration adopts only SAC's multimodal cluster-head training approach, without the official SAC text construction pipeline of `BLIP-2 caption -> SBERT embedding`
  Text counterparts follow TAC throughout: shared `WordNetNouns.csv`, OpenCLIP text encoding, TAC semantic noun filtering, and retrieval weighted by image similarity
  Image features also reuse this benchmark's shared raw OpenCLIP image feature cache
- Text filtering and retrieval parameters:
  `image_batch_size` defaults to `256`
  `text_batch_size` defaults to `2048`
  `retrieval_batch_size` defaults to `8192`
  `selected_nouns_per_center` defaults to `5`
  `retrieval_temperature`, or its alias `tau`, defaults to `0.005`
  Currently confirmed benchmark parameters for `LAION400M + ViT-B/32`; `temp` corresponds to the training parameter `contrastive_temperature`/`temperature`, and `default` means that the reported value has been checked against the current implementation and matches its fallback:

  | dataset | `tau` | `temp` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | default | default |
  | STL-10 | default | default |
  | ImageNet-10 | default | default |
  | ImageNet-Dogs | default | default |
  | DTD | default | default |
  | UCF101 | default | default |
  | ImageNet | 0.01 | default |
  | Places365-Standard | default | default |
  | Aircraft | 0.06 | default |
  | Cars | 0.05 | default |
  | Flowers | 0.01 | 2.0 |
  | Food | 0.01 | default |
  | Pets | default | default |
  | ImageNet-Variants | 0.01 | default |

  Currently confirmed benchmark parameters for `LAION400M + ViT-B/16`; `default` likewise means that the reported value has been checked against the current implementation and matches its fallback:

  | dataset | `tau` | `temp` |
  | --- | ---: | ---: |
  | CIFAR-10 | 0.01 | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 0.05 | default |
  | STL-10 | default | default |
  | ImageNet-10 | default | default |
  | ImageNet-Dogs | default | default |
  | DTD | default | default |
  | UCF101 | default | default |
  | ImageNet | 0.01 | default |
  | Places365-Standard | default | default |
  | Aircraft | 0.06 | 2.0 |
  | Cars | 0.05 | default |
  | Flowers | 0.001 | 2.0 |
  | Food | 0.02 | default |
  | Pets | default | default |
  | ImageNet-Variants | 0.01 | default |

  Currently confirmed benchmark parameters for `LAION400M + ViT-L/14`; `default` likewise means that the reported value has been checked against the current implementation and matches its fallback:

  | dataset | `tau` | `temp` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | 0.05 | default |
  | CIFAR-100 | default | default |
  | STL-10 | default | default |
  | ImageNet-10 | default | default |
  | ImageNet-Dogs | default | default |
  | DTD | 0.05 | default |
  | UCF101 | default | default |
  | ImageNet | 0.01 | default |
  | Places365-Standard | default | default |
  | Aircraft | 0.06 | default |
  | Cars | 0.05 | default |
  | Flowers | default | default |
  | Food | 0.01 | default |
  | Pets | default | default |
  | ImageNet-Variants | 0.01 | default |

  `semantic_cluster_size` defaults to `300`
  `semantic_clusters` defaults to `None`; the default follows TAC, preferring fixed semantic cluster counts for datasets covered by official TAC and inferring them from the training set size and `semantic_cluster_size` for other datasets
  `candidate_noun_limit` defaults to `None`
  `force_recompute_nouns` defaults to `False`
- Training parameters:
  `epochs` depends on `n_clusters` by default: `100` when `n_clusters > 512`, otherwise `30`
  `train_batch_size`, or its alias `batch_size`, depends on `n_clusters` by default: `8192` when `n_clusters > 512`, otherwise `512`
  This profile for high class counts is mainly intended for ImageNet-1K; SAC's batch-wise balance loss can drive predictions toward overly diffuse distributions when `train_batch_size < n_clusters`
  If the source training split contains fewer samples than `train_batch_size`, the effective training batch size is automatically reduced to the training split size to prevent the cluster head from receiving zero training steps
  `neighbors_topk`, or the official alias `topk`, depends on `n_clusters` by default: `50` when `n_clusters > 512`, otherwise `20`
  If the source training split contains no more samples than `neighbors_topk`, the effective top-k is automatically reduced to `train_size - 1`
  `learning_rate`, or the alias `lr`, defaults to `1e-3`
  `contrastive_temperature`, or the alias `temperature`, defaults to `1.1`
  `weight_scale`, or its alias `a` from the paper's notation, defaults to `1.0`
  `alpha` defaults to `0.9` and mixes SAC's intra/cross-neighborhood uncertainty weights
  `alignment_weight`, or the aliases `consist_coeff`/`lambda_a`, defaults to `0.6`
  `balance_weight`, or the alias `lambda_b`, defaults to `4.0`
  `entropy_coeff` is a compatibility alias for the official code; if explicitly provided, `balance_weight = -entropy_coeff` is used
  `consistency_beta`, or its alias `beta`, defaults to `0.5` and mixes instance-level and cluster-column consistency
  `weighted_contrastive` defaults to `True`
  `save_checkpoint` defaults to `False`; when enabled, it saves the final SAC cluster-head checkpoint
- Differences from the official SAC implementation:
  Official SAC generates a caption for each image with BLIP-2 and obtains text features with SBERT; this version replaces that process with TAC's WordNet/OpenCLIP text counterparts for benchmark consistency
  The official training script reads test labels during training for early stopping; this benchmark version does not use evaluation labels for model selection and instead evaluates once after training for a fixed number of `epochs`, avoiding test set leakage

## MAGIC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `wordnet_csv`/`wordnet_path`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`
- Integration conventions in this benchmark:
  MAGIC's fine/coarse text granularity construction reuses TAC-style WordNet noun selection and OpenCLIP text encoding, without the original MAGIC pipeline of Qwen captions and spaCy concepts
  `retrieval_temperature`, or the alias `tau`, defaults to `0.005`
  `train_batch_size`, or its alias `batch_size`, depends on `n_clusters` by default: `8192` when `n_clusters >= 100`, otherwise `512`
  Currently confirmed benchmark parameters for `LAION400M`; `bs` corresponds to `train_batch_size`/`batch_size`, and `default` means that the reported value has been checked against the current implementation and matches its fallback:

  `ViT-B/32`

  | dataset | `tau` | `bs` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 0.03 | 2048 |
  | STL-10 | 0.01 | default |
  | ImageNet-10 | 0.01 | default |
  | ImageNet-Dogs | 0.01 | default |
  | DTD | 0.03 | default |
  | UCF101 | 0.01 | 1024 |
  | Places365-Standard | 0.01 | default |
  | ImageNet | 0.02 | default |
  | Aircraft | 0.05 | 512 |
  | Cars | 0.02 | 1024 |
  | Flowers | 0.001 | 512 |
  | Food | 0.03 | 2048 |
  | Pets | 0.008 | default |
  | ImageNet-Variants | 0.02 | default |

  `ViT-B/16`

  | dataset | `tau` | `bs` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 0.03 | 2048 |
  | STL-10 | 0.01 | default |
  | ImageNet-10 | 0.01 | default |
  | ImageNet-Dogs | 0.01 | 1024 |
  | DTD | 0.02 | default |
  | UCF101 | 0.01 | 1024 |
  | Places365-Standard | 0.01 | default |
  | ImageNet | 0.01 | default |
  | Aircraft | 0.03 | 512 |
  | Cars | 0.01 | 1024 |
  | Flowers | 0.008 | 512 |
  | Food | 0.01 | 4096 |
  | Pets | 0.02 | default |
  | ImageNet-Variants | 0.01 | default |

  `ViT-L/14`

  | dataset | `tau` | `bs` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | 0.04 | default |
  | CIFAR-100 | default | 2048 |
  | STL-10 | 0.01 | default |
  | ImageNet-10 | 0.01 | default |
  | ImageNet-Dogs | 0.01 | default |
  | DTD | 0.02 | default |
  | UCF101 | 0.02 | 1024 |
  | Places365-Standard | 0.01 | default |
  | ImageNet | 0.01 | default |
  | Aircraft | default | 512 |
  | Cars | 0.02 | 1024 |
  | Flowers | default | 512 |
  | Food | 0.01 | 4096 |
  | Pets | 0.03 | default |
  | ImageNet-Variants | 0.01 | default |
- Other training parameters:
  `epochs` defaults to `100`
  `neighbors_topk`, or the alias `topk`, defaults to `50`
  `learning_rate`, or the alias `lr`, defaults to `1e-4`
  `distill_temperature`, or the alias `temperature`, defaults to `1.1`
  `alignment_weight`, or the aliases `consist_coeff`/`lambda`, defaults to `0.6`
  `balance_weight`, or the alias `mu`, defaults to `5.0`
  `bottleneck_ratio`, or the official aliases `r`/`l`, defaults to `0.25`
  `dropout_rate` defaults to `0.1`
  `fusion_heads` defaults to `8`
  `prediction_head`, or its alias `head`, defaults to `both`; supported choices are `image`, `text`, and `both`. By default, both image (V) and text (T) results are evaluated and saved, with `metrics` containing the main image metrics and `extra_metrics.text` containing the text metrics

## NTK-SC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `wordnet_csv`/`wordnet_path`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the target dataset's default evaluation split
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Default cluster-count rules:
  If not explicitly specified, `n_clusters` defaults to the number of classes in the evaluation dataset
- Noun filtering parameters:
  `filter_cluster_num`, or its alias `proxy_cluster_num`
  If not explicitly specified, the current default falls back to `3 x n_clusters`
  `selected_nouns_per_cluster`, or its alias `topK`, defaults to `5`
  `image_batch_size` defaults to `1024`
  `text_batch_size` defaults to `2048`
  `force_recompute_nouns` defaults to `False`
  `force_recompute_selected_nouns` defaults to `False`
- RED and spectral clustering parameters:
  `graph_k`, or its alias `k`, defaults to `30`
  `pntk_temperature`, or its alias `temp`, defaults to `0.04`
  Confirmed `LAION400M` benchmark parameters: `k` is always the current default, `30`, so it is not additionally written into the profile; `num` corresponds to `filter_cluster_num`/`proxy_cluster_num`. In the tables, `default` means that the reported value has been verified to match the current implementation's fallback:

  `ViT-B/32`

  | dataset | `temp` | `num` |
  | --- | ---: | ---: |
  | CIFAR-10 | 0.05 | 167 |
  | CIFAR-20 | 0.01 | 167 |
  | CIFAR-100 | 0.05 | 167 |
  | STL-10 | 0.06 | default |
  | ImageNet-10 | 0.02 | 43 |
  | ImageNet-Dogs | 0.01 | 65 |
  | DTD | 0.05 | default |
  | UCF101 | default | default |
  | ImageNet | default | 4271 |
  | Places365-Standard | default | 6012 |
  | Aircraft | 0.07 | default |
  | Cars | 0.09 | 300 |
  | Flowers | 0.03 | 300 |
  | Food | 0.06 | 300 |
  | Pets | default | 100 |
  | ImageNet-Variants | default | 4271 |

  `ViT-B/16`

  | dataset | `temp` | `num` |
  | --- | ---: | ---: |
  | CIFAR-10 | 0.07 | 167 |
  | CIFAR-20 | 0.03 | 167 |
  | CIFAR-100 | default | 167 |
  | STL-10 | 0.06 | default |
  | ImageNet-10 | 0.02 | 43 |
  | ImageNet-Dogs | default | 65 |
  | DTD | 0.06 | default |
  | UCF101 | default | default |
  | ImageNet | default | 4271 |
  | Places365-Standard | default | 6012 |
  | Aircraft | 0.07 | default |
  | Cars | 0.09 | 300 |
  | Flowers | 0.05 | 300 |
  | Food | 0.06 | 300 |
  | Pets | default | 100 |
  | ImageNet-Variants | default | 4271 |

  `ViT-L/14`

  | dataset | `temp` | `num` |
  | --- | ---: | ---: |
  | CIFAR-10 | 0.07 | 167 |
  | CIFAR-20 | 0.05 | 167 |
  | CIFAR-100 | 0.05 | 167 |
  | STL-10 | 0.01 | default |
  | ImageNet-10 | 0.02 | 43 |
  | ImageNet-Dogs | 0.01 | 65 |
  | DTD | default | default |
  | UCF101 | 0.05 | default |
  | ImageNet | default | 4271 |
  | Places365-Standard | default | 6012 |
  | Aircraft | 0.07 | default |
  | Cars | 0.09 | 300 |
  | Flowers | 0.05 | 300 |
  | Food | 0.06 | 300 |
  | Pets | 0.05 | 100 |
  | ImageNet-Variants | default | 4271 |

  `red_backend`, or its alias `red_mode`, defaults to `gpu_auto`
  `gpu_auto` / `auto_gpu` is the default GPU-first routing policy: on CUDA, when the evaluation sample count is at most `red_dense_max_samples`, it uses `dense_gpu`; above that threshold, it switches directly to `dense_cached_gpu_red`. On non-CUDA devices, it falls back to `auto`
  `dense` uses the original dense pNTK + CPU NumPy dense RED path, allowing the original experimental backend to be reproduced explicitly
  `dense_gpu` uses the original dense pNTK and runs dense matrix multiplication for RED diffusion on the CUDA GPU; it is suitable for small and medium evaluation splits such as CIFAR10/STL10
  `dense_cached` builds a transition graph with dense pNTK on the GPU for each prompt, writes it to the disk cache, and then runs out-of-core RED
  `dense_cached_gpu_red` builds a transition graph with dense pNTK on the GPU for each prompt, writes it to the disk cache, and performs block-wise matrix multiplication on the CUDA GPU during out-of-core RED
  Under `auto`, the original dense RED is used when the evaluation sample count is at most `red_dense_max_samples`; full-sparse RED over the entire dataset is used above that threshold but below `red_out_of_core_min_samples`; out-of-core dense RED is used once `red_out_of_core_min_samples` is reached
  `red_oom_fallback` defaults to `True`
  `red_oom_fallback_backend` defaults to `dense_cached_gpu_red`; it automatically switches to this backend when the `dense` or `dense_gpu` path raises an OOM error. Options are `dense_cached_gpu_red`, `dense_cached`, `out_of_core`, and `full_sparse`
  `red_dense_max_samples` defaults to `50000`
  `red_out_of_core_min_samples` defaults to `20000`
  `pntk_block_size` defaults to `512`
  `pntk_column_block_size` defaults to `1024`
  `red_block_size` defaults to `128`
  `max_sparse_nnz` defaults to `150000000`
  `red_prune_k` defaults to `None`; when explicitly specified, it retains the top-k entries in each row after every full-sparse RED step. This is an approximation parameter for acceleration and memory control
  `spectral_n_init`, or its alias `n_init`, defaults to `10`
  `force_recompute_red_graphs` defaults to `False`
  `keep_red_work_dir` defaults to `False`
  `red_work_dir` defaults to `None`
  `mu` defaults to `0.1`
  `lam` defaults to `10.0`
  `t_outer` defaults to `5`
  `t_diffuse` defaults to `8`
  `ta_cd` defaults to `4`
  `use_pntk_fast` defaults to `True`
  `seed` defaults to `0`
- Cache:
  Source-side noun selection retains the existing method-specific `selected_nouns.npz` cache; ImageNet domain-shift evaluation with `source_dataset_name=imagenet` reuses the same noun-selection cache from ImageNet training
  The per-prompt dense-pNTK transition graphs for `dense_cached` / `dense_cached_gpu_red` are saved to `red_dense_graphs/*__dense_pntk_transition.npz`
  The per-prompt pNTK transition graphs for sparse/out-of-core RED are saved to `red_graphs/*__transition.npz`; the key includes the evaluation dataset/split, OpenCLIP model, WordNet fingerprint, source dataset, noun filtering settings, selected-noun hash, `graph_k`, and `pntk_temperature`

## GradNorm

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `wordnet_csv`/`wordnet_path`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the target dataset's default evaluation split
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Default cluster-count rules:
  If not explicitly specified, `n_clusters` defaults to the number of classes in the evaluation dataset
- Only the no-training version is currently integrated; cluster-head training parameters are not included
- Noun filtering parameters:
  `filter_cluster_num`, or its alias `proxy_cluster_num`
  If not explicitly specified, the current implementation falls back to dataset-specific defaults:
  `cifar10=250`, `cifar20=200`, `stl10=30`, `imagenet10=50`, `imagenetdogs=50`, `dtd=180`, `ucf101=303`, `imagenet=2000`
  `selected_nouns_per_center`, or its alias `topK`, defaults to `5`
  `filter_temperature`, or its alias `temp`, defaults to `0.04`
  `filter_p`, or its alias `p`, defaults to `2.0`
  `image_batch_size` defaults to `256`
  `text_batch_size` defaults to `2048`
  `candidate_noun_limit` defaults to `None`
  `force_recompute_nouns` defaults to `False`
- Retrieval and final clustering parameters:
  `retrieval_batch_size` defaults to `8192`
  `retrieval_temperature` defaults to `0.005`
  `kmeans_niter` defaults to `300`
  `kmeans_nredo` defaults to `20`
  Confirmed `LAION400M + ViT-B/32` benchmark parameters: `cluster_num` corresponds to `filter_cluster_num`/`proxy_cluster_num`. In the table, `default` means that the reported value has been verified to match the current implementation's fallback:

  | dataset | `cluster_num` | `temp` | `retrieval_temp` |
  | --- | ---: | ---: | ---: |
  | CIFAR-10 | 200 | 0.01 | 0.01 |
  | CIFAR-20 | default | 0.005 | 0.02 |
  | CIFAR-100 | default | 0.005 | 0.02 |
  | STL-10 | default | 0.02 | 0.01 |
  | ImageNet-10 | 45 | 0.03 | 0.01 |
  | ImageNet-Dogs | 60 | 0.01 | 0.02 |
  | DTD | 141 | 0.01 | 0.02 |
  | UCF101 | default | 0.01 | 0.02 |
  | ImageNet | default | 0.02 | 0.01 |
  | Places365-Standard | 1500 | 0.01 | 0.01 |
  | Aircraft | 250 | 0.01 | 0.02 |
  | Cars | 250 | default | 0.02 |
  | Flowers | 200 | 0.01 | 0.02 |
  | Food | 250 | 0.01 | 0.02 |
  | Pets | 100 | 0.01 | 0.03 |
  | ImageNet-Variants | default | 0.02 | 0.01 |

  Confirmed `LAION400M + ViT-B/16` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback:

  | dataset | `cluster_num` | `temp` | `retrieval_temp` |
  | --- | ---: | ---: | ---: |
  | CIFAR-10 | 200 | 0.05 | 0.01 |
  | CIFAR-20 | 300 | 0.005 | 0.001 |
  | CIFAR-100 | default | 0.003 | 0.02 |
  | STL-10 | default | 0.02 | 0.03 |
  | ImageNet-10 | 45 | 0.02 | 0.05 |
  | ImageNet-Dogs | 60 | default | 0.02 |
  | DTD | default | 0.03 | 0.02 |
  | UCF101 | 350 | default | 0.02 |
  | ImageNet | default | 0.02 | 0.01 |
  | Places365-Standard | 1500 | 0.01 | 0.02 |
  | Aircraft | 250 | 0.02 | 0.04 |
  | Cars | 250 | 0.03 | 0.04 |
  | Flowers | 200 | 0.03 | 0.04 |
  | Food | 250 | 0.01 | 0.02 |
  | Pets | 100 | 0.03 | 0.04 |
  | ImageNet-Variants | default | 0.02 | 0.01 |

  Confirmed `LAION400M + ViT-L/14` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback:

  | dataset | `cluster_num` | `temp` | `retrieval_temp` |
  | --- | ---: | ---: | ---: |
  | CIFAR-10 | 200 | 0.01 | 0.01 |
  | CIFAR-20 | 300 | 0.005 | 0.01 |
  | CIFAR-100 | default | 0.01 | 0.02 |
  | STL-10 | default | 0.005 | 0.04 |
  | ImageNet-10 | 45 | 0.01 | 0.01 |
  | ImageNet-Dogs | 60 | 0.05 | 0.03 |
  | DTD | default | 0.01 | 0.03 |
  | UCF101 | 350 | 0.06 | 0.04 |
  | ImageNet | default | 0.01 | 0.01 |
  | Places365-Standard | 1500 | 0.01 | 0.03 |
  | Aircraft | 250 | 0.02 | 0.04 |
  | Cars | 250 | 0.01 | 0.04 |
  | Flowers | 200 | default | 0.05 |
  | Food | 250 | 0.02 | 0.04 |
  | Pets | 100 | 0.03 | 0.06 |
  | ImageNet-Variants | default | 0.01 | 0.01 |

## IDC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`
- Optional domain-shift evaluation parameters:
  `target_dataset_name`, or its alias `target_dataset`
  `target_test_split` is inferred from the target dataset's default evaluation split unless specified
  `eval_n_clusters` defaults to `n_clusters`; it does not implicitly use the target dataset's class count
- Value mining parameters:
  `value_budget`, or its alias `M`, defaults to `500`
  `candidate_cluster_topk`, or its alias `T`, defaults to `5`
  `representativeness_k`, or its alias `K`, defaults to `20`
  `value_chunk_size` defaults to `500`
- Cluster-head training parameters:
  `warmup_epochs` defaults to `20`
  `idc_warmup_epochs` defaults to `50`
  `epochs` defaults to `100`
  `learning_rate`, or its alias `lr`, defaults to `1e-3`
  `batch_size_inquiry` defaults to `100`
  `confidence_threshold`, or its alias `tau`, defaults to `0.99`
  `pseudo_gamma` defaults to `0.2`
  `pseudo_chunk_size` defaults to `2000`
  `save_checkpoint` defaults to `False`; when enabled, it saves the final IDC cluster-head checkpoint
  Confirmed `LAION400M + ViT-B/32` benchmark parameters: `default` means that the reported value has been verified to match the current implementation's fallback:

  | dataset | `value_budget` | `lr` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 2000 | default |
  | STL-10 | default | default |
  | ImageNet-10 | 3000 | default |
  | ImageNet-Dogs | default | default |
  | DTD | default | default |
  | UCF101 | 1000 | default |
  | ImageNet | 2000 | default |
  | Places365-Standard | default | default |
  | Aircraft | 2000 | default |
  | Cars | 2000 | default |
  | Flowers | 1020 | default |
  | Food | default | default |
  | Pets | 1000 | default |
  | ImageNet-Variants | 2000 | default |

  Confirmed `LAION400M + ViT-B/16` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback:

  | dataset | `value_budget` | `lr` |
  | --- | ---: | ---: |
  | CIFAR-10 | 100 | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 5000 | default |
  | STL-10 | default | default |
  | ImageNet-10 | 3000 | default |
  | ImageNet-Dogs | default | default |
  | DTD | 1000 | default |
  | UCF101 | 2000 | default |
  | ImageNet | 2000 | default |
  | Places365-Standard | default | default |
  | Aircraft | 2000 | default |
  | Cars | 2000 | default |
  | Flowers | 1020 | default |
  | Food | 1000 | default |
  | Pets | 2000 | default |
  | ImageNet-Variants | 2000 | default |

  Confirmed `LAION400M + ViT-L/14` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback:

  | dataset | `value_budget` | `lr` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | 5e-3 |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 3000 | default |
  | STL-10 | default | default |
  | ImageNet-10 | 3000 | default |
  | ImageNet-Dogs | default | default |
  | DTD | 1000 | default |
  | UCF101 | 2000 | default |
  | ImageNet | 2000 | default |
  | Places365-Standard | default | default |
  | Aircraft | 2000 | default |
  | Cars | 2000 | default |
  | Flowers | 1020 | default |
  | Food | 1000 | default |
  | Pets | 2000 | default |
  | ImageNet-Variants | 2000 | default |
- Notes:
  The supplied `idc.py` also defines `batch_size_confident=500`, but the script does not actually use this parameter in its computation graph, so the current benchmark version does not expose it as an effective runtime parameter
- K-Means initialization parameters:
  `image_batch_size` defaults to `256`
  `kmeans_niter` defaults to `300`
  `kmeans_nredo` defaults to `10`

## SSC-OMP

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `test_split`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the target dataset's default evaluation split
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Notes:
  `SSC-OMP` is integrated into this benchmark as a training-free, image-only method that clusters the evaluation split's OpenCLIP image features directly
- OMP and spectral clustering parameters:
  `n_nonzero` defaults to `10`
  `thr` defaults to `1e-6`
  `affinity` defaults to `symmetrize`
  `random_state` defaults to `None`
  `n_init` defaults to `10`
  `image_batch_size` defaults to `256`
- Confirmed `LAION400M + ViT-B/32` benchmark parameters: the reported `nonzero` corresponds to the implementation parameter `n_nonzero`. In the table, `default` means that this combination continues to use the general fallback of `10` and is not written into an explicit code profile:

  | Dataset | n_nonzero |
  | --- | ---: |
  | CIFAR-10 | 3 |
  | CIFAR-20 | 3 |
  | CIFAR-100 | 4 |
  | STL-10 | 4 |
  | ImageNet-10 | default |
  | ImageNet-Dogs | 4 |
  | DTD | 7 |
  | UCF101 | 4 |
  | ImageNet | default |
  | Places365-Standard | default |
  | Aircraft | 6 |
  | Cars | 15 |
  | Flowers | 7 |
  | Food | default |
  | Pets | 3 |
  | ImageNet-Variants | default |

- Confirmed `LAION400M + ViT-B/16` benchmark parameters: `default` has the same meaning as above:

  | Dataset | n_nonzero |
  | --- | ---: |
  | CIFAR-10 | 3 |
  | CIFAR-20 | 3 |
  | CIFAR-100 | 4 |
  | STL-10 | 3 |
  | ImageNet-10 | default |
  | ImageNet-Dogs | 4 |
  | DTD | default |
  | UCF101 | 3 |
  | ImageNet | default |
  | Places365-Standard | default |
  | Aircraft | default |
  | Cars | default |
  | Flowers | 6 |
  | Food | default |
  | Pets | 3 |
  | ImageNet-Variants | default |

- Confirmed `LAION400M + ViT-L/14` benchmark parameters: `default` has the same meaning as above:

  | Dataset | n_nonzero |
  | --- | ---: |
  | CIFAR-10 | 3 |
  | CIFAR-20 | 3 |
  | CIFAR-100 | 6 |
  | STL-10 | 3 |
  | ImageNet-10 | 3 |
  | ImageNet-Dogs | 4 |
  | DTD | 4 |
  | UCF101 | default |
  | ImageNet | default |
  | Places365-Standard | default |
  | Aircraft | default |
  | Cars | default |
  | Flowers | default |
  | Food | default |
  | Pets | 7 |
  | ImageNet-Variants | default |

## EnSC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `test_split`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the target dataset's default evaluation split
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Notes:
  `EnSC` is integrated into this benchmark as a training-free, image-only method that clusters the evaluation split's OpenCLIP image features directly
- Elastic net and spectral clustering parameters:
  `gamma` defaults to `50.0`
  `gamma_nz` defaults to `True`
  `tau` defaults to `1.0`
  `algorithm` defaults to `lasso_lars`
  When `algorithm` is `lasso_lars` or `lasso_cd` and `tau < 1`, the implementation falls back to `tau = 1`, following the official repository's logic
  `active_support` defaults to `True`
  `active_support_params` defaults to `None`; when `active_support=True` and these parameters are not explicitly specified, the official function defaults are used
  `support_init='knn'`, `support_size=100`, `maxiter=40`
  `n_nonzero` defaults to `50`
  `affinity` defaults to `symmetrize`
  `random_state` defaults to `None`
  `n_init` defaults to `20`
  `image_batch_size` defaults to `256`
- Dataset-specific defaults (LAION400M / LAION2B):
  `imagenet10` applies the default overrides `gamma=25.0`, `n_nonzero=20`, `n_init=100`,
  `active_support_params.support_size=100`
  `imagenet_dogs` applies the default overrides `gamma=50.0`, `n_nonzero=10`, `n_init=100`,
  `active_support_params.support_size=75`
  `ucf101` applies the default overrides `gamma=25.0`, `n_nonzero=20`, `n_init=100`,
  `active_support_params.support_size=100`
- LAION2B and SigLIP:
  All three `LAION2B` backbones use the dataset defaults above, but do not use the LAION400M backbone profiles.
  `SigLIP + ViT-B/16` skips the dataset and LAION400M profiles and uses the current generic fallback:
  `gamma=50.0`, `n_nonzero=50`, `n_init=20`; active support continues to use the general defaults
  `support_init=knn`, `support_size=100`, `maxiter=40`. Explicit parameters still take precedence.
- Confirmed `LAION400M + ViT-B/32` benchmark parameters: the reported `nonzero` corresponds to the implementation parameter `n_nonzero`. In the table, `default` means that this combination continues to use the dataset fallback or general fallback above and is not written into an explicit code profile:

  | Dataset | gamma | n_nonzero |
  | --- | ---: | ---: |
  | CIFAR-10 | 100 | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 100 | default |
  | STL-10 | 100 | default |
  | ImageNet-10 | default | default |
  | ImageNet-Dogs | 25 | default |
  | DTD | 200 | default |
  | UCF101 | default | default |
  | ImageNet | default | default |
  | Places365-Standard | 100 | default |
  | Aircraft | default | default |
  | Cars | default | default |
  | Flowers | default | 10 |
  | Food | default | default |
  | Pets | default | default |
  | ImageNet-Variants | default | default |

- Confirmed `LAION400M + ViT-B/16` benchmark parameters: `default` has the same meaning as above; the reported `gamma=50` for DTD has been verified to match the current generic fallback:

  | Dataset | gamma | n_nonzero |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 100 | default |
  | STL-10 | 100 | default |
  | ImageNet-10 | 100 | default |
  | ImageNet-Dogs | 25 | default |
  | DTD | default | default |
  | UCF101 | default | default |
  | ImageNet | default | default |
  | Places365-Standard | default | default |
  | Aircraft | default | default |
  | Cars | default | default |
  | Flowers | default | 10 |
  | Food | default | default |
  | Pets | default | default |
  | ImageNet-Variants | default | default |

- Confirmed `LAION400M + ViT-L/14` benchmark parameters: `default` has the same meaning as above:

  | Dataset | gamma | n_nonzero |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 100 | default |
  | STL-10 | default | default |
  | ImageNet-10 | 50 | default |
  | ImageNet-Dogs | 25 | default |
  | DTD | 200 | default |
  | UCF101 | default | default |
  | ImageNet | default | default |
  | Places365-Standard | default | default |
  | Aircraft | default | default |
  | Cars | default | default |
  | Flowers | default | 10 |
  | Food | default | default |
  | Pets | default | default |
  | ImageNet-Variants | default | default |

## SIC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `wordnet_csv`/`wordnet_path`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the target dataset's default evaluation split
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Default cluster-count rules:
  If not explicitly specified, `n_clusters` defaults to the number of classes in the source training dataset; the default for full ImageNet domain shift is `1000`
- Semantic-space construction parameters:
  `gamma_u` defaults to `0.05`; the ImageNet profile defaults to `0.1`
  `gamma_r` defaults to `500`; the ImageNet profile defaults to `300`, and the Food and Pets profiles default to `1000`
  `xi_c` defaults to `0.9`; the ImageNet profile defaults to `0.5`, and the Flowers, Food, and Pets profiles default to `0.6`
  `xi_a` defaults to `30`; the ImageNet, Food, and Pets profiles default to `20`, and the Flowers profile defaults to `10`
  `text_batch_size` defaults to `2048`
  `force_recompute_nouns` defaults to `False`
  WordNet noun prompts use the benchmark's shared 7-prompt ensemble: each prompt embedding is first L2-normalized, then the embeddings are averaged and L2-normalized again.
- Training and neighbor parameters:
  `num_heads` defaults to `1`
  `topk` defaults to `20`; the ImageNet profile also defaults to `20`
  `epochs` defaults to `100`; the Aircraft, Cars, Flowers, Food, and Pets profiles default to `200`
  `train_batch_size`, or its alias `batch_size`, defaults to `128`; the ImageNet profile defaults to `4096`
  `image_batch_size` defaults to `128`
  `normalize_features` defaults to `False`; when set to `True`, SIC uses L2-normalized image features for K-Means, noun selection, KNN, training, and evaluation, with a separate cache key
  `learning_rate`, or its alias `lr`, defaults to `1e-4`; the ImageNet profile defaults to `3e-4`
  `weight_decay` defaults vary by source dataset: `imagenet_dogs` defaults to `0.009`, and all others default to `1e-4`
  `entropy_weight` defaults to `5.0`; the ImageNet profile defaults to `1.0`
  `ce_weight` defaults to `1.0`; the ImageNet, Aircraft, Flowers, Food, and Pets profiles default to `0.1`, and the Cars profile defaults to `0.05`
  `center_init` defaults to `False`; the ImageNet profile defaults to `True`
  `force_recompute_neighbors` defaults to `False`
  `save_checkpoint` defaults to `False`; when enabled, it saves the final SIC cluster-head checkpoint
- LAION2B and SigLIP:
  All three `LAION2B` backbones use the dataset profiles above, but do not use the LAION400M backbone
  profiles. `SigLIP + ViT-B/16` skips both profile layers and uses the current generic fallback:
  `gamma_u=0.05`, `gamma_r=500`, `xi_c=0.9`, `xi_a=30`, `topk=20`,
  `epochs=100`, `train_batch_size=128`, `learning_rate=1e-4`, `entropy_weight=5.0`,
  `ce_weight=1.0`, `center_init=False`, `normalize_features=False`.
  The profile-independent special case `imagenet_dogs` `weight_decay=0.009` remains in effect for both model families;
  Explicit parameters still take precedence.
- Confirmed `LAION400M + ViT-B/32` benchmark parameters: `default` means that the reported value has been verified to match the current implementation's fallback, dataset profile, or the ImageNet profile used for ImageNet-Variants:

  | dataset | `xi_c` | `lr` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | default | default |
  | STL-10 | default | default |
  | ImageNet-10 | 0.8 | default |
  | ImageNet-Dogs | 0.8 | default |
  | DTD | default | default |
  | UCF101 | default | default |
  | ImageNet | default | default |
  | Places365-Standard | 0.8 | default |
  | Aircraft | default | default |
  | Cars | default | default |
  | Flowers | default | default |
  | Food | default | default |
  | Pets | default | default |
  | ImageNet-Variants | default | default |

  Confirmed `LAION400M + ViT-B/16` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback, dataset profile, or the ImageNet profile used for ImageNet-Variants:

  | dataset | `xi_c` | `lr` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | 1.0 | default |
  | STL-10 | default | 3e-4 |
  | ImageNet-10 | 0.8 | 3e-4 |
  | ImageNet-Dogs | 0.8 | 3e-4 |
  | DTD | 0.8 | default |
  | UCF101 | 0.8 | 1e-3 |
  | ImageNet | default | default |
  | Places365-Standard | 0.8 | default |
  | Aircraft | default | default |
  | Cars | default | default |
  | Flowers | 0.7 | default |
  | Food | default | default |
  | Pets | default | 3e-4 |
  | ImageNet-Variants | default | default |

  Confirmed `LAION400M + ViT-L/14` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback, dataset profile, or the ImageNet profile used for ImageNet-Variants:

  | dataset | `xi_c` | `lr` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | default | default |
  | STL-10 | default | 3e-4 |
  | ImageNet-10 | default | 3e-4 |
  | ImageNet-Dogs | default | 3e-4 |
  | DTD | 0.7 | default |
  | UCF101 | default | default |
  | ImageNet | default | default |
  | Places365-Standard | 0.8 | default |
  | Aircraft | default | default |
  | Cars | default | default |
  | Flowers | 0.7 | default |
  | Food | default | default |
  | Pets | default | default |
  | ImageNet-Variants | default | default |
- K-Means initialization parameters:
  `kmeans_n_init`, or its alias `kmeans_nredo`, defaults to `20`
  `kmeans_max_iter`, or its alias `kmeans_niter`, defaults to `300`
  `random_state`, or its alias `seed`, defaults to `None`
- Notes:
  `SIC` retains the official repository's main `KNN + semantic space + cluster head` pipeline in this benchmark,
  but uses features extracted with the benchmark's `OpenCLIP` setup throughout;
  to follow this benchmark's `train/test` protocol, it does not use the original repository's early stopping based on evaluation labels.
  Although the official repository provides its own `noun.csv`, this benchmark consistently uses the shared `WordNetNouns.csv` dataset;
  it is read as a regular data resource from `data/WordNetNouns.csv` by default, or from a path explicitly specified with `wordnet_csv`/`wordnet_path`.

## CPP

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`, `n_clusters`,
  `head_mode`/`single_head`, `select_best_epoch`, `eval_every_epochs`, `best_epoch_metric`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the target dataset's default evaluation split
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Cluster-count parameters:
  `eval_n_clusters` defaults to `n_clusters` if provided, otherwise to the number of classes in the evaluation dataset
  CPP's MLC training objective does not explicitly depend on the number of classes; the class count is mainly used for the final spectral clustering of the membership matrix
- Features and caching:
  `image_batch_size` defaults to `256`
  CPP uses the shared cache of raw OpenCLIP image features
  Derived outputs such as training history are stored in a method-specific directory keyed by `cpp + openclip model key + dataset/k/head_mode`; checkpoints are saved only when `save_checkpoint=True`
- Network architecture parameters:
  `hidden_dim` defaults to the official value of `4096`; the `ImageNet` profile defaults to `2048`
  `z_dim` defaults to the official value of `128`; the `ImageNet` profile defaults to `1024`
  `head_mode` defaults to `two_head`, retaining the two-head architecture from the paper and the official `main.py`:
  The `subspace` head outputs `Z`, the `cluster` head outputs logits/code, and membership `Pi` is constructed from the logits/code.
  Set `head_mode="single_head"` or `single_head=true` to use the approach from the official `main_efficient.py`:
  It uses `Z` directly to construct membership `Pi`, and the cluster head does not participate in training or evaluation.
- Training parameters:
  `epochs`, or its alias `epo`, defaults to the official value of `30`
  `batch_size`, or its alias `bs`, defaults to the official value of `2000`
  If the source training split contains fewer samples than `batch_size`, the effective training batch size automatically shrinks to the training split size, preventing `drop_last=True` from producing an empty training loop
  `learning_rate`, or its alias `lr`, defaults to the official value of `3e-3`
  `cluster_learning_rate`, or its alias `lr_c`, defaults to the official value of `2e-3`
  `momentum`, or its alias `momo`, defaults to `0.9`
  `weight_decay_main`, or its alias `wd1`, defaults to `1e-4`
  `weight_decay_cluster`, or its alias `wd2`, defaults to `5e-3`
  `eps` defaults to `0.1`
  `gamma` defaults to `1.0`
  `pi_regularization`, or its alias `pigam`, defaults to `0.05`
  `pieta` defaults to `0.175`
  `piiter` defaults to `5`
  `warmup_steps`, or its alias `warmup`, defaults to `0`
  `save_checkpoint` defaults to `False`; when enabled, `save_every` or `checkpoint_every` defaults to `50`
  `seed` defaults to `42`
- Dataset profiles (LAION400M / LAION2B):
  For datasets explicitly covered by Table 8 of the paper, the paper's experimental profiles take priority by default; `warmup_steps`
  counts optimizer steps in the current implementation, so the paper's initialization epochs are converted approximately.

  | dataset | profile rationale | hidden_dim | z_dim | epochs | batch_size | lr | lr_c | pieta | warmup |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `cifar10` | Paper's CIFAR-10 profile; 1 warmup epoch is approximately 50 steps | 4096 | 128 | 5 | 1024 | 1e-4 | 1e-4 | 0.175 | 50 |
  | `cifar20` | Paper's CIFAR-20 profile; class count and sparsity lie between CIFAR-10/100 | 4096 | 128 | 15 | 1024 | 1e-4 | 1e-4 | 0.13 | 50 |
  | `cifar100` | Paper's CIFAR-100 profile; longer training, with 1 warmup epoch approximately 33 steps | 4096 | 128 | 50 | 1500 | 1e-4 | 1e-4 | 0.1 | 33 |
  | `imagenet` | Paper/README ImageNet profile; lower hidden dimension and higher z dimension for large-scale data | 2048 | 1024 | 20 | 1024 | 1e-4 | 1e-4 | 0.12 | 2000 |
  | `stl10` | Small natural-image dataset with 10 classes, using CIFAR-10-style representations and batch size | 4096 | 128 | 15 | 1024 | 1e-4 | 1e-4 | 0.175 | 5 |
  | `imagenet10` | ImageNet subset with 10 classes, using a CIFAR-10-style profile while retaining a longer warmup | 4096 | 128 | 15 | 1024 | 1e-4 | 1e-4 | 0.175 | 50 |
  | `imagenet_dogs` | Fine-grained ImageNet subset with 15 classes, using CIFAR-20-style sparsity | 4096 | 128 | 15 | 1024 | 1e-4 | 1e-4 | 0.13 | 50 |
  | `places365_standard` | Large-scale scene dataset, using an ImageNet-scale architecture with `z_dim >= 365` | 2048 | 512 | 20 | 1024 | 1e-4 | 1e-4 | 0.1 | 1000 |
  | `dtd` | Small texture dataset with 47 classes; smaller batches avoid overly coarse structure within a batch | 4096 | 128 | 30 | 512 | 1e-4 | 1e-4 | 0.1 | 10 |
  | `aircraft` | Small fine-grained dataset with 100 classes, maintaining `z_dim >= n_clusters` and a smaller batch size | 4096 | 128 | 50 | 512 | 1e-4 | 1e-4 | 0.1 | 10 |
  | `cars` | Fine-grained dataset with 196 classes; z dimension increased to 256 | 4096 | 256 | 50 | 1024 | 1e-4 | 1e-4 | 0.1 | 10 |
  | `flowers` | Small fine-grained dataset with 102 classes, retaining a moderate training duration and smaller batch size | 4096 | 128 | 50 | 512 | 1e-4 | 1e-4 | 0.1 | 10 |
  | `food` | Medium-to-large natural-image dataset with 101 classes, using a CIFAR-100-style batch size and medium sparsity | 4096 | 128 | 30 | 1500 | 1e-4 | 1e-4 | 0.1 | 50 |
  | `pets` | Small-to-medium fine-grained dataset with 37 classes, using low-dimensional representations and smaller batches | 4096 | 128 | 30 | 512 | 1e-4 | 1e-4 | 0.1 | 10 |
  | `ucf101` | Video frames/action images with 101 classes, treated as medium-scale multiclass features | 4096 | 128 | 30 | 1024 | 1e-4 | 1e-4 | 0.1 | 100 |

  Unknown datasets under LAION400M and LAION2B still use the official parser fallback:
  `hidden_dim=4096`, `z_dim=128`, `epochs=30`, `batch_size=2000`, `lr=3e-3`, `lr_c=2e-3`,
  `pieta=0.175`, `warmup=0`.
- LAION2B and SigLIP:
  All three `LAION2B` backbones use the paper/benchmark dataset profiles above, but do not use the LAION400M
  backbone profiles. `SigLIP + ViT-B/16` skips dataset/model profiles for all source datasets,
  using the current official parser fallback: `hidden_dim=4096`, `z_dim=128`,
  `epochs=30`, `batch_size=2000`, `lr=3e-3`, `lr_c=2e-3`, `pieta=0.175`,
  `warmup=0`, with `two_head` as the default. This also applies to domain shift with ImageNet as the source;
  Explicit parameters still take precedence.
- Confirmed `LAION400M` benchmark parameters: `single` corresponds to `single_head`, and `double` corresponds to the current default, `two_head`. In the tables, `default` means that the reported value has been verified to match the current implementation's fallback or dataset profile:

  `ViT-B/32`

  | dataset | `head` | `pieta` |
  | --- | ---: | ---: |
  | CIFAR-10 | single | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | single | default |
  | STL-10 | default | default |
  | ImageNet-10 | default | default |
  | ImageNet-Dogs | default | 0.2 |
  | DTD | default | 0.225 |
  | UCF101 | default | 0.2 |
  | ImageNet | single | default |
  | Places365-Standard | single | 0.2 |
  | Aircraft | default | 0.2 |
  | Cars | default | default |
  | Flowers | default | default |
  | Food | default | default |
  | Pets | default | default |
  | ImageNet-Variants | single | default |

  `ViT-B/16`

  | dataset | `head` | `pieta` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | single | default |
  | CIFAR-100 | default | default |
  | STL-10 | default | default |
  | ImageNet-10 | single | default |
  | ImageNet-Dogs | default | 0.2 |
  | DTD | default | 0.2 |
  | UCF101 | default | 0.2 |
  | ImageNet | single | default |
  | Places365-Standard | single | 0.2 |
  | Aircraft | default | 0.2 |
  | Cars | default | default |
  | Flowers | default | default |
  | Food | default | default |
  | Pets | default | 0.15 |
  | ImageNet-Variants | single | default |

  `ViT-L/14`

  | dataset | `head` | `pieta` |
  | --- | ---: | ---: |
  | CIFAR-10 | single | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | single | default |
  | STL-10 | default | default |
  | ImageNet-10 | default | default |
  | ImageNet-Dogs | default | 0.2 |
  | DTD | default | 0.15 |
  | UCF101 | default | 0.2 |
  | ImageNet | single | default |
  | Places365-Standard | single | 0.2 |
  | Aircraft | default | 0.2 |
  | Cars | default | default |
  | Flowers | default | default |
  | Food | default | default |
  | Pets | default | 0.15 |
  | ImageNet-Variants | single | default |
- Spectral clustering evaluation parameters:
  `select_best_epoch` defaults to `True`: training still uses only the source training split, but complete membership spectral clustering is performed on the held-out eval/test split after each evaluation epoch, and the predictions from the epoch with the best test metric are returned in the final report.
  `eval_every_epochs` defaults to `1`: evaluation runs after every epoch; increasing it reduces evaluation cost on large datasets.
  `best_epoch_metric` defaults to `acc` and can be set to `nmi` or `ari`; these metrics are used only to select the final predictions across epochs and do not participate in training.
  `spectral_kmeans_n_init` defaults to `10`
  `spectral_random_state` falls back to `seed` by default
  `spectral_solver_type` defaults to `lm`
  `spectral_tol` defaults to `0.0`
  `normalize_spectral_embedding` defaults to `True`
- Notes:
  This benchmark version integrates only CPP's main clustering method, excluding the additional analysis modules `optimalcluster.py` and `labeling.py`.
  Inputs are standardized to shared, pre-extracted OpenCLIP features;
  by default, the algorithm follows the two-head definition from the paper and official `main.py`, namely `Pi = Sinkhorn(abs(logits @ logits.T))`.
  If `single_head` is explicitly enabled, it uses the official `main_efficient.py` shortcut, `logits = z`, namely
  `Pi = Sinkhorn(abs(Z @ Z.T))`, allowing single-head and two-head settings to be compared across datasets.
  Unlike the official code's mini-batch validation during training, this benchmark maintains train/test separation:
  training uses only the training split; epoch selection and final reporting use only the test/eval split.

## PRO-DSC

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the target dataset's default evaluation split
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Cluster-count parameters:
  `train_n_clusters` defaults to the number of classes in the source training dataset
  `n_clusters` serves as a compatibility alias for `train_n_clusters` in this benchmark
  `eval_n_clusters` defaults to the number of classes in the evaluation dataset
  For domain shift, this means training uses the source class count by default, while final spectral clustering evaluation uses the target class count
- Features and caching:
  `image_batch_size` defaults to `256`
  PRO-DSC currently uses the shared cache of raw OpenCLIP image features directly, without manually downloading `.pt` feature files as required by the official repository
  Derived training outputs are stored in a method-specific directory keyed by `method_name + openclip model key + dataset token`; checkpoints are saved only when `save_checkpoint=True`
- Training parameters:
  `hidden_dim` defaults to the dataset profile value, or `4096` if no profile exists
  `z_dim` defaults to the dataset profile value; if no profile exists, it falls back according to the cluster count:
  `n_clusters >= 1000 -> 1024`, `n_clusters >= 100 -> 256`, otherwise `128`
  `epochs`, or its alias `epo`, defaults to the dataset profile value, or `100` if no profile exists
  `batch_size`, or its alias `bs`, defaults to the dataset profile value, or `1500` if no profile exists
  If the source training split contains fewer samples than `batch_size`, the effective training batch size automatically shrinks to the training split size; PRO-DSC still requires the effective batch size to exceed `train_n_clusters`
  `learning_rate`, or its alias `lr`, defaults to the dataset profile value, or `1e-4` if no profile exists
  `cluster_learning_rate`, or its alias `lr_c`, defaults to the dataset profile value, or `1e-4` if no profile exists
  `momentum`, or its alias `momo`, defaults to `0.9`
  `weight_decay_main`, or its alias `wd1`, defaults to `1e-4`
  `weight_decay_cluster`, or its alias `wd2`, defaults to `5e-3`
  `gamma` defaults to the dataset profile value, or `300` if no profile exists
  `beta` defaults to the dataset profile value, or `400` if no profile exists
  `pieta` defaults to the dataset profile value, or `0.1` if no profile exists
  `piiter` defaults to the dataset profile value, or `1` if no profile exists
  `eps` defaults to the dataset profile value, or `0.1` if no profile exists
  `warmup_steps`, or its alias `warmup`, defaults to the dataset profile value, or `-1` if no profile exists
  `save_checkpoint` defaults to `False`; when enabled, `save_every` or `checkpoint_every` defaults to `50`
  `seed` defaults to `42`
- LAION2B and SigLIP:
  All three `LAION2B` backbones use this section's dataset profiles, but do not use the LAION400M backbone
  profiles. `SigLIP + ViT-B/16` skips both profile layers and directly uses the general fallback above for cases without a profile.
  Thus, the SigLIP defaults for ImageNet are `gamma=300`, `beta=400`;
  Aircraft defaults to `gamma=300`. Explicit command-line parameters still have the highest priority; for example,
  `method.gamma=200` overrides Aircraft's `gamma` to `200`.
- Confirmed `LAION400M + ViT-B/32` benchmark parameters: `default` means that the reported value has been verified to match the current implementation's fallback, dataset profile, or the ImageNet profile used for ImageNet-Variants:

  | dataset | `gamma` | `beta` | `pieta` | `bs` | `lr_c` |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | CIFAR-10 | default | default | default | default | default |
  | CIFAR-20 | default | default | default | default | default |
  | CIFAR-100 | default | default | default | default | default |
  | STL-10 | default | default | default | default | default |
  | ImageNet-10 | default | default | default | default | default |
  | ImageNet-Dogs | default | default | default | default | default |
  | DTD | default | default | default | default | default |
  | UCF101 | default | default | default | default | default |
  | ImageNet | default | default | default | default | default |
  | Places365-Standard | default | default | default | default | default |
  | Aircraft | default | default | default | default | default |
  | Cars | default | default | default | default | default |
  | Flowers | default | default | default | default | default |
  | Food | default | default | default | default | default |
  | Pets | default | default | default | default | default |
  | ImageNet-Variants | default | default | default | default | default |

  Confirmed `LAION400M + ViT-B/16` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback, dataset profile, or the ImageNet profile used for ImageNet-Variants:

  | dataset | `gamma` | `beta` | `pieta` | `bs` | `lr_c` |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | CIFAR-10 | 200 | 500 | default | default | default |
  | CIFAR-20 | 200 | 500 | default | default | default |
  | CIFAR-100 | 100 | 300 | default | default | default |
  | STL-10 | default | default | default | default | default |
  | ImageNet-10 | default | default | default | default | default |
  | ImageNet-Dogs | default | default | default | default | default |
  | DTD | 100 | default | 0.15 | default | default |
  | UCF101 | default | default | default | default | default |
  | ImageNet | default | default | default | default | default |
  | Places365-Standard | default | default | default | default | default |
  | Aircraft | default | default | default | default | default |
  | Cars | default | default | default | default | default |
  | Flowers | default | default | default | 128 | default |
  | Food | default | default | default | default | default |
  | Pets | 200 | 300 | default | default | default |
  | ImageNet-Variants | default | default | default | default | default |

  Confirmed `LAION400M + ViT-L/14` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback, dataset profile, or the ImageNet profile used for ImageNet-Variants:

  | dataset | `gamma` | `beta` | `pieta` | `bs` | `lr_c` |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | CIFAR-10 | default | default | default | default | default |
  | CIFAR-20 | 200 | 600 | default | default | default |
  | CIFAR-100 | default | default | default | default | default |
  | STL-10 | default | default | default | default | default |
  | ImageNet-10 | default | default | default | default | default |
  | ImageNet-Dogs | default | default | default | default | default |
  | DTD | default | default | default | default | default |
  | UCF101 | default | default | default | 256 | default |
  | ImageNet | default | default | default | default | default |
  | Places365-Standard | default | default | default | default | default |
  | Aircraft | default | default | default | default | default |
  | Cars | default | default | default | default | default |
  | Flowers | default | default | default | 256 | default |
  | Food | 200 | default | default | default | default |
  | Pets | default | default | default | default | default |
  | ImageNet-Variants | default | default | default | default | default |
- Spectral clustering evaluation parameters:
  `spectral_kmeans_n_init` defaults to `10`
  `spectral_random_state` falls back to `seed` by default
  `spectral_solver_type` defaults to `lm`
  `spectral_tol` defaults to `0.0`
  `normalize_spectral_embedding` defaults to `True`
- Dataset profiles (LAION400M / LAION2B):
  The following profiles provide source-dataset defaults for both LAION400M and LAION2B; SigLIP skips them according to the rule above.
  The profiles provided by the official repository for `cifar10 / cifar20 / cifar100 / tinyimagenet / imagenet / imagenet_dogs` are retained.
  In this benchmark, ImageNet is fixed to `gamma=800`, `beta=600`, strengthening the block prior to improve cluster assignment when the class count is large.
  Datasets integrated into this benchmark but not covered by the official repository no longer all use a single fallback; benchmark profiles are instead provided according to dataset size, class count, and visual domain:

  | dataset | profile rationale | gamma | beta | pieta | piiter | z_dim | epochs | batch_size | warmup |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | `stl10` | CIFAR-10-like image subset with 10 classes but a smaller training split; longer training | 300 | 600 | 0.175 | 1 | 128 | 100 | 1024 | 200 |
  | `imagenet10` | ImageNet-Dogs-like ImageNet subset, retaining longer training and multiple Sinkhorn projections | 300 | 400 | 0.1 | 5 | 128 | 200 | 1024 | -1 |
  | `places365_standard` | Large-scale scene dataset, using an ImageNet-scale profile | 800 | 400 | 0.09 | 1 | 1024 | 100 | 2048 | 2000 |
  | `dtd` | Small texture dataset; smaller batches and low-dimensional representations avoid overly coarse structure within a batch | 300 | 400 | 0.1 | 1 | 128 | 200 | 512 | 100 |
  | `aircraft` | Small fine-grained dataset with 100 classes, using a moderate z dimension and smaller batch size | 200 | 400 | 0.1 | 1 | 256 | 200 | 512 | 100 |
  | `cars` | Fine-grained dataset with 196 classes; more classes and a larger training split, using a moderate z dimension and a batch size of 1024 | 300 | 400 | 0.1 | 1 | 256 | 200 | 1024 | 200 |
  | `flowers` | Small fine-grained dataset with 102 classes, using mild regularization and a higher cluster-head learning rate | 200 | 300 | 0.12 | 1 | 256 | 200 | 512 | 100 |
  | `food` | Medium-to-large natural-image dataset with 101 classes, using a CIFAR-100-style batch size and moderate z dimension | 300 | 400 | 0.1 | 1 | 256 | 100 | 1500 | 200 |
  | `pets` | Small-to-medium fine-grained dataset with 37 classes, using low-dimensional representations and smaller batches | 300 | 400 | 0.1 | 1 | 128 | 100 | 512 | 100 |
  | `ucf101` | Video frames/action data with 101 classes, treated as medium-scale multiclass image features with reduced regularization strength | 200 | 300 | 0.1 | 1 | 256 | 100 | 1024 | 200 |

  `flowers` additionally defaults to `cluster_learning_rate=3e-4`; all other benchmark profiles inherit the general `cluster_learning_rate=1e-4`.

  Unknown datasets still use the fallback: `gamma=300`, `beta=400`, `pieta=0.1`, `piiter=1`, `hidden_dim=4096`, `epochs=100`, `batch_size=1500`, `warmup=-1`, with `z_dim` inferred from the class count.
- Notes:
  The official PRO-DSC repository itself implements training on CLIP features.
  This benchmark version retains the official main pipeline, `TCR warmup + Sinkhorn projection + self-expression/block-prior + final spectral clustering`,
  but standardizes the input features to shared, raw `OpenCLIP` image features;
  to follow this benchmark's `train/test` protocol, it does not periodically read evaluation split metrics during training for reporting or model selection. It performs a single label-free spectral clustering prediction using the model from the final epoch.

## TEMI

- Common interface parameters:
  `openclip_pretraining`, `openclip_backbone`, `train_split`, `test_split`, `max_train_samples`, `max_test_samples`
- Domain-shift interface parameters:
  `source_dataset_name` defaults to `image_dataset.name` in the experiment configuration
  `target_dataset_name`, or its alias `target_dataset`, defaults to `None`
  `target_test_split` defaults to the target dataset's default evaluation split
  If the target is `ImageNet-Variants`, the source must be `ImageNet`
- Output dimension and cluster count:
  `out_dim` defaults to `n_clusters`; if neither is provided, it uses the number of classes in the source training dataset, with a default of `1000` for full ImageNet domain shift
  Final predictions are obtained by applying `argmax` to the head logits
- Features and KNN:
  `image_batch_size` defaults to `256`
  `knn` defaults to `50`; the ImageNet Table 10 profile and Places365 large-scale profile default to `25`
  `knn_chunk_size` defaults to `4096`; the Places365 large-scale profile defaults to `512`
  `force_recompute_knn` defaults to `False`
  TEMI uses shared, raw OpenCLIP image features; the training KNN neighbor table is a method-specific derived output saved under `temi + openclip model key + dataset/out_dim/knn`
- Embedding augmentation:
  `embed_aug` defaults to `none`; `gaussian` is also available
  `num_augs` defaults to `1`
  `gaussian_std` defaults to `0.1`
- Student/teacher head parameters:
  `num_heads` defaults to `16`; the Table 10 profile defaults to `50`
  `nlayers` defaults to `2`
  `hidden_dim` defaults to `512`
  `bottleneck_dim` defaults to `256`
  `use_bn_in_head` defaults to `False`
  `head_dropout_prob` defaults to `0.0`
  `head_final_gelu` defaults to `False`
  `norm_last_layer` defaults to `False`
  `l2_norm` defaults to `False`
  `embed_norm` defaults to `False`
- TEMI loss parameters:
  `beta` defaults to `0.6`
  `student_temp` defaults to `0.1`
  `warmup_teacher_temp` defaults to `0.1`
  `teacher_temp` defaults to `0.1`
  `warmup_teacher_temp_epochs` defaults to `30`
  `probs_momentum` defaults to `0.9`
  `positive_pmi` defaults to `False`
  Confirmed `LAION400M + ViT-B/32` benchmark parameters: `default` means that the reported value has been verified to match the current implementation's fallback or dataset profile:

  | dataset | `knn` | `teacher_temp` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | default | default |
  | STL-10 | default | default |
  | ImageNet-10 | default | default |
  | ImageNet-Dogs | default | default |
  | DTD | default | default |
  | UCF101 | default | default |
  | ImageNet | default | default |
  | Places365-Standard | default | default |
  | Aircraft | default | default |
  | Cars | default | default |
  | Flowers | default | default |
  | Food | default | default |
  | Pets | default | default |

  Confirmed `LAION400M + ViT-B/16` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback or dataset profile:

  | dataset | `knn` | `teacher_temp` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | default | default |
  | STL-10 | default | default |
  | ImageNet-10 | default | default |
  | ImageNet-Dogs | default | default |
  | DTD | default | default |
  | UCF101 | default | default |
  | ImageNet | default | default |
  | Places365-Standard | default | default |
  | Aircraft | default | 0.05 |
  | Cars | default | default |
  | Flowers | 25 | 0.02 |
  | Food | default | default |
  | Pets | 25 | 0.15 |

  Confirmed `LAION400M + ViT-L/14` benchmark parameters: `default` likewise means that the reported value has been verified to match the current implementation's fallback or dataset profile:

  | dataset | `knn` | `teacher_temp` |
  | --- | ---: | ---: |
  | CIFAR-10 | default | default |
  | CIFAR-20 | default | default |
  | CIFAR-100 | default | default |
  | STL-10 | default | default |
  | ImageNet-10 | default | default |
  | ImageNet-Dogs | default | default |
  | DTD | default | 0.05 |
  | UCF101 | default | default |
  | ImageNet | default | default |
  | Places365-Standard | default | default |
  | Aircraft | default | default |
  | Cars | default | default |
  | Flowers | 25 | 0.02 |
  | Food | default | default |
  | Pets | default | default |
- Optimization parameters:
  `epochs` defaults to `100`; the Table 10 profile defaults to `200`, and the STL10 Table 10 profile defaults to `800`
  `batch_size` defaults to `1024`
  In the Table 10 profiles, CIFAR10/CIFAR20/CIFAR100/STL10 default to `512`, and ImageNet defaults to `1024`;
  Small and medium dataset profiles default to `512`, and the Places365 large-scale profile defaults to `1024`
  If the source training split contains fewer samples than `batch_size`, the effective training batch size automatically shrinks to the training split size, preventing `drop_last=True` from producing an empty training loop
  `eval_batch_size` falls back to `batch_size` by default
  `optimizer` defaults to `adamw`; `adamw` and `sgd` are currently supported
  `learning_rate`, or its alias `lr`, defaults to `1e-4`
  `min_learning_rate`, or its alias `min_lr`, defaults to `1e-4`
  `warmup_epochs` defaults to `20`; the ImageNet Table 10 profile defaults to `10`
  `weight_decay` defaults to `1e-4`
  `weight_decay_end` defaults to `1e-4`
  `momentum_teacher` defaults to `0.996`
  `max_momentum_teacher` defaults to `0.996`
  `clip_grad` defaults to `0.0`
  `freeze_last_layer` defaults to `1`
  `use_fp16` defaults to `False`
  `save_checkpoint` defaults to `False`; when enabled, `save_checkpoint_frequency`, or its aliases `saveckp_freq`/`save_every`, defaults to `20`
  `seed` defaults to `0`
- Dataset profiles (LAION400M / LAION2B):
  `cifar10`, `cifar100`, `cifar20`, `stl10`, and `imagenet` use the clustering-head training settings from Table 10 of the paper by default.
  Shared settings: `num_heads=50`, `optimizer=adamw`, `learning_rate=1e-4`, `min_learning_rate=1e-4`,
  `weight_decay=1e-4`, `weight_decay_end=1e-4`, `momentum_teacher=0.996`,
  `max_momentum_teacher=0.996`, `embed_aug=none`, `num_augs=1`
  CIFAR10/CIFAR100:`knn=50`, `epochs=200`, `batch_size=512`, `warmup_epochs=20`, `beta=0.6`
  CIFAR20:`knn=50`, `epochs=200`, `batch_size=512`, `warmup_epochs=20`, `beta=0.55`
  STL10:`knn=50`, `epochs=800`, `batch_size=512`, `warmup_epochs=20`, `beta=0.6`
  ImageNet:`knn=25`, `epochs=200`, `batch_size=1024`, `warmup_epochs=10`, `beta=0.6`
  `dtd`, `aircraft`, `cars`, `flowers`, `food`, `pets`, `ucf101`, `imagenet10`, `imagenet_dogs`
  use the small and medium dataset profile by default:
  `knn=50`, `num_heads=50`, `epochs=200`, `batch_size=512`, `warmup_epochs=20`, `beta=0.6`
  `places365_standard` uses the large-scale dataset profile by default:
  `knn=25`, `num_heads=50`, `epochs=200`, `batch_size=1024`, `warmup_epochs=10`,
  `beta=0.6`, `knn_chunk_size=512`
- LAION2B and SigLIP:
  All three `LAION2B` backbones use the Table 10 / dataset profiles above, but do not use the LAION400M
  backbone profiles; for example, STL10 uses `num_heads=50`, `epochs=800`, `batch_size=512`.
  `SigLIP + ViT-B/16` skips both profile layers and uses the current generic fallback:
  `knn=50`, `num_heads=16`, `epochs=100`, `batch_size=1024`, `optimizer=adamw`,
  `learning_rate=1e-4`, `min_learning_rate=1e-4`, `weight_decay=1e-4`,
  `weight_decay_end=1e-4`, `warmup_epochs=20`, `beta=0.6`, `teacher_temp=0.1`,
  `student_temp=0.1`; explicit parameters still take precedence.
- Notes:
  The official TEMI repository first generates pretrained-model embeddings and training KNN, then trains a student/teacher multi-head clustering head.
  This benchmark version retains the main pipeline, `KNN pair sampling + EMA teacher + multi-head TEMI weighted mutual-information loss + train-loss best-head selection`,
  but standardizes the input embeddings to shared, raw `OpenCLIP` image features.
  The official `eval_experiment.py` scans checkpoints and selects the best checkpoint using validation labels; to avoid evaluation/test leakage, this benchmark does not use evaluation labels for model selection. It selects the best head using only the training-side head loss and evaluates once after the final epoch.
