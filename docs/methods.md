# Methods, Sources, and Benchmark Scope

VLM4Cluster's core comparison contains **16 methods**: 4 classical, 5 deep, and
7 language-assisted image clustering methods. The CLI registers these same
**16 implementations**. Method availability does not imply that every
dataset/model combination has a recorded result.

Use the [reproduction guide](reproduction.md) for commands and resources, and the
[hyperparameter reference](method_hyperparameters.md) for the fixed profiles and
complete parameter descriptions.

## Core methods and references

The repository links below identify author-maintained reference implementations,
where verified. They are references for understanding each method, not additional
repositories that must be installed to run VLM4Cluster.

### Classical image clustering

| Method / CLI identifier | Paper or algorithm reference | Reference implementation |
| --- | --- | --- |
| K-means / `clip_kmeans` | [FAISS clustering documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization) | [FAISS](https://github.com/facebookresearch/faiss), with [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) fallback |
| Spectral Clustering / `clip_sc` | [scikit-learn SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) | [scikit-learn](https://github.com/scikit-learn/scikit-learn) |
| SSC-OMP / `ssc_omp` | [Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit](https://openaccess.thecvf.com/content_cvpr_2016/html/You_Scalable_Sparse_Subspace_CVPR_2016_paper.html), CVPR 2016 | [ChongYou/subspace-clustering](https://github.com/ChongYou/subspace-clustering) |
| EnSC / `ensc` | [Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering](https://openaccess.thecvf.com/content_cvpr_2016/html/You_Oracle_Based_Active_CVPR_2016_paper.html), CVPR 2016 | [ChongYou/subspace-clustering](https://github.com/ChongYou/subspace-clustering) |

K-means and spectral clustering are general algorithms rather than methods with a
single official image-clustering repository. Their entries identify the actual
computational backends used here.

### Deep image clustering

| Method / CLI identifier | Paper | Official reference repository |
| --- | --- | --- |
| IDC / `idc` | [Interactive Deep Clustering via Value Mining](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ac4365b98bc242acd5ab974a05c68a8-Abstract-Conference.html), NeurIPS 2024 | [XLearning-SCU/2024-NeurIPS-IDC](https://github.com/XLearning-SCU/2024-NeurIPS-IDC) |
| SCAN / `scan` | [SCAN: Learning to Classify Images without Labels](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1057_ECCV_2020_paper.php), ECCV 2020 | [wvangansbeke/Unsupervised-Classification](https://github.com/wvangansbeke/Unsupervised-Classification) |
| CPP / `cpp` | [Image Clustering via the Principle of Rate Reduction in the Age of Pretrained Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/697200c9d1710c2799720b660abd11bb-Abstract-Conference.html), ICLR 2024 | [LeslieTrue/CPP](https://github.com/LeslieTrue/CPP) |
| TEMI / `temi` | [Exploring the Limits of Deep Image Clustering using Pretrained Models](https://proceedings.bmvc2023.org/297/), BMVC 2023 | [HHU-MMBS/TEMI-official-BMVC2023](https://github.com/HHU-MMBS/TEMI-official-BMVC2023) |
| PRO-DSC / `pro_dsc` | [Exploring a Principled Framework for Deep Subspace Clustering](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c20ac0df6c213db6d3a930fe9c7296c8-Abstract-Conference.html), ICLR 2025 | [mengxianghan123/PRO-DSC](https://github.com/mengxianghan123/PRO-DSC) |

### Language-assisted image clustering

| Method / CLI identifier | Paper | Official reference repository |
| --- | --- | --- |
| SIC / `sic` | [Semantic-Enhanced Image Clustering](https://ojs.aaai.org/index.php/AAAI/article/view/25841), AAAI 2023 | [Bruce-XJChen/SIC](https://github.com/Bruce-XJChen/SIC) |
| TAC / `tac` | [Image Clustering with External Guidance](https://proceedings.mlr.press/v235/li24aa.html), ICML 2024 | [XLearning-SCU/2024-ICML-TAC](https://github.com/XLearning-SCU/2024-ICML-TAC) |
| SAC / `sac` | [Semantic-Augmented Image Clustering via Adaptive Multi-Modal Collaboration](https://ojs.aaai.org/index.php/AAAI/article/view/40073), AAAI 2026 | Not verified in this source audit |
| GradNorm / `gradnorm` | [On the Provable Importance of Gradients for Autonomous Language-Assisted Image Clustering](https://openaccess.thecvf.com/content/ICCV2025/html/Peng_On_the_Provable_Importance_of_Gradients_for_Autonomous_Language-Assisted_Image_ICCV_2025_paper.html), ICCV 2025; [extended version](https://arxiv.org/abs/2510.16335) | [60pen9/On-the-Provable-Importance-of-Gradients-for-Language-Assisted-Image-Clustering](https://github.com/60pen9/On-the-Provable-Importance-of-Gradients-for-Language-Assisted-Image-Clustering) |
| NTK-SC / `ntk_sc` | [Delving into Spectral Clustering with Vision-Language Representations](https://arxiv.org/abs/2602.09586), ICLR 2026 | [hyw1008/ICLR2026-Delving-into-Spectral-Clustering-with-Vision-Language-Representations](https://github.com/hyw1008/ICLR2026-Delving-into-Spectral-Clustering-with-Vision-Language-Representations) |
| SEIC / `seic` | [Self-Enhanced Image Clustering with Cross-Modal Semantic Consistency](https://ojs.aaai.org/index.php/AAAI/article/view/39506), AAAI 2026 | Paper-based implementation; official code link not verified |
| MAGIC / `magic` | [MAGIC: Multi-Granularity Language-Informed Image Clustering](https://openreview.net/forum?id=eyo7TITaF9), ICML 2026 | Not verified in this source audit |

TAC and TAC* count as one method. In benchmark result tables, **TAC** selects the
training-free variant (`method.train_cluster_heads=false`), while **TAC*** selects
cluster-head training (`method.train_cluster_heads=true`). This produces 17
method/variant rows for 16 methods. The implementation defaults to the trained
variant, so reproduction commands specify this switch explicitly.

## Reference snapshots and upstream licenses

The following public upstream snapshots and repository-level license files were
checked on **2026-09-08**. These commits make this source audit traceable. They
are **not claimed to be the commits originally used during method integration**:
that historical provenance was not recorded in the existing implementation
notes. The exception is the bundled WordNet CSV, whose exact source commit and
byte identity are recorded in its [manifest](../data/WordNetNouns.source.json).

| Methods | Upstream snapshot checked | Repository-level license at that snapshot |
| --- | --- | --- |
| SSC-OMP, EnSC | [55e330e](https://github.com/ChongYou/subspace-clustering/tree/55e330e38415951f140e74739c869306acdb7a98) | [MIT](https://github.com/ChongYou/subspace-clustering/blob/55e330e38415951f140e74739c869306acdb7a98/LICENSE) |
| IDC | [96cb852](https://github.com/XLearning-SCU/2024-NeurIPS-IDC/tree/96cb852a27d25c59580cc78b3a88ee11a83ee5f2) | [Apache-2.0](https://github.com/XLearning-SCU/2024-NeurIPS-IDC/blob/96cb852a27d25c59580cc78b3a88ee11a83ee5f2/LICENSE) |
| SCAN | [952ec31](https://github.com/wvangansbeke/Unsupervised-Classification/tree/952ec31eee2c38e7233d8ad3ac0de39bb031877a) | [CC BY-NC 4.0](https://github.com/wvangansbeke/Unsupervised-Classification/blob/952ec31eee2c38e7233d8ad3ac0de39bb031877a/LICENSE) |
| CPP | [cb39bdc](https://github.com/LeslieTrue/CPP/tree/cb39bdc65f346b4cb5950d14af2e393a5f6a3566) | No repository-level license identified |
| TEMI | [c6a81a6](https://github.com/HHU-MMBS/TEMI-official-BMVC2023/tree/c6a81a6d684f7b8d19500c4ec0f846335e90ee3a) | [Apache-2.0](https://github.com/HHU-MMBS/TEMI-official-BMVC2023/blob/c6a81a6d684f7b8d19500c4ec0f846335e90ee3a/LICENSE) |
| PRO-DSC | [33d5f1f](https://github.com/mengxianghan123/PRO-DSC/tree/33d5f1f0077ba25b44151e6ef67ce3906f281951) | [CC BY-NC 4.0](https://github.com/mengxianghan123/PRO-DSC/blob/33d5f1f0077ba25b44151e6ef67ce3906f281951/LICENSE) |
| SIC | [995c4c2](https://github.com/Bruce-XJChen/SIC/tree/995c4c26856980e514adb7388d2eaf020badb689) | [Apache-2.0](https://github.com/Bruce-XJChen/SIC/blob/995c4c26856980e514adb7388d2eaf020badb689/LICENSE) |
| TAC | [830e02c](https://github.com/XLearning-SCU/2024-ICML-TAC/tree/830e02c625d98784dfe97bb57226e94bf53413de) | No repository-level license identified; [WordNet resource notice](../data/WordNet-LICENSE.txt) is separate |
| GradNorm | [d45db7a](https://github.com/60pen9/On-the-Provable-Importance-of-Gradients-for-Language-Assisted-Image-Clustering/tree/d45db7a6f82fa99be56f990ed18e1765b61a66b0) | No repository-level license identified |
| NTK-SC | [a6aec45](https://github.com/hyw1008/ICLR2026-Delving-into-Spectral-Clustering-with-Vision-Language-Representations/tree/a6aec45910e98057c928eb1851886297d50a4a18) | No repository-level license identified |
| SAC, SEIC, MAGIC | Official source commit not verified | Not verified |

[FAISS](https://github.com/facebookresearch/faiss) uses MIT and
[scikit-learn](https://github.com/scikit-learn/scikit-learn) uses BSD-3-Clause.
Their installed package versions, rather than an upstream development snapshot,
determine the backends for an experiment; preserve the environment export with
the results.

This table records upstream declarations, not a project-wide relicensing of
third-party work. In particular, SCAN and PRO-DSC declare noncommercial terms.
An absent license entry is not evidence of unrestricted reuse. File-level
provenance and any redistribution obligations still need to be resolved before
assigning a blanket license to the assembled codebase. Paper licenses and dataset
licenses do not establish a license for the associated program code.

## What the benchmark implementation runs

The benchmark uses the shared OpenCLIP model pool and the documented dataset
splits. WordNet-based methods share the bundled TAC vocabulary. Those common
choices can differ from an original paper's backbone, text pool, or evaluation
protocol. The details below describe the current local execution paths; they do
not claim numerical equivalence to the original papers' tables.

| Method and local implementation | Material implementation or evaluation details |
| --- | --- |
| [K-means](../vlm4cluster/methods/clip_kmeans/adapter.py) | Clusters OpenCLIP features directly on the evaluation split. FAISS is preferred, with scikit-learn fallback. This is an evaluation-only clustering baseline. |
| [Spectral Clustering](../vlm4cluster/methods/clip_sc/adapter.py) | Uses the full dot-product Gram matrix of normalized evaluation features, with zero diagonal, as the precomputed affinity for scikit-learn spectral clustering. The previous full or mutual-neighbor RBF graph is available through `affinity_kernel=rbf`. |
| [SSC-OMP](../vlm4cluster/methods/ssc_omp/adapter.py) | Sparse self-representation and spectral clustering operate on evaluation-split OpenCLIP features. It does not learn a source-side classifier. |
| [EnSC](../vlm4cluster/methods/ensc/adapter.py) | Elastic-net self-representation operates on evaluation features. The default solver is `lasso_lars`; the existing solver logic uses `tau=1` for the lasso solvers. SPAMS is optional and requires a separate installation if explicitly selected. |
| [IDC](../vlm4cluster/methods/idc/adapter.py) | Uses frozen OpenCLIP features and a trainable clustering head. Training labels simulate oracle feedback for value mining, so this is an interactive-supervision setting. Original IDC/TCL pretrained checkpoints are not required by this implementation. |
| [SCAN](../vlm4cluster/methods/scan/adapter.py) | Trains clustering heads on frozen features using neighbor consistency and entropy regularization. Backbone pretraining, image augmentation training, and the original self-labeling stage are not run. Head/epoch selection uses SCAN loss on the evaluation split by default, although it does not use evaluation labels. |
| [CPP](../vlm4cluster/methods/cpp/adapter.py) | Uses frozen OpenCLIP features, the two-head formulation by default, and membership-based spectral clustering. `select_best_epoch=true` selects final predictions using evaluation/test ACC by default. This is test-metric selection, even though training updates use the training split. |
| [TEMI](../vlm4cluster/methods/temi/adapter.py) | Uses shared frozen OpenCLIP features, training KNN, and EMA teacher/student clustering heads. The best head is selected by training loss, followed by evaluation after the final epoch. |
| [PRO-DSC](../vlm4cluster/methods/pro_dsc/adapter.py) | Uses shared OpenCLIP features rather than manually downloaded upstream `.pt` feature files. Retains TCR warmup, Sinkhorn projection, self-expression/block-prior training, and final spectral clustering. Evaluation uses the final training epoch. |
| [SIC](../vlm4cluster/methods/sic/adapter.py) | Retains semantic-space construction, KNN, and cluster-head training. Uses shared WordNet instead of upstream `noun.csv`, and fixed training duration rather than evaluation-label early stopping. |
| [TAC](../vlm4cluster/methods/tac/adapter.py) | Uses source-training images for noun selection and text construction, followed by training-free concatenated-feature clustering or cluster-head training, according to `train_cluster_heads`. |
| [SAC](../vlm4cluster/methods/sac/adapter.py) | Uses TAC-style WordNet/OpenCLIP text counterparts in place of BLIP-2 captions and SBERT embeddings. Preserves adaptive multimodal cluster-head training and evaluates after a fixed training duration. |
| [GradNorm](../vlm4cluster/methods/gradnorm/adapter.py) | Applies gradient-based noun filtering to the shared vocabulary using source-training features, then constructs text counterparts for training-free clustering. |
| [NTK-SC](../vlm4cluster/methods/ntk_sc/adapter.py) | Uses source-side noun selection, prompt-specific pNTK graphs, and the configured diffusion/spectral backend. Record `red_backend` and graph settings because the automatic backend depends on problem size. |
| [SEIC](../vlm4cluster/methods/seic/adapter.py) | The default is Stage 1 only (`stage2=false`, reported internally as `seic_stage1`). Full self-enhancement requires `stage2=true`; it fine-tunes visual LoRA parameters and image heads. Temperature and LoRA details are local interpretations of the paper. |
| [MAGIC](../vlm4cluster/methods/magic/adapter.py) | Uses TAC-style WordNet text construction for fine/coarse semantic granularities in place of Qwen-generated captions and spaCy concepts. The multimodal training path and prediction variants are described in the hyperparameter reference. |

For source-to-target experiments, distinguish **source-trained inference** from
**clustering the target evaluation features**. The four classical implementations
currently perform the latter; having `source_dataset_name`/`target_dataset_name`
arguments does not turn them into source-trained methods. They should not be
reported as ImageNet-trained transfer results.

The shared `internal_metrics_only` option changes the final evaluator. It does
not remove IDC feedback, CPP's default test-metric selection, SCAN's use of
evaluation features for selection, or a configured cluster count. See
[evaluation choices](reproduction.md#evaluation-and-variant-choices) before
interpreting an experiment as fully label-free.
