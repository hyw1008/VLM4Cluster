# Leaderboard and Recorded Results

Explore the [interactive leaderboard](https://vlm4cluster-leaderboard.yuanwei-hu.chatgpt.site)
to compare recorded results by dataset, method, metric, and vision-language model
setting. The public website is maintained separately from this experiment codebase.

## Recorded coverage

The snapshot dated **2026-09-08** has one row per method/variant, dataset, model
setting, and evaluation track. Each row contains NMI, ACC, and ARI. The counts
below describe imported table rows; they do not count repeated random-seed runs
or represent the total number of experiments performed for the benchmark.

| Evaluation | Pretrained model | Backbone | Recorded rows |
| --- | --- | --- | ---: |
| Standard | LAION-400M | ViT-B/32 | 320 |
| Standard | LAION-400M | ViT-B/16 | 320 |
| Standard | LAION-400M | ViT-L/14 | 320 |
| Standard | LAION-2B | ViT-B/32 | 320 |
| Model comparison | SigLIP / WebLI | ViT-B/16 | 320 |
| AnyAttack-Cos, epsilon 8/255 | LAION-400M | ViT-B/32 | 255 |

For each of the first five settings, the four classical methods have results
on the 15 non-OOD datasets. The other 13 method/variant rows cover all 20
datasets. Classical-method OOD measurements are unreported in this snapshot.
AnyAttack results cover 17 method/variant rows on the 15 non-OOD datasets.
Model settings supported by the codebase but absent from this table have no
recorded results in the snapshot.

## Reading the tables

- **16 methods, 17 rows:** TAC is the training-free variant
  (`method.train_cluster_heads=false`); TAC* trains clustering heads
  (`method.train_cluster_heads=true`). They count as one method and are
  displayed separately.
- **Metric scale:** NMI, ACC, and ARI are multiplied by 100 for display and
  shown with one decimal place. For example, a recorded value of `0.9457`
  is displayed as `94.6`. ARI can be negative.
- **Ranking:** the selected metric is sorted using the recorded four-decimal
  values, before display rounding. Equal recorded scores share a rank;
  scores that only look equal after rounding can have different ranks.
- **Missing measurements:** a dash and the `Pending` status indicate an
  unreported result. They do not represent zero scores and are not ranked.
- **Model comparison:** the LAION-400M and SigLIP ViT-B/16 checkpoints also
  differ in pretraining data and training recipe. Their score differences
  do not isolate the effect of the training objective.

The snapshot contains recorded point estimates. It does not establish
across-seed means, standard deviations, or statistical significance.

## Robustness and efficiency

The recorded robustness setting uses the AnyAttack-Cos `coco_cos.pt` decoder,
epsilon **8/255**, and shuffled targets. Training uses clean images; the
evaluation images are perturbed. The framework default is **8/255**, matching
this snapshot. Set `attack.eps=0.03137254901960784` (8 divided by 255) to state
the budget explicitly in a reproduction command.

Seventeen ImageNet-1K AnyAttack rows additionally include combined training and
evaluation time in **minutes**, peak CPU memory in **GiB**, and peak GPU memory
in **GiB**. The framework's report fields use seconds and MiB; compare quantities
after converting their units. These efficiency records were provided alongside
the experiment results and are not regenerated from per-run reports by the
leaderboard.

NTK-SC efficiency was measured on a different compute node from the other
methods. These records are not a controlled comparison on identical hardware.
Other datasets and tracks have no efficiency measurements in this snapshot.

Use the [reproduction guide](reproduction.md) for experiment commands and the
[method notes](methods.md#what-the-benchmark-implementation-runs) for adaptation
and model-selection details. Preserve each run's reports, resolved configuration,
seed, environment, and hardware information when producing new results.
