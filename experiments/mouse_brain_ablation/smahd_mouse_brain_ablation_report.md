# SMAHD mouse brain real-data ablation

Dataset: mouse brain ATAC from the configured SMART data directory. Each variant was run with seeds 101, 202 and 303 for 80 epochs. Evaluation uses the available `mclust` reference clustering labels, so these values should be interpreted as agreement with reference clusters rather than manual anatomical ground truth.

## Per-dataset summary

| Dataset | Variant | ARI | NMI | AMI | FMI | Peak memory MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Mouse brain ATAC | SMAHD full | 0.388 +/- 0.013 | 0.400 +/- 0.010 | 0.399 +/- 0.010 | 0.480 +/- 0.011 | 57.2 +/- 0.2 |
| Mouse brain ATAC | Equal weights | 0.390 +/- 0.012 | 0.408 +/- 0.007 | 0.407 +/- 0.007 | 0.485 +/- 0.007 | 57.2 +/- 0.2 |
| Mouse brain ATAC | Early concat | 0.378 +/- 0.023 | 0.399 +/- 0.014 | 0.398 +/- 0.014 | 0.478 +/- 0.014 | 42.1 +/- 0.0 |
| Mouse brain ATAC | Full-batch | 0.369 +/- 0.032 | 0.389 +/- 0.031 | 0.387 +/- 0.031 | 0.476 +/- 0.031 | 866.8 +/- 0.0 |
| Mouse brain ATAC | RNA only | 0.297 +/- 0.006 | 0.364 +/- 0.012 | 0.363 +/- 0.012 | 0.409 +/- 0.013 | 41.9 +/- 0.2 |
| Mouse brain ATAC | ATAC only | 0.367 +/- 0.010 | 0.383 +/- 0.003 | 0.382 +/- 0.003 | 0.462 +/- 0.006 | 41.9 +/- 0.2 |
| Mouse brain ATAC | GCN encoder | 0.375 +/- 0.010 | 0.396 +/- 0.009 | 0.395 +/- 0.009 | 0.475 +/- 0.008 | 23.3 +/- 0.0 |
| Mouse brain ATAC | SAGE encoder | 0.286 +/- 0.011 | 0.384 +/- 0.014 | 0.383 +/- 0.014 | 0.372 +/- 0.012 | 22.3 +/- 0.0 |

## Average across mouse brain datasets

| Variant | ARI_mean | NMI_mean | peak_memory_mib_mean |
| --- | --- | --- | --- |
| SMAHD full | 0.3879 | 0.3999 | 57.2477 |
| Equal weights | 0.3895 | 0.4084 | 57.2477 |
| Early concat | 0.3779 | 0.3989 | 42.1112 |
| Full-batch | 0.3689 | 0.3888 | 866.8057 |
| RNA only | 0.2969 | 0.3645 | 41.9028 |
| ATAC only | 0.3666 | 0.3834 | 41.9028 |
| GCN encoder | 0.3751 | 0.3965 | 23.3433 |
| SAGE encoder | 0.2856 | 0.3838 | 22.3345 |

## Interpretation

- The mouse brain datasets provide a broader real-data stress test than the previous RNA+ADT sections because the second modality is ATAC/peak-derived rather than protein-derived.
- Micro-cluster training should be interpreted primarily as a scalability result: compare SMAHD full with full-batch memory at similar ARI/NMI.
- Modality and backbone behavior is dataset-dependent under the available reference clustering labels. Treat backbone choice and modality weights as sensitivity settings rather than universal constants.
