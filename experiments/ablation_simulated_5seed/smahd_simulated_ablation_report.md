# SMAHD simulated ablation report

Configuration: synthetic three-view spatial multi-omics benchmark; 4 domains; 5 random seeds (11, 22, 33, 44, 55); 80 epochs; 1,400 spots per replicate; KNN spatial graph with k=15; clustering by KMeans with the known number of domains.

## Summary table

| Variant | ARI | NMI | AMI | FMI | Runtime_s | Peak_memory_MiB |
| --- | --- | --- | --- | --- | --- | --- |
| SMAHD (GAT, view-specific, micro-cluster) | 0.479 ± 0.016 | 0.458 ± 0.016 | 0.456 ± 0.016 | 0.609 ± 0.012 | 11.90 ± 0.50 | 53.1 ± 0.4 |
| Equal reconstruction weights | 0.479 ± 0.016 | 0.458 ± 0.017 | 0.457 ± 0.017 | 0.610 ± 0.012 | 11.69 ± 0.39 | 53.1 ± 0.4 |
| Early concatenation, single encoder | 0.428 ± 0.050 | 0.417 ± 0.041 | 0.416 ± 0.042 | 0.572 ± 0.037 | 5.00 ± 0.33 | 44.0 ± 0.4 |
| Full-batch training | 0.474 ± 0.022 | 0.455 ± 0.020 | 0.453 ± 0.020 | 0.606 ± 0.016 | 2.69 ± 0.16 | 267.4 ± 0.1 |
| GCN encoder | 0.453 ± 0.035 | 0.435 ± 0.031 | 0.434 ± 0.031 | 0.590 ± 0.026 | 6.13 ± 0.21 | 22.6 ± 0.1 |
| SAGE encoder | 0.815 ± 0.041 | 0.774 ± 0.035 | 0.774 ± 0.035 | 0.861 ± 0.031 | 7.08 ± 0.33 | 21.0 ± 0.0 |
| GraphConv encoder | 0.151 ± 0.057 | 0.180 ± 0.065 | 0.178 ± 0.065 | 0.381 ± 0.047 | 4.80 ± 0.14 | 20.3 ± 0.0 |

## Paired comparisons against SMAHD_full

- Equal reconstruction weights: mean ARI delta (SMAHD_full - variant) = -0.0004, paired t-test p = 0.5932.
- Early concatenation, single encoder: mean ARI delta (SMAHD_full - variant) = 0.0514, paired t-test p = 0.02957.
- Full-batch training: mean ARI delta (SMAHD_full - variant) = 0.0049, paired t-test p = 0.1496.
- GCN encoder: mean ARI delta (SMAHD_full - variant) = 0.0264, paired t-test p = 0.04216.
- SAGE encoder: mean ARI delta (SMAHD_full - variant) = -0.3356, paired t-test p = 0.0001016.
- GraphConv encoder: mean ARI delta (SMAHD_full - variant) = 0.3284, paired t-test p = 0.000394.

## Interpretation

Micro-cluster sampling reduced peak GPU memory by 5.0x relative to full-batch training on the same simulated benchmark, with only a small ARI change (delta = 0.0049).
Replacing view-specific encoders with early feature concatenation reduced ARI by 0.0514 on average, supporting the use of modality-specific feature extraction before fusion.
Equal reconstruction weighting was essentially tied with the selected weights in this simulation (delta = -0.0004); this result supports reporting the weighting scheme as a sensitivity parameter rather than claiming it is universally superior.
The encoder comparison is mixed: GAT outperformed GCN and GraphConv, whereas SAGE produced the highest ARI in this synthetic setting. This should be described as encoder sensitivity unless additional real-data experiments show GAT is consistently preferred.

## Suggested manuscript text

We further performed module-wise ablation on a simulated three-view spatial multi-omics benchmark over five random seeds. Micro-cluster mini-batch training preserved clustering accuracy while reducing peak GPU memory by approximately five-fold relative to full-batch training. Removing view-specific encoders and using early feature concatenation reduced mean ARI, supporting separate modality-specific graph encoders before latent fusion. Reconstruction-weight sensitivity was modest in this simulation, whereas encoder choice affected performance, with GAT outperforming GCN and GraphConv but SAGE performing strongly. These results suggest that scalable subgraph sampling and modality-aware encoding are important components of SMAHD, while reconstruction weights and graph convolution backbones should be reported explicitly as tunable implementation choices.
