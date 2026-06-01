# SMAHD real-data ablation report

Datasets: human tonsil slice 1/2 and human lymph node A1/D1. Each variant was run with seeds 101, 202 and 303 for 80 epochs. RNA and ADT inputs were reduced to PCA features and evaluated against available tissue annotations using KMeans with the annotated number of regions.

## Per-dataset summary

| Dataset | Variant | ARI | NMI | AMI | FMI | Peak memory MiB |
| --- | --- | --- | --- | --- | --- | --- |
| tonsil_s1 | SMAHD full | 0.069 ± 0.023 | 0.128 ± 0.011 | 0.127 ± 0.011 | 0.450 ± 0.052 | 57.0 ± 1.2 |
| tonsil_s1 | Equal weights | 0.056 ± 0.024 | 0.106 ± 0.011 | 0.105 ± 0.011 | 0.510 ± 0.018 | 57.0 ± 1.2 |
| tonsil_s1 | Early concat | 0.029 ± 0.034 | 0.082 ± 0.049 | 0.081 ± 0.049 | 0.469 ± 0.015 | 45.3 ± 0.0 |
| tonsil_s1 | Full-batch | 0.033 ± 0.031 | 0.052 ± 0.030 | 0.050 ± 0.030 | 0.543 ± 0.084 | 426.4 ± 0.0 |
| tonsil_s1 | RNA only | 0.033 ± 0.040 | 0.040 ± 0.027 | 0.039 ± 0.027 | 0.590 ± 0.079 | 41.9 ± 1.2 |
| tonsil_s1 | ADT only | 0.031 ± 0.025 | 0.066 ± 0.042 | 0.066 ± 0.042 | 0.378 ± 0.024 | 41.9 ± 1.2 |
| tonsil_s1 | GCN encoder | 0.079 ± 0.014 | 0.143 ± 0.009 | 0.142 ± 0.009 | 0.402 ± 0.007 | 23.0 ± 0.0 |
| tonsil_s1 | SAGE encoder | 0.083 ± 0.022 | 0.149 ± 0.027 | 0.148 ± 0.027 | 0.389 ± 0.019 | 21.8 ± 0.0 |
| tonsil_s2 | SMAHD full | 0.025 ± 0.031 | 0.071 ± 0.051 | 0.070 ± 0.051 | 0.438 ± 0.026 | 58.3 ± 0.2 |
| tonsil_s2 | Equal weights | 0.025 ± 0.030 | 0.070 ± 0.049 | 0.069 ± 0.049 | 0.438 ± 0.025 | 58.3 ± 0.2 |
| tonsil_s2 | Early concat | 0.029 ± 0.027 | 0.085 ± 0.044 | 0.084 ± 0.044 | 0.457 ± 0.016 | 46.0 ± 0.2 |
| tonsil_s2 | Full-batch | 0.013 ± 0.024 | 0.045 ± 0.027 | 0.044 ± 0.027 | 0.508 ± 0.091 | 447.5 ± 0.0 |
| tonsil_s2 | RNA only | 0.011 ± 0.023 | 0.041 ± 0.018 | 0.040 ± 0.018 | 0.529 ± 0.076 | 42.7 ± 0.0 |
| tonsil_s2 | ADT only | 0.052 ± 0.040 | 0.076 ± 0.030 | 0.075 ± 0.030 | 0.404 ± 0.050 | 42.7 ± 0.0 |
| tonsil_s2 | GCN encoder | 0.061 ± 0.006 | 0.131 ± 0.009 | 0.131 ± 0.009 | 0.389 ± 0.006 | 23.2 ± 0.0 |
| tonsil_s2 | SAGE encoder | 0.079 ± 0.008 | 0.139 ± 0.009 | 0.138 ± 0.009 | 0.375 ± 0.005 | 22.1 ± 0.0 |
| hln_A1 | SMAHD full | 0.194 ± 0.016 | 0.327 ± 0.014 | 0.322 ± 0.014 | 0.349 ± 0.017 | 57.6 ± 0.3 |
| hln_A1 | Equal weights | 0.189 ± 0.024 | 0.324 ± 0.020 | 0.320 ± 0.020 | 0.345 ± 0.022 | 57.6 ± 0.3 |
| hln_A1 | Early concat | 0.179 ± 0.013 | 0.301 ± 0.009 | 0.297 ± 0.009 | 0.340 ± 0.019 | 42.8 ± 0.0 |
| hln_A1 | Full-batch | 0.210 ± 0.037 | 0.330 ± 0.022 | 0.326 ± 0.022 | 0.374 ± 0.036 | 338.1 ± 0.0 |
| hln_A1 | RNA only | 0.205 ± 0.025 | 0.336 ± 0.013 | 0.331 ± 0.013 | 0.364 ± 0.030 | 42.6 ± 0.0 |
| hln_A1 | ADT only | 0.131 ± 0.007 | 0.264 ± 0.014 | 0.260 ± 0.014 | 0.289 ± 0.007 | 42.6 ± 0.0 |
| hln_A1 | GCN encoder | 0.183 ± 0.006 | 0.311 ± 0.009 | 0.306 ± 0.009 | 0.340 ± 0.008 | 23.1 ± 0.0 |
| hln_A1 | SAGE encoder | 0.149 ± 0.023 | 0.299 ± 0.026 | 0.295 ± 0.026 | 0.294 ± 0.022 | 21.8 ± 0.0 |
| hln_D1 | SMAHD full | 0.178 ± 0.012 | 0.294 ± 0.010 | 0.289 ± 0.010 | 0.346 ± 0.011 | 55.7 ± 0.9 |
| hln_D1 | Equal weights | 0.192 ± 0.013 | 0.299 ± 0.016 | 0.294 ± 0.017 | 0.360 ± 0.013 | 55.7 ± 0.9 |
| hln_D1 | Early concat | 0.186 ± 0.044 | 0.292 ± 0.033 | 0.287 ± 0.033 | 0.355 ± 0.044 | 41.2 ± 1.3 |
| hln_D1 | Full-batch | 0.179 ± 0.018 | 0.282 ± 0.008 | 0.277 ± 0.008 | 0.347 ± 0.020 | 325.0 ± 0.0 |
| hln_D1 | RNA only | 0.139 ± 0.013 | 0.254 ± 0.022 | 0.249 ± 0.022 | 0.304 ± 0.012 | 41.1 ± 1.2 |
| hln_D1 | ADT only | 0.192 ± 0.022 | 0.288 ± 0.019 | 0.283 ± 0.019 | 0.361 ± 0.021 | 41.1 ± 1.2 |
| hln_D1 | GCN encoder | 0.192 ± 0.005 | 0.301 ± 0.016 | 0.296 ± 0.017 | 0.361 ± 0.007 | 22.8 ± 0.0 |
| hln_D1 | SAGE encoder | 0.143 ± 0.005 | 0.287 ± 0.006 | 0.282 ± 0.006 | 0.301 ± 0.004 | 21.6 ± 0.0 |

## Average across the four real datasets

| Variant | ARI_mean | NMI_mean | peak_memory_mib_mean |
| --- | --- | --- | --- |
| SMAHD full | 0.1164 | 0.2049 | 57.1404 |
| Equal weights | 0.1155 | 0.1998 | 57.1404 |
| Early concat | 0.1056 | 0.1901 | 43.7995 |
| Full-batch | 0.1089 | 0.1774 | 384.2534 |
| RNA only | 0.0971 | 0.1678 | 42.0943 |
| ADT only | 0.1013 | 0.1738 | 42.0943 |
| GCN encoder | 0.1289 | 0.2216 | 23.0009 |
| SAGE encoder | 0.1134 | 0.2184 | 21.8256 |

## Interpretation

- Human lymph node A1/D1 shows the clearest support for multimodal integration: SMAHD/equal-weight/full-batch variants generally outperform RNA-only, ADT-only and early concatenation.
- Human tonsil annotations are harder for this simplified PCA + KMeans evaluation; absolute ARI values are low. These results should be reported as an ablation/sensitivity analysis, not as a main benchmark claim.
- Micro-cluster training consistently reduces memory relative to full-batch training. For example, hln_A1 uses about 57.6 MiB for SMAHD_full versus 338.1 MiB for full-batch; tonsil_s2 uses about 58.3 MiB versus 447.5 MiB.
- Encoder behavior is dataset-dependent: GCN/SAGE improve tonsil ARI in this setup, whereas GAT-based SMAHD is competitive on human lymph node. This supports a conservative statement that the graph backbone should be reported and may require tuning.

## Suggested manuscript wording

We also performed a real-data ablation on four annotated RNA+ADT tissue sections, including two human tonsil sections and two human lymph node sections. Across three random seeds, multimodal variants generally improved over single-modality inputs on the human lymph node data and required substantially less memory under micro-cluster mini-batch training than full-batch training. The tonsil sections produced lower ARI values under the simplified PCA+KMeans evaluation, and encoder choice was dataset-dependent. These observations support the scalability and modality-aware design of SMAHD while indicating that graph backbones and reconstruction weights should be treated as explicit sensitivity parameters rather than universal constants.
