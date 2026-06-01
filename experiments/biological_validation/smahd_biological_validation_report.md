# SMAHD biological validation

This analysis adds marker-level biological support for the real-data results.

Mouse brain validation uses RNA expression of canonical layer and cell-type markers (Reln, Cux2, Rorb, Bcl11b, Fezf2, Tle4, Foxp2, Mbp, Plp1, Gad1, Gad2 and Aqp4) over SMAHD domains inferred from RNA+ATAC data.

Mouse spleen validation uses antibody-derived tags from Mouse_Spleen1. T-cell markers (CD3, CD4, CD8), B-cell/follicle markers (CD19, B220/CD45R, CD20, IgM, IgD), macrophage/red-pulp markers (F4/80, CD68, CD163) and vascular/stromal markers (CD31, CD105, MadCAM1) were summarized by SMAHD domain.

These marker analyses should be reported as biological plausibility checks rather than supervised validation, because the markers were not used for model training and the SMAHD domains were inferred in an unsupervised manner.
