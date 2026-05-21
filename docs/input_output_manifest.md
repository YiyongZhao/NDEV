# Input and output manifest

## MAG_PD

Required inputs:
- Metadata table with sample ID, participant ID, diagnosis, time point, sex, age, brain proportion, ancestry PCs, `MDS-UPDRS_part_iii_score` and `MoCA_score`.
- Raw count matrix with genes as rows and samples as columns.
- RUVs factor table with samples as rows.

Main outputs:
- `Progression_Interaction_UPDRS/Final_Interaction_UPDRS_PD_specific.tsv`
- `Progression_Interaction_MoCA/Final_Interaction_MoCA_PD_specific.tsv`
- Corrected expression matrices for visualization.

## VEL_PD primary

Required inputs:
- Metadata table containing `MDS-UPDRS_total_score` and `MoCA_score`.
- Raw count matrix, RUVs factor table and GENCODE/Ensembl annotation table.

Main outputs:
- Full model statistics for time, clinical coupling and aging models.
- Tier 1, Tier 2 and Tier 3 VELOCITY gene tables.
- Master progression table.

## VEL_PD sensitivity

Required inputs:
- Metadata table containing `MDS-UPDRS_part_iii_score` and `MoCA_score`.
- Raw count matrix, RUVs factor table and GENCODE/Ensembl annotation table.

Main outputs:
- The same output classes as the primary VELOCITY analysis, using UPDRS-III as the motor phenotype.

## Machine learning

Required inputs:
- EV and whole-blood linear RNA count matrices.
- EV and whole-blood DEG tables.
- EV and whole-blood circRNA matrices and DEG tables.
- Sample matching table with diagnosis labels.
- GENCODE/Ensembl annotation table.

Main outputs:
- Nested-CV fold assignments and per-fold predictions.
- Performance metrics, ROC/PR curves, calibration and decision-curve figures.
- SHAP summary tables and figures.

## Feature-screening note

The machine-learning script treats DEG tables as external or pre-specified candidate feature sets. If candidate DEG tables are generated de novo from the same labels used in cross-validation, DEG screening should be performed within each outer training fold to avoid feature-selection leakage.
