# Reproducibility notes

## Input requirements

The scripts require de-identified metadata, raw count matrices, RUVs factors and gene annotation files generated in the study workflow. Controlled-access data are not stored in this repository.

## Randomness

- R analyses use `set.seed(42)` where stochastic steps are present.
- Machine-learning analysis uses `--seed 42` by default and resets fold-level seeds for nested CV and SHAP computation.

## Clinical phenotypes

- MAG_PD motor coupling uses `MDS-UPDRS_part_iii_score`.
- VEL_PD primary uses `MDS-UPDRS_total_score`.
- VEL_PD sensitivity uses `MDS-UPDRS_part_iii_score`.
- Cognitive coupling uses `MoCA_score`.

## Main outputs

- MAG_PD: interaction tables and corrected expression matrices for UPDRS-III and MoCA coupling.
- VEL_PD: full model statistics, Tier 1/2/3 tables and master progression tables.
- Machine learning: nested-CV metrics, prediction tables, ROC/PR/calibration/DCA figures and SHAP summary tables.
