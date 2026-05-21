# NDEV Parkinson's Disease RNA Analysis

Reproducible analysis code for the NDEV Parkinson's disease study by **Yiyong Zhao**.

This repository contains the final scripts used for four main analyses:

1. **MAG_PD**: PD-specific cross-sectional clinical coupling using `dream` mixed models.
2. **VEL_PD primary**: longitudinal VELOCITY analysis using MDS-UPDRS Total Score as the primary motor phenotype.
3. **VEL_PD sensitivity**: longitudinal VELOCITY analysis using MDS-UPDRS Part III as the sensitivity motor phenotype.
4. **Machine learning**: nested cross-validated PD classification using EV/WB linear RNA and circRNA features.

The controlled-access expression matrices and clinical metadata are not included. Scripts are path-parameterized so that the analyses can be rerun on the original study data or equivalent input files.

## Repository structure

```text
scripts/
  magnitude/
    01_MAG_PD_dream_interaction.R
  velocity/
    02_VEL_PD_primary_total_score.R
    03_VEL_PD_sensitivity_updrs_iii.R
  ml/
    04_machine_learning_nested_cv.py
config/
  input_paths.template.sh
docs/
  reproducibility.md
```

## Main analysis scripts

### 1. MAG_PD interaction analysis

Runs PD-specific expression-clinical coupling for motor and cognitive phenotypes:

```bash
Rscript scripts/magnitude/01_MAG_PD_dream_interaction.R \
  --meta /path/to/S0_meta_1109_EV_DEGs_short.tsv \
  --counts /path/to/1115_EMapper_readcounts.tsv \
  --ruvs /path/to/RUVs_W_factors_k3_all_samples.tsv \
  --out-dir results/MAG_PD \
  --cores 32
```

Key model:

```text
gene ~ Group + Score + Group:Score + Sex + BaselineAge + BrainProp + W1 + W2 + W3 + PC1 + PC2 + PC3 + (1|ParticipantID)
```

### 2. VEL_PD primary analysis

Runs the primary VELOCITY framework using MDS-UPDRS Total Score:

```bash
Rscript scripts/velocity/02_VEL_PD_primary_total_score.R \
  --meta /path/to/S0_meta_1109_EV_DEGs_short.tsv \
  --counts /path/to/1115_EMapper_readcounts.tsv \
  --ruvs /path/to/RUVs_W_factors_k3_all_samples.tsv \
  --gtf /path/to/GRCh38_Genecodev45_ENSG_ENST_Entrez.tsv \
  --out-dir results/VEL_PD_primary \
  --cores 32
```

### 3. VEL_PD sensitivity analysis

Runs the UPDRS-III sensitivity VELOCITY framework:

```bash
Rscript scripts/velocity/03_VEL_PD_sensitivity_updrs_iii.R \
  --meta /path/to/S0_meta_1109_EV_DEGs_short.tsv \
  --counts /path/to/1115_EMapper_readcounts.tsv \
  --ruvs /path/to/RUVs_W_factors_k3_all_samples.tsv \
  --gtf /path/to/GRCh38_Genecodev45_ENSG_ENST_Entrez.tsv \
  --out-dir results/VEL_PD_sensitivity_UPDRSIII \
  --cores 32
```

### 4. Machine-learning analysis

Runs nested cross-validation, model benchmarking, calibration, decision-curve summaries and SHAP interpretation:

```bash
python scripts/ml/04_machine_learning_nested_cv.py \
  --base-dir /path/to/ml_input_dir \
  --ev-circ-counts /path/to/EV_circRNA_BSJ.tsv \
  --ev-circ-deg /path/to/EV_circRNA_DEG.tsv \
  --wb-circ-counts /path/to/WB_circRNA_BSJ.tsv \
  --wb-circ-deg /path/to/WB_circRNA_DEG.tsv \
  --gene-map /path/to/GRCh38_Genecodev45_ENSG_ENST_Entrez.tsv \
  --output-dir results/ML_Final_Paper_v70_FixedSHAP \
  --n-jobs 32
```

By default, the script expects the following files under `--base-dir`:

```text
EV/1115_EMapper_readcounts.txt
EV/EV_EMapper_ENSG_brain_proportion_Final_DEGs_DREAM_PD_vs_Control.tsv
WB/1015samples_whole_blood_readcounts.tsv
WB/1014_WB_Neutrophil_Final_DEGs_DREAM_PD_vs_Control.tsv
WB_EV_match_ID.tsv
```

## Software

A conda environment template is provided:

```bash
conda env create -f environment.yml
conda activate ndev-analysis
```

R packages: `data.table`, `dplyr`, `tidyr`, `edgeR`, `limma`, `variancePartition`, `BiocParallel`, `lme4`.

Python packages: `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `joblib`; optional classifiers include `xgboost`, `lightgbm`, `catboost`, `interpret` and `pytorch-tabnet`.

## Notes

- The primary VELOCITY analysis uses **MDS-UPDRS Total Score**.
- The sensitivity VELOCITY analysis uses **MDS-UPDRS Part III**.
- MAG_PD uses **MDS-UPDRS Part III** for motor coupling and MoCA for cognitive coupling.
- All scripts use fixed random seeds where applicable.
- Output directories are intentionally excluded from version control.

## Citation

If you use this code, please cite the associated NDEV Parkinson's disease study.
