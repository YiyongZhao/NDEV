#!/usr/bin/env Rscript
# ==========================================================================================
# Script: UPDRS-III sensitivity VELOCITY
# Target: Nature-Standard Longitudinal Progression Analysis for PD
# Focus:  100% Nominal P Filtering, True Timepoint Contrasts, LMER Bobyqa, + Interaction Output
# Update: V20 Inclusive Dual-Trajectory Logic
#         - Removed exclusionary BL/12M gates (BL_Not_Diverged, Flat12)
#         - Path 1: Continuous Linear (LMM slope + effect size at 24M)
#         - Path 2: NonLinear Expansion (24M > BL + MARGIN, no BL significance required)
#         - Dual_Evidence: genes satisfying both paths (strongest candidates)
# ==========================================================================================
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(edgeR)
  library(variancePartition)
  library(limma)
  library(BiocParallel)
  library(lme4)
})

# --- Input arguments ---
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  cfg <- list(meta = NULL, counts = NULL, ruvs = NULL, gtf = NULL, out_dir = "results/VEL_PD_sensitivity_UPDRSIII", cores = "8")
  if (length(args) %% 2 != 0) stop("Arguments must be supplied as --key value pairs.")
  if (length(args) > 0) {
    for (i in seq(1, length(args), by = 2)) {
      key <- sub("^--", "", args[i])
      key <- gsub("-", "_", key)
      if (!key %in% names(cfg)) stop(sprintf("Unknown argument: %s", args[i]))
      cfg[[key]] <- args[i + 1]
    }
  }
  required <- c("meta", "counts", "ruvs", "gtf")
  missing <- required[vapply(cfg[required], is.null, logical(1))]
  if (length(missing) > 0) {
    stop("Missing required arguments: ", paste(missing, collapse = ", "))
  }
  cfg$cores <- as.integer(cfg$cores)
  cfg
}

cfg <- parse_args()
META_FILE <- cfg$meta
COUNTS_FILE <- cfg$counts
RUVS_FILE <- cfg$ruvs
GTF_FILE <- cfg$gtf
OUT_DIR <- cfg$out_dir
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

NUM_CORES <- min(cfg$cores, parallel::detectCores())
param <- SnowParam(workers = NUM_CORES, type = "SOCK", progressbar = TRUE)
register(param)


# === SCIENTIFIC THRESHOLDS ===
P_THRESH    <- 0.05   # Universal nominal P threshold
LOGFC_THRESH <- 0.1   # Minimum biological effect size (EV-sensitive)
MARGIN      <- 0.03   # Effect expansion tolerance
EPS         <- 0.01   # Noise floor for directional signs

cat("==========================================================\n")
cat(" Running PD Progression Analysis (V20 Dual-Trajectory)\n")
cat(sprintf(" Threshold: Nominal P < %.2f | MinFC: %.2f | Margin: %.2f\n",
            P_THRESH, LOGFC_THRESH, MARGIN))
cat("==========================================================\n")

# --- Helper ---
strip_ver <- function(x) sub("\\..*$", "", x)
safe_sign <- function(x) ifelse(abs(x) < EPS, 0, sign(x))

# --- [2] Load Gene Mapping ---
cat(">> Loading ID Mapping...\n")
gtf <- fread(GTF_FILE, sep="\t", header=TRUE)
gtf_map <- gtf %>%
  select(ENSEMBL_ID, SYMBOL_Name) %>%
  mutate(Ensembl_Gene_ID = strip_ver(ENSEMBL_ID)) %>%
  distinct() %>%
  group_by(Ensembl_Gene_ID) %>% slice(1) %>% ungroup() %>%
  mutate(Gene_Symbol = ifelse(!is.na(SYMBOL_Name) & SYMBOL_Name != "",
                              gsub("_ENSG.*", "", SYMBOL_Name), Ensembl_Gene_ID)) %>%
  select(Ensembl_Gene_ID, Gene_Symbol)
stopifnot(!anyDuplicated(gtf_map$Ensembl_Gene_ID))

save_full_stats <- function(fit, coef_name, file_name) {
  if (!coef_name %in% colnames(fit$coefficients))
    stop(sprintf("FATAL: Coef '%s' not found in model matrix!", coef_name))
  tt <- topTable(fit, coef=coef_name, number=Inf, sort.by="none")
  tt$Ensembl_Gene_ID <- strip_ver(rownames(tt))
  tt <- tt %>%
    rename(Log2_Fold_Change=logFC, Average_Expression=AveExpr,
           T_Statistic=t, Nominal_P_Value=P.Value,
           FDR_Adjusted_P_Value=adj.P.Val, B_Statistic=B) %>%
    left_join(gtf_map, by="Ensembl_Gene_ID") %>%
    select(Ensembl_Gene_ID, Gene_Symbol, everything())
  fwrite(tt, file.path(OUT_DIR, file_name), sep="\t")
  return(tt)
}

# --- [3] Data Loading & Rigorous Cleaning ---
cat(">> Loading Metadata...\n")
meta_all <- fread(META_FILE, sep="\t", header=TRUE)
colnames(meta_all) <- trimws(colnames(meta_all))

required_cols <- c("TGen_EV_sample_ID","Participant_ID","Diagnosis","Time_point",
                   "Sex","Age","Brain_proportion",
                   "Ancestry_PC1","Ancestry_PC2","Ancestry_PC3")
stopifnot(all(required_cols %in% colnames(meta_all)))

meta <- meta_all %>%
  filter(Diagnosis %in% c("No Neurological Diagnosis","Parkinson's Disease")) %>%
  mutate(
    SampleID      = trimws(as.character(TGen_EV_sample_ID)),
    ParticipantID = trimws(as.character(Participant_ID)),
    Group         = factor(ifelse(Diagnosis=="Parkinson's Disease","PD","Ctrl"),
                           levels=c("Ctrl","PD")),
    TimePoint     = factor(trimws(Time_point), levels=c("BL","12M","24M")),
    Sex           = factor(Sex),
    Age_Raw       = as.numeric(Age),
    Brain_Prop    = as.numeric(Brain_proportion),
    UPDRS         = as.numeric(`MDS-UPDRS_part_iii_score`),
    MoCA          = as.numeric(`MoCA_score`),
    PC1           = as.numeric(Ancestry_PC1),
    PC2           = as.numeric(Ancestry_PC2),
    PC3           = as.numeric(Ancestry_PC3)
  )

meta$TimeNum <- case_when(
  as.character(meta$TimePoint)=="BL"  ~ 0,
  as.character(meta$TimePoint)=="12M" ~ 12,
  as.character(meta$TimePoint)=="24M" ~ 24)
meta <- meta[!is.na(meta$TimeNum), ]

# Baseline Age (subject-specific offset)
meta <- meta %>%
  group_by(ParticipantID) %>%
  mutate(Baseline_Age = Age_Raw[which.min(TimeNum)] - (min(TimeNum)/12)) %>%
  ungroup()

# Group-aware imputation
meta <- meta %>%
  group_by(Group) %>%
  mutate(across(c(Brain_Prop, Baseline_Age, PC1, PC2, PC3),
                ~ifelse(is.na(.), mean(., na.rm=TRUE), .))) %>%
  ungroup()
for (col in c("Brain_Prop","Baseline_Age","PC1","PC2","PC3"))
  meta[[col]][is.na(meta[[col]])] <- mean(meta[[col]], na.rm=TRUE)

# RUV factors
W_df <- read.table(RUVS_FILE, header=TRUE, row.names=1, sep="\t")
colnames(W_df) <- paste0("W", 1:ncol(W_df))
w_cols <- grep("^W\\d+$", colnames(W_df), value=TRUE)
W_df$SampleID <- rownames(W_df)
meta <- meta %>% left_join(W_df, by="SampleID")
if (any(is.na(meta$W1))) stop("FATAL: Missing RUV W factors after join!")

# Scale covariates
cols_scale <- c("Baseline_Age","Brain_Prop","PC1","PC2","PC3")
for (col in cols_scale)
  meta[[paste0(col,"_Sc_All")]] <- as.numeric(scale(meta[[col]]))

# Counts
cat(">> Loading Counts...\n")
counts_dt <- fread(COUNTS_FILE, sep="\t")
raw_genes <- strip_ver(counts_dt[[1]])
counts_mat_raw <- as.matrix(counts_dt[,-1,with=FALSE])
mode(counts_mat_raw) <- "numeric"
if (anyDuplicated(raw_genes)) {
  cat("  INFO: Resolving duplicated ENSEMBL IDs via rowsum...\n")
  counts_mat_raw <- rowsum(counts_mat_raw, group=raw_genes)
  raw_genes <- rownames(counts_mat_raw)
} else {
  rownames(counts_mat_raw) <- raw_genes
}

common_samples <- intersect(meta$SampleID, colnames(counts_mat_raw))
meta <- meta %>% filter(SampleID %in% common_samples) %>%
        arrange(match(SampleID, common_samples))
counts_mat_all <- counts_mat_raw[, meta$SampleID]
stopifnot(all(meta$SampleID == colnames(counts_mat_all)))
cat(sprintf("  Final Master Matrix: %d Genes, %d Samples\n",
            nrow(counts_mat_all), ncol(counts_mat_all)))

isexpr <- rowSums(cpm(counts_mat_all) > 0.5) >= (0.15 * ncol(counts_mat_all))
dge_all <- calcNormFactors(DGEList(counts=counts_mat_all[isexpr,]), method="TMM")

# --- [4] Robust Dream Runner ---
run_dream_robust <- function(formula_str, metadata, dge_obj, prefix,
                              formula_fallback=NULL) {
  cat(sprintf(">> Running %s...\n", prefix))
  form <- as.formula(formula_str)
  if (is.null(formula_fallback))
    formula_fallback <- sub("\\(.*?\\|ParticipantID\\)", "(1|ParticipantID)", formula_str)
  res <- tryCatch({
    vobj <- voomWithDreamWeights(dge_obj, form, metadata, BPPARAM=param)
    fit  <- dream(vobj, form, metadata, BPPARAM=param)
    list(fit=eBayes(fit), vobj=vobj)
  }, error = function(e) {
    cat(sprintf("  WARNING: Convergence issue in %s. Applying fallback...\n", prefix))
    form_fb  <- as.formula(formula_fallback)
    vobj_fb  <- voomWithDreamWeights(dge_obj, form_fb, metadata, BPPARAM=param)
    fit_fb   <- dream(vobj_fb, form_fb, metadata, BPPARAM=param)
    list(fit=eBayes(fit_fb), vobj=vobj_fb)
  })
  return(res)
}

w_formula_part <- paste(w_cols, collapse=" + ")
NUISANCE_AGING <- paste0("+ Sex + Brain_Prop_Sc_All + PC1_Sc_All + PC2_Sc_All + PC3_Sc_All + ", w_formula_part)
NUISANCE_MAIN  <- paste0("+ Sex + Baseline_Age_Sc_All + Brain_Prop_Sc_All + PC1_Sc_All + PC2_Sc_All + PC3_Sc_All + ", w_formula_part)

# --- [5] Model 0: Aging Exclusion ---
cat("--- Model 0: Aging in Controls ---\n")
meta_ctrl <- meta %>% filter(Group=="Ctrl") %>%
             arrange(match(SampleID, colnames(counts_mat_all)))
counts_ctrl  <- counts_mat_all[, meta_ctrl$SampleID]
isexpr_ctrl  <- rowSums(cpm(counts_ctrl) > 0.5) >= (0.15 * ncol(counts_ctrl))
dge_ctrl     <- calcNormFactors(DGEList(counts=counts_ctrl[isexpr_ctrl,]), method="TMM")
meta_ctrl    <- meta_ctrl %>% arrange(match(SampleID, colnames(dge_ctrl$counts)))
stopifnot(all(meta_ctrl$SampleID == colnames(dge_ctrl$counts)))
meta_ctrl$Age_Current_Sc <- as.numeric(scale(meta_ctrl$Age_Raw))
meta_ctrl$TimeNum_Sc     <- as.numeric(scale(meta_ctrl$TimeNum))

res_age  <- run_dream_robust(paste0("~ Age_Current_Sc ",  NUISANCE_AGING, " + (1|ParticipantID)"), meta_ctrl, dge_ctrl, "Model_0A_Aging")
tt_age   <- save_full_stats(res_age$fit,  "Age_Current_Sc", "Stats_Model_0A_Aging_by_Age.tsv")
res_time <- run_dream_robust(paste0("~ TimeNum_Sc ", NUISANCE_AGING, " + (1|ParticipantID)"), meta_ctrl, dge_ctrl, "Model_0B_Time")
tt_time  <- save_full_stats(res_time$fit, "TimeNum_Sc",     "Stats_Model_0B_Aging_by_Time.tsv")

aging_merge <- tt_age %>%
  select(Ensembl_Gene_ID, Age_P=Nominal_P_Value, Age_FC=Log2_Fold_Change) %>%
  inner_join(tt_time %>% select(Ensembl_Gene_ID, Time_P=Nominal_P_Value, Time_FC=Log2_Fold_Change),
             by="Ensembl_Gene_ID")

aging_genes <- aging_merge$Ensembl_Gene_ID[
  (aging_merge$Age_P < P_THRESH | aging_merge$Time_P < P_THRESH) &
  (safe_sign(aging_merge$Age_FC) == safe_sign(aging_merge$Time_FC)) &
  (safe_sign(aging_merge$Age_FC) != 0) &
  (pmax(abs(aging_merge$Age_FC), abs(aging_merge$Time_FC), na.rm=TRUE) >= LOGFC_THRESH)
]
cat(sprintf("  Tagged %d Validated Aging Genes.\n", length(aging_genes)))

# --- [6] Model A: Factor Timepoint Contrasts ---
cat("--- Model A: Factor-Time (True Absolute Gaps) ---\n")
form_factor <- paste0("~ Group * TimePoint ", NUISANCE_MAIN, " + (1|ParticipantID)")
res_factor  <- run_dream_robust(form_factor, meta, dge_all, "Model_A_Factor_Timepoints")

L_time <- makeContrastsDream(as.formula(form_factor), meta, contrasts=c(
  PD_vs_Ctrl_BL   = "GroupPD",
  PD_vs_Ctrl_12M  = "GroupPD + GroupPD:TimePoint12M",
  PD_vs_Ctrl_24M  = "GroupPD + GroupPD:TimePoint24M",
  Progression_Accel = "GroupPD:TimePoint24M - GroupPD:TimePoint12M"
))
fit_time_contrasts <- eBayes(dream(res_factor$vobj, as.formula(form_factor),
                                   meta, L=L_time, BPPARAM=param))
tt_bl   <- save_full_stats(fit_time_contrasts, "PD_vs_Ctrl_BL",    "Stats_Model_A_AbsoluteGap_BL.tsv")
tt_12m  <- save_full_stats(fit_time_contrasts, "PD_vs_Ctrl_12M",   "Stats_Model_A_AbsoluteGap_12M.tsv")
tt_24m  <- save_full_stats(fit_time_contrasts, "PD_vs_Ctrl_24M",   "Stats_Model_A_AbsoluteGap_24M.tsv")
tt_accel <- save_full_stats(fit_time_contrasts, "Progression_Accel","Stats_Model_A_Acceleration_Contrast.tsv")

factor_merged <- tt_bl %>%
  select(Ensembl_Gene_ID, FC_BL=Log2_Fold_Change, P_BL=Nominal_P_Value) %>%
  left_join(tt_12m  %>% select(Ensembl_Gene_ID, FC_12M=Log2_Fold_Change, P_12M=Nominal_P_Value), by="Ensembl_Gene_ID") %>%
  left_join(tt_24m  %>% select(Ensembl_Gene_ID, FC_24M=Log2_Fold_Change, P_24M=Nominal_P_Value, FDR_24M=FDR_Adjusted_P_Value), by="Ensembl_Gene_ID") %>%
  left_join(tt_accel %>% select(Ensembl_Gene_ID, FC_Accel=Log2_Fold_Change, P_Accel=Nominal_P_Value), by="Ensembl_Gene_ID")

# --- [7] Model B: Continuous Trend ---
cat("--- Model B: Random Slopes (Continuous Trend) ---\n")
meta$TimeNum_Sc_All <- as.numeric(scale(meta$TimeNum))
form_cont <- paste0("~ Group * TimeNum_Sc_All ", NUISANCE_MAIN,
                    " + (1 + TimeNum_Sc_All|ParticipantID)")
res_cont  <- run_dream_robust(form_cont, meta, dge_all, "Model_B_Continuous")
tt_slope  <- save_full_stats(res_cont$fit, "GroupPD:TimeNum_Sc_All",
                              "Stats_Model_B_Continuous_Trend.tsv")
res_slope <- tt_slope %>%
  select(Ensembl_Gene_ID, P_Trend=Nominal_P_Value,
         FDR_Trend=FDR_Adjusted_P_Value, LogFC_Trend=Log2_Fold_Change)

# --- [8] Model C: PD-Only Clinical Associations ---
cat("--- Model C: PD-Only Clinical Models (True Independent TMM) ---\n")
meta_pd <- meta %>% filter(Group=="PD") %>%
           arrange(match(SampleID, colnames(counts_mat_all)))
counts_pd   <- counts_mat_all[, meta_pd$SampleID]
isexpr_pd   <- rowSums(cpm(counts_pd) > 0.5) >= (0.15 * ncol(counts_pd))
dge_pd      <- calcNormFactors(DGEList(counts=counts_pd[isexpr_pd,]), method="TMM")
meta_pd$TimeNum_Sc_PD <- as.numeric(scale(meta_pd$TimeNum))
for (col in cols_scale)
  meta_pd[[paste0(col,"_Sc_PD")]] <- as.numeric(scale(meta_pd[[col]]))

NUISANCE_PD <- paste0("+ Sex + Baseline_Age_Sc_PD + Brain_Prop_Sc_PD + PC1_Sc_PD + PC2_Sc_PD + PC3_Sc_PD + ", w_formula_part)

# C1. UPDRS
meta_updrs <- meta_pd %>% filter(!is.na(UPDRS)) %>%
              arrange(match(SampleID, colnames(dge_pd$counts)))
dge_updrs  <- dge_pd[, meta_updrs$SampleID]
stopifnot(all(meta_updrs$SampleID == colnames(dge_updrs$counts)))
meta_updrs$UPDRS_Sc <- as.numeric(scale(meta_updrs$UPDRS))
res_updrs  <- run_dream_robust(
  paste0("~ UPDRS_Sc + TimeNum_Sc_PD ", NUISANCE_PD, " + (1|ParticipantID)"),
  meta_updrs, dge_updrs, "Model_C_UPDRS")
tt_updrs   <- save_full_stats(res_updrs$fit, "UPDRS_Sc", "Stats_Model_C_UPDRS_PD_Only.tsv")
df_updrs   <- tt_updrs %>% select(Ensembl_Gene_ID, P_UPDRS=Nominal_P_Value, FDR_UPDRS=FDR_Adjusted_P_Value)

# C2. MoCA
meta_moca <- meta_pd %>% filter(!is.na(MoCA)) %>%
             arrange(match(SampleID, colnames(dge_pd$counts)))
dge_moca  <- dge_pd[, meta_moca$SampleID]
stopifnot(all(meta_moca$SampleID == colnames(dge_moca$counts)))
meta_moca$MoCA_Sc <- as.numeric(scale(meta_moca$MoCA))
res_moca  <- run_dream_robust(
  paste0("~ MoCA_Sc + TimeNum_Sc_PD ", NUISANCE_PD, " + (1|ParticipantID)"),
  meta_moca, dge_moca, "Model_C_MoCA")
tt_moca   <- save_full_stats(res_moca$fit, "MoCA_Sc", "Stats_Model_C_MoCA_PD_Only.tsv")
df_moca   <- tt_moca %>% select(Ensembl_Gene_ID, P_MoCA=Nominal_P_Value, FDR_MoCA=FDR_Adjusted_P_Value)

# --- [9] Supplementary: LME4 Time-Residualized Models ---
cat("--- Supp Model: LME4 Strict Time-Residualized Scores ---\n")
lmer_ctrl <- lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000))
NUISANCE_PD_NO_TIME <- paste0("+ Sex + Baseline_Age_Sc_PD + Brain_Prop_Sc_PD + PC1_Sc_PD + PC2_Sc_PD + PC3_Sc_PD + ", w_formula_part)

meta_updrs$TimeYear <- meta_updrs$TimeNum / 12
mod_updrs_resid <- tryCatch(
  lmer(UPDRS ~ TimeYear + (TimeYear|ParticipantID), data=meta_updrs,
       control=lmer_ctrl, na.action=na.exclude),
  warning = function(w) lmer(UPDRS ~ TimeYear + (1|ParticipantID),
                              data=meta_updrs, control=lmer_ctrl, na.action=na.exclude),
  error   = function(e) lmer(UPDRS ~ TimeYear + (1|ParticipantID),
                              data=meta_updrs, control=lmer_ctrl, na.action=na.exclude)
)
meta_updrs$UPDRS_Resid_Sc <- as.numeric(scale(residuals(mod_updrs_resid)))
res_resid_updrs <- run_dream_robust(
  paste0("~ UPDRS_Resid_Sc ", NUISANCE_PD_NO_TIME, " + (1|ParticipantID)"),
  meta_updrs, dge_updrs, "Supp_Model_UPDRS_Resid")
tt_resid_updrs <- save_full_stats(res_resid_updrs$fit, "UPDRS_Resid_Sc",
                                   "Stats_Model_Supp_UPDRS_Residualized.tsv")

meta_moca$TimeYear <- meta_moca$TimeNum / 12
mod_moca_resid <- tryCatch(
  lmer(MoCA ~ TimeYear + (TimeYear|ParticipantID), data=meta_moca,
       control=lmer_ctrl, na.action=na.exclude),
  warning = function(w) lmer(MoCA ~ TimeYear + (1|ParticipantID),
                              data=meta_moca, control=lmer_ctrl, na.action=na.exclude),
  error   = function(e) lmer(MoCA ~ TimeYear + (1|ParticipantID),
                              data=meta_moca, control=lmer_ctrl, na.action=na.exclude)
)
meta_moca$MoCA_Resid_Sc <- as.numeric(scale(residuals(mod_moca_resid)))
res_resid_moca <- run_dream_robust(
  paste0("~ MoCA_Resid_Sc ", NUISANCE_PD_NO_TIME, " + (1|ParticipantID)"),
  meta_moca, dge_moca, "Supp_Model_MoCA_Resid")
tt_resid_moca <- save_full_stats(res_resid_moca$fit, "MoCA_Resid_Sc",
                                  "Stats_Model_Supp_MoCA_Residualized.tsv")

resid_merged <- tt_resid_updrs %>%
  select(Ensembl_Gene_ID, P_Resid_UPDRS=Nominal_P_Value) %>%
  left_join(tt_resid_moca %>% select(Ensembl_Gene_ID, P_Resid_MoCA=Nominal_P_Value),
            by="Ensembl_Gene_ID")

# --- [9.5] Supp: Full Sample Clinical Interaction (visualization only) ---
cat("--- Supp Model: Full Sample Clinical Interactions ---\n")

meta_updrs_all <- meta %>% filter(!is.na(UPDRS)) %>%
                  arrange(match(SampleID, colnames(counts_mat_all)))
counts_updrs_all <- counts_mat_all[, meta_updrs_all$SampleID]
isexpr_updrs_all <- rowSums(cpm(counts_updrs_all) > 0.5) >= (0.15 * ncol(counts_updrs_all))
dge_updrs_all    <- calcNormFactors(DGEList(counts=counts_updrs_all[isexpr_updrs_all,]), method="TMM")
stopifnot(all(meta_updrs_all$SampleID == colnames(dge_updrs_all$counts)))
meta_updrs_all$UPDRS_Sc <- as.numeric(scale(meta_updrs_all$UPDRS))
res_int_updrs <- run_dream_robust(
  paste0("~ Group * UPDRS_Sc + TimeNum_Sc_All ", NUISANCE_MAIN, " + (1|ParticipantID)"),
  meta_updrs_all, dge_updrs_all, "Supp_Int_UPDRS")
invisible(save_full_stats(res_int_updrs$fit, "GroupPD:UPDRS_Sc",
                           "Stats_Model_Supp_UPDRS_Interaction_FullSample.tsv"))

meta_moca_all <- meta %>% filter(!is.na(MoCA)) %>%
                 arrange(match(SampleID, colnames(counts_mat_all)))
counts_moca_all <- counts_mat_all[, meta_moca_all$SampleID]
isexpr_moca_all <- rowSums(cpm(counts_moca_all) > 0.5) >= (0.15 * ncol(counts_moca_all))
dge_moca_all    <- calcNormFactors(DGEList(counts=counts_moca_all[isexpr_moca_all,]), method="TMM")
stopifnot(all(meta_moca_all$SampleID == colnames(dge_moca_all$counts)))
meta_moca_all$MoCA_Sc <- as.numeric(scale(meta_moca_all$MoCA))
res_int_moca <- run_dream_robust(
  paste0("~ Group * MoCA_Sc + TimeNum_Sc_All ", NUISANCE_MAIN, " + (1|ParticipantID)"),
  meta_moca_all, dge_moca_all, "Supp_Int_MoCA")
invisible(save_full_stats(res_int_moca$fit, "GroupPD:MoCA_Sc",
                           "Stats_Model_Supp_MoCA_Interaction_FullSample.tsv"))
cat("  Successfully extracted Diagnosis x Clinical Score interaction effects.\n")

# --- [10] Integration & Tiered Filtering (V20 Inclusive Dual-Trajectory Logic) ---
cat(">> Integrating and Applying Scientific Tiers (V20 Dual-Trajectory)...\n")

final_df <- factor_merged %>%
  left_join(res_slope,   by="Ensembl_Gene_ID") %>%
  left_join(df_updrs,    by="Ensembl_Gene_ID") %>%
  left_join(df_moca,     by="Ensembl_Gene_ID") %>%
  left_join(resid_merged,by="Ensembl_Gene_ID") %>%
  left_join(gtf_map,     by="Ensembl_Gene_ID") %>%
  mutate(
    Is_Normal_Aging_Gene = (Ensembl_Gene_ID %in% aging_genes),

    # Fundamental 24M endpoint gate (used by both paths)
    Is_Sig_Prog_24M = (!is.na(P_24M) & P_24M < P_THRESH & abs(FC_24M) >= LOGFC_THRESH),

    # -------------------------------------------------------------------------
    # PATH 1: Continuous Linear Progression
    # LMM trend is significant, direction consistent with 24M, and 24M effect
    # size meets minimum biological threshold (prevents tiny-effect inclusions)
    # -------------------------------------------------------------------------
    Is_Continuous_Linear = (
      !is.na(P_Trend) & P_Trend < P_THRESH &
      safe_sign(LogFC_Trend) == safe_sign(FC_24M) &
      abs(FC_24M) >= LOGFC_THRESH                    # minimum effect at endpoint
    ),

    # -------------------------------------------------------------------------
    # PATH 2: Non-linear Effect Expansion
    # 24M gap is significant AND larger than BL gap by MARGIN.
    # No requirement on BL significance - captures genes already dysregulated
    # at baseline in PDBP's heterogeneous-stage cohort ("from having to more").
    # Direction: FC_24M and (FC_24M - FC_BL) must agree (not a flip artefact).
    # -------------------------------------------------------------------------
    Is_NonLinear_Expansion = (
      Is_Sig_Prog_24M &
      abs(FC_24M) > abs(FC_BL) + MARGIN &
      (safe_sign(FC_24M) == safe_sign(FC_24M - FC_BL) | safe_sign(FC_BL) == 0)
    ),

    # -------------------------------------------------------------------------
    # TIER 1: Unified Progression Type
    # Dual_Evidence genes (both paths) are the highest-confidence candidates.
    # -------------------------------------------------------------------------
    Progression_Type = case_when(
      Is_Continuous_Linear & Is_NonLinear_Expansion ~ "Dual_Evidence",
      Is_Continuous_Linear                          ~ "Continuous_Linear",
      Is_NonLinear_Expansion                        ~ "NonLinear_Expansion",
      TRUE                                          ~ NA_character_
    ),
    Tier_1_Time_Progression = !is.na(Progression_Type),

    # -------------------------------------------------------------------------
    # TIER 2: Clinical Coupling (PD-only, time-controlled, independent TMM)
    # -------------------------------------------------------------------------
    Is_Clinically_Coupled = (
      (!is.na(P_UPDRS) & P_UPDRS < P_THRESH) |
      (!is.na(P_MoCA)  & P_MoCA  < P_THRESH)
    ),
    Tier_2_Clinically_Coupled = Tier_1_Time_Progression & Is_Clinically_Coupled,

    # -------------------------------------------------------------------------
    # TIER 3: Platinum Robust - exclude aging confounds + residualized validation
    # -------------------------------------------------------------------------
    Is_Resid_Robust = (
      (!is.na(P_Resid_UPDRS) & P_Resid_UPDRS < P_THRESH) |
      (!is.na(P_Resid_MoCA)  & P_Resid_MoCA  < P_THRESH)
    ),
    Tier_3_Platinum_Robust = Tier_2_Clinically_Coupled & !Is_Normal_Aging_Gene & Is_Resid_Robust
  ) %>%
  select(
    Ensembl_Gene_ID, Gene_Symbol, Progression_Type,
    Log2FC_Progression_24M   = FC_24M,
    Nominal_P_Progression_24M= P_24M,
    FDR_Progression_24M      = FDR_24M,
    Log2FC_Continuous_Trend  = LogFC_Trend,
    Nominal_P_Continuous_Trend= P_Trend,
    FDR_Continuous_Trend     = FDR_Trend,
    Nominal_P_UPDRS_PD_Only  = P_UPDRS,
    FDR_UPDRS_PD_Only        = FDR_UPDRS,
    Nominal_P_MoCA_PD_Only   = P_MoCA,
    FDR_MoCA_PD_Only         = FDR_MoCA,
    Nominal_P_UPDRS_Residualized = P_Resid_UPDRS,
    Nominal_P_MoCA_Residualized  = P_Resid_MoCA,
    everything()
  )

fwrite(final_df, file.path(OUT_DIR, "Master_Ultimate_Progression_Table.tsv"), sep="\t")
fwrite(final_df %>% filter(Tier_1_Time_Progression) %>% arrange(Nominal_P_Progression_24M),
       file.path(OUT_DIR, "List_Tier1_Time_Progression.tsv"), sep="\t")
fwrite(final_df %>% filter(Tier_2_Clinically_Coupled) %>% arrange(Nominal_P_Progression_24M),
       file.path(OUT_DIR, "List_Tier2_Clinically_Coupled.tsv"), sep="\t")
fwrite(final_df %>% filter(Tier_3_Platinum_Robust) %>% arrange(Nominal_P_Progression_24M),
       file.path(OUT_DIR, "List_Tier3_Platinum_Robust.tsv"), sep="\t")

# Tier 1 breakdown by Progression_Type
type_counts <- final_df %>%
  filter(Tier_1_Time_Progression) %>%
  count(Progression_Type, sort=TRUE)

capture.output(sessionInfo(), file=file.path(OUT_DIR, "sessionInfo.txt"))

cat(sprintf("\n>> PIPELINE SUMMARY (V20 Dual-Trajectory):\n"))
cat(sprintf("   Aging genes excluded:          %d\n", length(aging_genes)))
cat(sprintf("   Tier 1 (Time Progression):     %d genes\n", sum(final_df$Tier_1_Time_Progression, na.rm=TRUE)))
for (i in seq_len(nrow(type_counts)))
  cat(sprintf("     %-25s %d\n", paste0(type_counts$Progression_Type[i],":"), type_counts$n[i]))
cat(sprintf("   Tier 2 (Clinically Coupled):   %d genes\n", sum(final_df$Tier_2_Clinically_Coupled, na.rm=TRUE)))
cat(sprintf("   Tier 3 (Platinum Robust):      %d genes\n", sum(final_df$Tier_3_Platinum_Robust, na.rm=TRUE)))
cat(">> Execution Complete. All tables saved in:", OUT_DIR, "\n")
