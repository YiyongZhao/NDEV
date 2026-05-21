#!/usr/bin/env Rscript

# Run MAG_PD interaction analysis for motor and cognitive phenotypes.
# Example:
# Rscript scripts/magnitude/01_MAG_PD_dream_interaction.R \
#   --meta data/S0_meta_1109_EV_DEGs_short.tsv \
#   --counts data/1115_EMapper_readcounts.tsv \
#   --ruvs data/RUVs_W_factors_k3_all_samples.tsv \
#   --out-dir results/MAG_PD --cores 32

set.seed(42)

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

cat("=== STEP 3: dream Progression Interaction Analysis (Final Optimized) ===\n")
cat("Targets: 1. UPDRS-III (Motor), 2. MoCA (Cognitive)\n")
cat("Model: Interaction (Diagnosis * Score) with Random Effects (1|ID)\n")
cat(sprintf("Start: %s\n", Sys.time()))

# --- Input arguments ---
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  cfg <- list(meta = NULL, counts = NULL, ruvs = NULL, out_dir = "results/MAG_PD", cores = "8")

  if (length(args) == 3 && !grepl("^--", args[1])) {
    cfg$meta <- args[1]
    cfg$counts <- args[2]
    cfg$cores <- args[3]
    cfg$ruvs <- "../../RUVs_PCA_output/RUVs_W_factors_k3_all_samples.tsv"
  } else {
    if (length(args) %% 2 != 0) stop("Arguments must be supplied as --key value pairs.")
    for (i in seq(1, length(args), by = 2)) {
      key <- sub("^--", "", args[i])
      key <- gsub("-", "_", key)
      if (!key %in% names(cfg)) stop(sprintf("Unknown argument: %s", args[i]))
      cfg[[key]] <- args[i + 1]
    }
  }

  required <- c("meta", "counts", "ruvs")
  missing <- required[vapply(cfg[required], is.null, logical(1))]
  if (length(missing) > 0) {
    stop("Missing required arguments: ", paste(missing, collapse = ", "))
  }
  cfg$cores <- as.integer(cfg$cores)
  cfg
}

cfg <- parse_args()
meta_file <- cfg$meta
counts_file <- cfg$counts
ruvs_w_file <- cfg$ruvs
num_cores <- min(cfg$cores, parallel::detectCores())
out_root <- cfg$out_dir

cat(sprintf("Using %d cores.\n", num_cores))
dir.create(out_root, showWarnings = FALSE, recursive = TRUE)
out_dir_updrs <- file.path(out_root, "Progression_Interaction_UPDRS")
out_dir_moca  <- file.path(out_root, "Progression_Interaction_MoCA")
dir.create(out_dir_updrs, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir_moca, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(ruvs_w_file)) stop("RUVs W factor file not found: ", ruvs_w_file)

# --- [1] Process Metadata (Common Step) ---
cat("\n[1] Processing Metadata...\n")
meta_all <- fread(meta_file, sep = "\t", header = TRUE, nThread = num_cores)
colnames(meta_all) <- trimws(colnames(meta_all))

# Filter PD + Control
meta_run_base <- meta_all %>%
  filter(Diagnosis %in% c("No Neurological Diagnosis", "Parkinson's Disease")) %>%
  mutate(
    SampleID = trimws(as.character(TGen_EV_sample_ID)),
    ParticipantID = trimws(as.character(Participant_ID)),
    TimePoint = factor(Time_point, levels = c("BL", "12M", "24M")),
    Sex = factor(Sex),
    # Define Factor: Control is Reference level
    Group = factor(ifelse(Diagnosis == "Parkinson's Disease", "PD", "Ctrl"), levels = c("Ctrl", "PD")),
    Age_Raw = as.numeric(Age),
    # Scores
    UPDRS_Raw = as.numeric(`MDS-UPDRS_part_iii_score`),
    MoCA_Raw = as.numeric(`MoCA_score`),
    # Covariates
    Ancestry_PC1 = as.numeric(Ancestry_PC1),
    Ancestry_PC2 = as.numeric(Ancestry_PC2),
    Ancestry_PC3 = as.numeric(Ancestry_PC3),
    Brain_proportion = as.numeric(Brain_proportion)
  )

# --- Hierarchical Baseline Age Logic ---
# Ensure Age is fixed per subject (BL > 12M > 24M)
meta_run_base <- meta_run_base %>%
  group_by(ParticipantID) %>%
  mutate(
    Baseline_Age = case_when(
      any(TimePoint == "BL" & !is.na(Age_Raw)) ~ Age_Raw[TimePoint == "BL" & !is.na(Age_Raw)][1],
      any(TimePoint == "12M" & !is.na(Age_Raw)) ~ Age_Raw[TimePoint == "12M" & !is.na(Age_Raw)][1],
      any(TimePoint == "24M" & !is.na(Age_Raw)) ~ Age_Raw[TimePoint == "24M" & !is.na(Age_Raw)][1],
      TRUE ~ min(Age_Raw, na.rm = TRUE)
    )
  ) %>%
  ungroup()

# Imputation for covariates
for (col in c("Ancestry_PC1", "Ancestry_PC2", "Ancestry_PC3", "Brain_proportion", "Baseline_Age")) {
  meta_run_base[[col]][is.na(meta_run_base[[col]])] <- mean(meta_run_base[[col]], na.rm=TRUE)
}

# Add W factors
W_df <- read.table(ruvs_w_file, header=TRUE, row.names=1, sep="\t")
colnames(W_df) <- paste0("W", 1:ncol(W_df))
common_samples <- intersect(meta_run_base$SampleID, rownames(W_df))
meta_run_base <- cbind(meta_run_base[match(common_samples, meta_run_base$SampleID), ], 
                       W_df[common_samples, ])

# --- [2] Load Counts ---
cat("\n[2] Loading Counts...\n")
counts_dt <- fread(counts_file, sep = "\t", nThread = num_cores)
genes <- counts_dt$Key; counts_dt$Key <- NULL
counts <- as.matrix(counts_dt); rownames(counts) <- genes
counts <- counts[, common_samples]

# Filter Low Count Genes
keep <- rowSums(counts >= 10) >= (0.15 * ncol(counts))
counts <- counts[keep, ]
cat(sprintf("  Genes: %d, Samples: %d\n", nrow(counts), ncol(counts)))

# --- ANALYSIS FUNCTION ---
run_interaction_analysis <- function(target_score_col, out_dir, score_name) {
  
  cat(sprintf("\n--- Running Analysis for: %s ---\n", score_name))
  
  # 1. Subset Metadata (Remove NAs for this specific score)
  meta_local <- meta_run_base[!is.na(meta_run_base[[target_score_col]]), ]
  
  # 2. Scale Variables (Scale within the analyzed subset)
  # NOTE: Score is scaled, Age is Baseline Fixed & Scaled
  meta_local$Score_Scaled <- scale(meta_local[[target_score_col]])[,1]
  meta_local$Age_Scaled <- scale(meta_local$Baseline_Age)[,1]
  meta_local$Brain_Prop_Scaled <- scale(meta_local$Brain_proportion)[,1]
  meta_local$PC1_Scaled <- scale(meta_local$Ancestry_PC1)[,1]
  meta_local$PC2_Scaled <- scale(meta_local$Ancestry_PC2)[,1]
  meta_local$PC3_Scaled <- scale(meta_local$Ancestry_PC3)[,1]
  
  # Match Counts to this subset
  counts_local <- counts[, meta_local$SampleID]
  
  # 3. Create DGE Object
  dge <- DGEList(counts = counts_local)
  dge <- calcNormFactors(dge, method = "TMM")
  
  # 4. Define Formula with Random Effect AND Interaction
  # Formula Explanation:
  # - Group: Mean difference between PD and Ctrl (Intercept shift)
  # - Score_Scaled: General effect of score change (Slope for Ctrl)
  # - Group:Score_Scaled: Difference in slope for PD compared to Ctrl (Interaction)
  # - (1|ParticipantID): Random intercept for repeated measures
  formula_str <- "~ Group + Score_Scaled + Group:Score_Scaled + Sex + Age_Scaled + Brain_Prop_Scaled + W1 + W2 + W3 + PC1_Scaled + PC2_Scaled + PC3_Scaled + (1|ParticipantID)"
  form <- as.formula(formula_str)
  cat(sprintf("  Formula: %s\n", formula_str))
  
  # 5. Run dream (LMM)
  param <- SnowParam(workers = num_cores, type = "SOCK", progressbar = FALSE)
  vobj <- voomWithDreamWeights(dge, form, meta_local, BPPARAM = param)
  fit <- dream(vobj, form, meta_local, BPPARAM = param)
  fit <- eBayes(fit)
  
  # 6. Extract Interaction Term
  # Term name is usually "GroupPD:Score_Scaled" because Ctrl is reference
  term_name <- "GroupPD:Score_Scaled"
  
  if (!term_name %in% colnames(fit$coefficients)) {
    cat("  Error: Interaction term 'GroupPD:Score_Scaled' not found.\n")
    print(colnames(fit$coefficients))
    return(NULL)
  }
  
  res <- topTable(fit, coef = term_name, number = Inf, sort.by = "none")
  
  # 7. Save Results
  score_sd <- sd(meta_local[[target_score_col]])
  
  out_df <- data.frame(
    Gene = rownames(res),
    baseMean = 2^res$AveExpr,
    # LogFC of Interaction: How much steeper (or shallower) is the PD slope vs Ctrl?
    Estimate_Interaction_Scaled = res$logFC,
    Estimate_Interaction_Raw = res$logFC / score_sd, 
    pvalue = res$P.Value,
    padj = res$adj.P.Val,
    t_stat = res$t
  ) %>% arrange(pvalue)
  
  outfile <- file.path(out_dir, sprintf("Final_Interaction_%s_PD_specific.tsv", score_name))
  fwrite(out_df, outfile, sep = "\t")
  cat(sprintf("  Significant Interaction Genes (FDR < 0.05): %d\n", sum(out_df$padj < 0.05, na.rm=TRUE)))
  
  # 8. Calculate Corrected Matrix (Visualization)
  # Remove Nuisance (Sex, Age, BrainProp, Ws, PCs)
  # KEEP Biological: Intercept, Group, Score, Interaction
  form_fixed <- as.formula("~ Group + Score_Scaled + Group:Score_Scaled + Sex + Age_Scaled + Brain_Prop_Scaled + W1 + W2 + W3 + PC1_Scaled + PC2_Scaled + PC3_Scaled")
  design_full <- model.matrix(form_fixed, data = meta_local)
  
  cols_keep <- grep("Group|Score|Intercept", colnames(design_full), value = TRUE)
  cols_remove <- setdiff(colnames(design_full), cols_keep)
  
  beta_nuisance <- fit$coefficients[, cols_remove, drop=FALSE]
  design_nuisance <- design_full[, cols_remove, drop=FALSE]
  corrected <- vobj$E - t(design_nuisance %*% t(beta_nuisance))
  
  out_mat <- as.data.frame(corrected)
  out_mat$Gene <- rownames(corrected)
  out_mat <- out_mat %>% select(Gene, everything())
  fwrite(out_mat, file.path(out_dir, "fully_corrected_matrix_interaction.tsv"), sep="\t")
}

# --- [3] Execute Analyses ---

# A. UPDRS Interaction
run_interaction_analysis("UPDRS_Raw", out_dir_updrs, "UPDRS")

# B. MoCA Interaction
run_interaction_analysis("MoCA_Raw", out_dir_moca, "MoCA")

cat("\nAll Interaction Analyses Complete.\n")
