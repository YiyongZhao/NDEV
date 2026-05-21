import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import shap
import warnings
import os
import argparse
import gc
import traceback
import random
from datetime import datetime
from scipy.stats import sem, t, wilcoxon, gaussian_kde
from scipy.special import expit
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.stats.multitest import multipletests
from itertools import product as iter_product
from joblib import Parallel, delayed

# Sklearn Imports
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (
    StratifiedGroupKFold, GroupShuffleSplit,
    cross_validate, StratifiedKFold
)
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, balanced_accuracy_score,
    confusion_matrix, make_scorer, brier_score_loss, f1_score,
    precision_recall_curve, auc, roc_curve, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.utils import resample
from sklearn.decomposition import PCA

# Try importing adjustText
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False

# Models
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Optional Models Check
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False
try:
    from interpret.glassbox import ExplainableBoostingClassifier
    EBM_AVAILABLE = True
except ImportError:
    EBM_AVAILABLE = False
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False

warnings.filterwarnings('ignore')

# Plotting Standards (Publication Ready)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})

def log(msg, indent=0):
    timestamp = datetime.now().strftime('%H:%M:%S')
    prefix = " " * indent
    print(f"[{timestamp}] {prefix}{msg}", flush=True)

# ==========================================
# 1. Configuration and reproducibility
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nested cross-validated PD classification using EV/WB gene and circRNA features."
    )
    parser.add_argument("--base-dir", default=".", help="Directory containing EV, WB and match files.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <base-dir>/Final_Paper_v70_FixedSHAP.")
    parser.add_argument("--ev-counts", default=None, help="EV linear RNA count matrix.")
    parser.add_argument("--ev-deg", default=None, help="EV linear RNA DEG table.")
    parser.add_argument("--ev-circ-counts", required=True, help="EV circRNA BSJ/count matrix.")
    parser.add_argument("--ev-circ-deg", required=True, help="EV circRNA DEG table.")
    parser.add_argument("--wb-counts", default=None, help="Whole-blood linear RNA count matrix.")
    parser.add_argument("--wb-deg", default=None, help="Whole-blood linear RNA DEG table.")
    parser.add_argument("--wb-circ-counts", required=True, help="Whole-blood circRNA BSJ/count matrix.")
    parser.add_argument("--wb-circ-deg", required=True, help="Whole-blood circRNA DEG table.")
    parser.add_argument("--match-file", default=None, help="Sample match and diagnosis table.")
    parser.add_argument("--gene-map", required=True, help="GENCODE/Ensembl gene annotation table.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p-thresh", type=float, default=0.05)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--lasso-c", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=40)
    parser.add_argument("--fast", action="store_true", help="Skip optional slow classifiers.")
    return parser.parse_args()

ARGS = parse_args()
BASE_DIR = ARGS.base_dir

EV_COUNTS_PATH = ARGS.ev_counts or os.path.join(BASE_DIR, "EV/1115_EMapper_readcounts.txt")
EV_DEG_FILE = ARGS.ev_deg or os.path.join(BASE_DIR, "EV/EV_EMapper_ENSG_brain_proportion_Final_DEGs_DREAM_PD_vs_Control.tsv")
EV_CIRC_COUNTS_PATH = ARGS.ev_circ_counts
EV_CIRC_DEG_FILE = ARGS.ev_circ_deg

WB_COUNTS_PATH = ARGS.wb_counts or os.path.join(BASE_DIR, "WB/1015samples_whole_blood_readcounts.tsv")
WB_DEG_FILE = ARGS.wb_deg or os.path.join(BASE_DIR, "WB/1014_WB_Neutrophil_Final_DEGs_DREAM_PD_vs_Control.tsv")
WB_CIRC_COUNTS_PATH = ARGS.wb_circ_counts
WB_CIRC_DEG_FILE = ARGS.wb_circ_deg

MATCH_FILE_PATH = ARGS.match_file or os.path.join(BASE_DIR, "WB_EV_match_ID.tsv")
GENE_MAP_FILE = ARGS.gene_map
OUTPUT_DIR = ARGS.output_dir or os.path.join(BASE_DIR, "Final_Paper_v70_FixedSHAP")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = ARGS.seed
P_VAL_THRESH = ARGS.p_thresh
POS_LABEL = "Parkinson's Disease"
N_OUTER_FOLDS = ARGS.outer_folds
N_INNER_FOLDS = ARGS.inner_folds
FIXED_LASSO_C = ARGS.lasso_c
ALLOW_MISSING_INDICATORS = False
GRID_N_JOBS = ARGS.n_jobs
SELECTOR_LASSO_NJOBS = 1
VERBOSE_LEVEL = 0
SHAP_BG_SAMPLES = 30
SHAP_MAX_TEST = 100
SHAP_NSAMPLES = 200

INCLUDE_SLOW_MODELS = not ARGS.fast
MODELS_TO_RUN = ['LR', 'RF', 'SVM_rbf', 'KNN', 'MLP', 'LDA', 'ET', 'ABC', 'SGD']
if XGB_AVAILABLE: MODELS_TO_RUN.append('XGB')
if LGBM_AVAILABLE: MODELS_TO_RUN.append('LGBM')
if INCLUDE_SLOW_MODELS:
    if CAT_AVAILABLE: MODELS_TO_RUN.append('CAT')
    if EBM_AVAILABLE: MODELS_TO_RUN.append('EBM')

log(f"Output: {OUTPUT_DIR}")
log(f"Models: {MODELS_TO_RUN}")
log(f"Strategy: NaN-fill + median imputation + missing indicator (allow={ALLOW_MISSING_INDICATORS})")
log(f"CV: {N_OUTER_FOLDS}-fold outer x {N_INNER_FOLDS}-fold inner nested CV")
log(f"Pipeline: imputer -> scaler -> SimpleLassoSelector(C={FIXED_LASSO_C}) -> classifier")
log("Grid search: manual group-aware implementation")
log(f"SHAP: bg={SHAP_BG_SAMPLES}, max_test={SHAP_MAX_TEST}, nsamples={SHAP_NSAMPLES}")
log("SHAP: fixed to best_algo and aggregated across all outer folds")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

seed_everything(SEED)

# ==========================================
# 2. Helpers
# ==========================================

def load_gene_mapping(map_path):
    try:
        df = pd.read_csv(map_path, sep='\t')
        mapping = dict(zip(df['ENSG_ID1'], df['SYMBOL_Name']))
        mapping.update(dict(zip(df['ENSEMBL_ID'], df['SYMBOL_Name'])))
        return mapping
    except Exception as e:
        log(f"Gene map warning: {e}")
        return {}

def load_process_counts(path):
    log(f"Reading {os.path.basename(path)}...", indent=2)
    try:
        df = pd.read_csv(path, sep='\t', index_col=0).T
    except:
        df = pd.read_csv(path, sep='\t', index_col='Key').T
    df_cpm = df.div(df.sum(axis=1), axis=0) * 1e6
    df_log = np.log2(df_cpm + 1)
    df_log = df_log.fillna(0)
    return df_log

def get_prior_stats(deg_path, thresh=0.05):
    log(f"Filtering DEGs from {os.path.basename(deg_path)} (p < {thresh})...", indent=2)
    df = pd.read_csv(deg_path, sep='\t', index_col=0)
    p_cols = [c for c in df.columns if 'pval' in c.lower() or 'p.val' in c.lower()]
    fc_cols = [c for c in df.columns if 'log2' in c.lower() or 'fold' in c.lower()]
    if not p_cols:
        raise ValueError(f"No P-val in {deg_path}")
    p_col = p_cols[0]
    df_sig = df[df[p_col] < thresh].copy()
    stats = pd.DataFrame(index=df_sig.index)
    stats['P_Value'] = df_sig[p_col]
    stats['Log2FC'] = df_sig[fc_cols[0]] if fc_cols else pd.Series(0, index=df_sig.index)
    return df_sig.index.tolist(), stats

def stratified_group_split(X, y, groups, test_size=0.2, seed=42):
    unique_groups = np.unique(groups)
    group_labels = []
    for g in unique_groups:
        mask = groups == g
        group_labels.append(y[mask][0])
    group_labels = np.array(group_labels)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_g_idx, te_g_idx = next(sss.split(unique_groups, group_labels))
    train_idx = np.where(np.isin(groups, unique_groups[tr_g_idx]))[0]
    test_idx = np.where(np.isin(groups, unique_groups[te_g_idx]))[0]
    return train_idx, test_idx

def get_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        nb = 0 if thresh == 1.0 else (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(nb)
    return net_benefits

def get_metrics(y_true, y_prob, n_genes, prefix="test"):
    if np.min(y_prob) < 0 or np.max(y_prob) > 1:
        y_prob = expit(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'N_Genes': n_genes,
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        f"{prefix}_specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_roc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        f"{prefix}_auprc": float(average_precision_score(y_true, y_prob)),
    }

def map_gene_symbols(features, gene_map):
    symbols = []
    for g in features:
        if 'circ:' in g:
            symbols.append(g)
        elif ':' in g:
            prefix, gene_id = g.split(':', 1)
            clean_id = gene_id.split('.')[0] if '.' in gene_id else gene_id
            symbol = gene_map.get(clean_id, clean_id)
            symbols.append(f"{prefix}:{symbol}")
        else:
            clean_id = g.split('.')[0] if '.' in g else g
            symbols.append(gene_map.get(clean_id, clean_id))
    return symbols

scoring_dict = {'auc': 'roc_auc', 'acc': 'accuracy'}


# ==========================================
# 3. SimpleLassoSelector
# ==========================================

class SimpleLassoSelector(BaseEstimator, TransformerMixin):
    """
    Simpler selection: Fixed C LogisticRegression with L1 penalty.
    """
    def __init__(self, C=0.2, seed=42):
        self.C = C
        self.seed = seed

    def fit(self, X, y=None, groups=None):
        n_features = X.shape[1]

        self.estimator_ = LogisticRegression(
            C=self.C,
            penalty='l1',
            solver='liblinear',
            class_weight='balanced',
            random_state=self.seed,
            max_iter=5000
        )

        self.estimator_.fit(X, y)

        if self.estimator_.coef_.ndim > 1:
            coefs = np.abs(self.estimator_.coef_[0])
        else:
            coefs = np.abs(self.estimator_.coef_)

        self.support_ = coefs > 1e-6
        self.n_features_selected_ = int(self.support_.sum())

        self.best_C_ = self.C
        self.selection_info_ = f"Lasso(C={self.C})"
        self.used_fallback_ = False

        if self.n_features_selected_ == 0:
            log(f"      Warning: Lasso(C={self.C}) selected 0 features. Fallback to top 5 by coef magnitude.", indent=6)
            top_k_idx = np.argsort(coefs)[::-1][:5]
            self.support_[top_k_idx] = True
            self.n_features_selected_ = int(self.support_.sum())
            self.selection_info_ += "_FallbackTop5"
            self.used_fallback_ = True

        log(f"       Selected {self.n_features_selected_} features using {self.selection_info_}", indent=6)
        return self

    def transform(self, X):
        return X[:, self.support_] if not hasattr(X, 'iloc') else X.values[:, self.support_]

    def get_support(self):
        return self.support_

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features)[self.support_]
        return None


# ==========================================
# 3b. MissingIndicatorFilter
# ==========================================

class MissingIndicatorFilter(BaseEstimator, TransformerMixin):
    def __init__(self, allow_indicators=True, n_original_features=None):
        self.allow_indicators = allow_indicators
        self.n_original_features = n_original_features

    def fit(self, X, y=None, **fit_params):
        n_cols = X.shape[1]
        if self.allow_indicators or self.n_original_features is None:
            self.keep_mask_ = np.ones(n_cols, dtype=bool)
        else:
            self.keep_mask_ = np.zeros(n_cols, dtype=bool)
            self.keep_mask_[:self.n_original_features] = True
        self.n_removed_ = int((~self.keep_mask_).sum())
        return self

    def transform(self, X):
        return X[:, self.keep_mask_]


# ==========================================
# 3c. Build Full Pipeline + Classifier
# ==========================================

def build_full_pipeline(clf, seed, n_original_features=None):
    steps = [
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('indicator_filter', MissingIndicatorFilter(
            allow_indicators=ALLOW_MISSING_INDICATORS,
            n_original_features=n_original_features
        )),
        ('scaler', StandardScaler()),
        ('selector', SimpleLassoSelector(
            C=FIXED_LASSO_C,
            seed=seed
        )),
        ('classifier', clf)
    ]
    return Pipeline(steps)


def get_classifier(model_name, seed):
    clf = None
    params = {}
    if model_name == 'LR':
        clf = LogisticRegression(class_weight='balanced', random_state=seed, n_jobs=1)
        params = {'classifier__C': [0.01, 0.1, 1]}
    elif model_name == 'RF':
        clf = RandomForestClassifier(class_weight='balanced', random_state=seed, n_jobs=1)
        params = {'classifier__n_estimators': [100, 300],
                  'classifier__max_depth': [3, 5],
                  'classifier__min_samples_leaf': [5, 10]}
    elif model_name == 'XGB':
        clf = XGBClassifier(eval_metric='logloss', random_state=seed, n_jobs=1)
        params = {'classifier__n_estimators': [100, 200],
                  'classifier__max_depth': [3],
                  'classifier__min_child_weight': [5, 10]}
    elif model_name == 'SVM_rbf':
        clf = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed, cache_size=1000)
        params = {'classifier__C': [0.1, 1, 10]}
    elif model_name == 'KNN':
        clf = KNeighborsClassifier(n_jobs=1)
        params = {'classifier__n_neighbors': [9]}
    elif model_name == 'MLP':
        clf = MLPClassifier(random_state=seed, max_iter=500, early_stopping=True, tol=1e-4)
        params = {'classifier__hidden_layer_sizes': [(50,)]}
    elif model_name == 'LDA':
        clf = LinearDiscriminantAnalysis()
        params = {}
    elif model_name == 'ET':
        clf = ExtraTreesClassifier(class_weight='balanced', random_state=seed, n_jobs=1)
        params = {'classifier__n_estimators': [100, 300],
                  'classifier__max_depth': [3, 5],
                  'classifier__min_samples_leaf': [5, 10]}
    elif model_name == 'ABC':
        clf = AdaBoostClassifier(random_state=seed)
        params = {'classifier__n_estimators': [50]}
    elif model_name == 'SGD':
        clf = SGDClassifier(
            loss='modified_huber', class_weight='balanced',
            random_state=seed, n_jobs=1
        )
        params = {'classifier__alpha': [0.001, 0.01]}
    elif model_name == 'LGBM' and LGBM_AVAILABLE:
        clf = LGBMClassifier(random_state=seed, verbose=-1, n_jobs=1, deterministic=True)
        params = {'classifier__n_estimators': [100, 200],
                  'classifier__max_depth': [3],
                  'classifier__min_child_samples': [10, 20]}
    elif model_name == 'CAT' and CAT_AVAILABLE:
        clf = CatBoostClassifier(random_seed=seed, verbose=0, thread_count=1, depth=3)
        params = {}
    elif model_name == 'EBM' and EBM_AVAILABLE:
        clf = ExplainableBoostingClassifier(random_state=seed, n_jobs=1, interactions=0)
        params = {'classifier__max_bins': [256]}
    return clf, params


# ==========================================
# 3d. Manual Grid Search
# ==========================================

def _eval_one_fold(pipe_clone, params, X, y, groups,
                   inner_train_idx, inner_val_idx, scoring_fn):
    pipe = clone(pipe_clone)
    if params:
        pipe.set_params(**params)

    X_inn_tr = X.iloc[inner_train_idx] if hasattr(X, 'iloc') else X[inner_train_idx]
    y_inn_tr = y[inner_train_idx]
    groups_inn_tr = groups[inner_train_idx]

    X_inn_val = X.iloc[inner_val_idx] if hasattr(X, 'iloc') else X[inner_val_idx]
    y_inn_val = y[inner_val_idx]

    pipe.fit(X_inn_tr, y_inn_tr, selector__groups=groups_inn_tr)

    if hasattr(pipe, 'predict_proba'):
        y_prob = pipe.predict_proba(X_inn_val)[:, 1]
    else:
        y_prob = expit(pipe.decision_function(X_inn_val))

    try:
        score = scoring_fn(y_inn_val, y_prob)
    except ValueError:
        score = 0.5
    return score


def manual_grid_search(pipeline, param_grid, X, y, groups, inner_cv,
                       scoring_fn=roc_auc_score, n_jobs=1):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = [dict(zip(keys, v)) for v in iter_product(*values)] if keys else [{}]

    splits = list(inner_cv.split(X, y, groups))

    best_score = -np.inf
    best_params = {}

    for params in all_combos:
        if n_jobs == 1 or len(splits) <= 2:
            fold_scores = []
            for inner_train_idx, inner_val_idx in splits:
                s = _eval_one_fold(pipeline, params, X, y, groups,
                                   inner_train_idx, inner_val_idx, scoring_fn)
                fold_scores.append(s)
        else:
            fold_scores = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(_eval_one_fold)(
                    pipeline, params, X, y, groups,
                    tr_idx, val_idx, scoring_fn
                )
                for tr_idx, val_idx in splits
            )

        mean_score = np.mean(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    best_pipe = clone(pipeline)
    if best_params:
        best_pipe.set_params(**best_params)
    best_pipe.fit(X, y, selector__groups=groups)

    return best_pipe, best_params, best_score


def manual_oof_predict(pipeline, X, y, groups, cv_splitter):
    oof_probs = np.full(len(y), np.nan)

    for fold_train_idx, fold_val_idx in cv_splitter.split(X, y, groups):
        pipe = clone(pipeline)

        X_f_tr = X.iloc[fold_train_idx] if hasattr(X, 'iloc') else X[fold_train_idx]
        y_f_tr = y[fold_train_idx]
        groups_f_tr = groups[fold_train_idx]
        X_f_val = X.iloc[fold_val_idx] if hasattr(X, 'iloc') else X[fold_val_idx]

        pipe.fit(X_f_tr, y_f_tr, selector__groups=groups_f_tr)

        if hasattr(pipe, 'predict_proba'):
            probs = pipe.predict_proba(X_f_val)[:, 1]
        else:
            probs = expit(pipe.decision_function(X_f_val))

        oof_probs[fold_val_idx] = probs

    return oof_probs


# ==========================================
# 3e. Extract Pipeline Info with Assertions
# ==========================================

def extract_pipeline_info(fitted_pipe, original_feature_names):
    imputer = fitted_pipe.named_steps['imputer']
    ind_filter = fitted_pipe.named_steps['indicator_filter']
    selector = fitted_pipe.named_steps['selector']

    orig_names = list(original_feature_names)
    if hasattr(imputer, 'indicator_') and imputer.indicator_ is not None:
        indicator_feat_indices = imputer.indicator_.features_
        indicator_names = [f"{orig_names[i]}_is_missing" for i in indicator_feat_indices]
    else:
        indicator_names = []

    all_names_post_imputer = orig_names + indicator_names

    filter_mask = ind_filter.keep_mask_
    assert len(all_names_post_imputer) == len(filter_mask), (
        f"Indicator filter alignment error: {len(all_names_post_imputer)} names vs "
        f"{len(filter_mask)} filter mask"
    )
    all_names_post_filter = [n for n, m in zip(all_names_post_imputer, filter_mask) if m]

    sel_mask = selector.get_support()
    assert len(all_names_post_filter) == len(sel_mask), (
        f"Feature name alignment error: {len(all_names_post_filter)} names vs "
        f"{len(sel_mask)} selector mask entries"
    )

    selected_names = [all_names_post_filter[i] for i, m in enumerate(sel_mask) if m]
    n_selected = int(sel_mask.sum())
    best_C = getattr(selector, 'best_C_', 'N/A')
    used_fallback = getattr(selector, 'used_fallback_', False)
    sel_info = getattr(selector, 'selection_info_', '')
    c_info = sel_info

    bio_features = [f for f in selected_names if not f.endswith('_is_missing')]
    missing_features = [f for f in selected_names if f.endswith('_is_missing')]
    n_bio = len(bio_features)
    n_miss = len(missing_features)

    return selected_names, n_selected, c_info, n_bio, n_miss, missing_features


# ==========================================
# 4. Data Loading with NaN-Fill Strategy
# ==========================================
log("1. Loading & Aligning Data...")
gene_map = load_gene_mapping(GENE_MAP_FILE)

df_match = pd.read_csv(MATCH_FILE_PATH, sep='\t')
df_match = df_match[df_match['Diagnosis'].isin([POS_LABEL, "No Neurological Diagnosis"])].copy()
df_match.dropna(subset=['EV_ID', 'WB_ID', 'Participant_ID'], inplace=True)

log(f"  Original samples after diagnosis filter: {len(df_match)}")

# --- 4.1 Load Raw Counts ---
X_ev_raw = load_process_counts(EV_COUNTS_PATH)
X_wb_raw = load_process_counts(WB_COUNTS_PATH)
X_ev_circ_raw = load_process_counts(EV_CIRC_COUNTS_PATH)
X_wb_circ_raw = load_process_counts(WB_CIRC_COUNTS_PATH)

# --- 4.2 Load & Filter DEGs (P < 0.05) ---
ev_genes_init, ev_stats = get_prior_stats(EV_DEG_FILE, P_VAL_THRESH)
ev_genes = sorted(list(set(ev_genes_init) & set(X_ev_raw.columns)))
log(f"  Valid EV Genecode genes found (p<0.05): {len(ev_genes)}")

wb_genes_init, wb_stats = get_prior_stats(WB_DEG_FILE, P_VAL_THRESH)
wb_genes = sorted(list(set(wb_genes_init) & set(X_wb_raw.columns)))
log(f"  Valid WB Genecode genes found (p<0.05): {len(wb_genes)}")

ev_circ_genes_init, ev_circ_stats = get_prior_stats(EV_CIRC_DEG_FILE, P_VAL_THRESH)
ev_circ_genes = sorted(list(set(ev_circ_genes_init) & set(X_ev_circ_raw.columns)))
log(f"  Valid EV circRNAs found (p<0.05): {len(ev_circ_genes)}")

wb_circ_genes_init, wb_circ_stats = get_prior_stats(WB_CIRC_DEG_FILE, P_VAL_THRESH)
wb_circ_genes = sorted(list(set(wb_circ_genes_init) & set(X_wb_circ_raw.columns)))
log(f"  Valid WB circRNAs found (p<0.05): {len(wb_circ_genes)}")

# --- 4.3 Sample Availability ---
log("\n" + "=" * 60)
log("SAMPLE AVAILABILITY DIAGNOSTICS")
log("=" * 60)

valid_ev = df_match['EV_ID'].isin(X_ev_raw.index)
valid_wb = df_match['WB_ID'].isin(X_wb_raw.index)
valid_ev_circ = df_match['EV_ID'].isin(X_ev_circ_raw.index)
valid_wb_circ = df_match['WB_ID'].isin(X_wb_circ_raw.index)

log(f"  Samples with EV Genecode: {valid_ev.sum()}, WB Genecode: {valid_wb.sum()}")
log(f"  Samples with EV circRNA: {valid_ev_circ.sum()}, WB circRNA: {valid_wb_circ.sum()}")
log(f"  Samples with EV+WB Genecode: {(valid_ev & valid_wb).sum()}")
log(f"  Samples with all 4 data types: {(valid_ev & valid_wb & valid_ev_circ & valid_wb_circ).sum()}")

missing_ev_circ = valid_ev & valid_wb & ~valid_ev_circ
missing_wb_circ = valid_ev & valid_wb & ~valid_wb_circ
log(f"  Missing EV circRNA (NaN-fill): {missing_ev_circ.sum()}")
log(f"  Missing WB circRNA (NaN-fill): {missing_wb_circ.sum()}")
log("=" * 60 + "\n")

# --- 4.4 Base requirement: EV + WB genecode ---
df_final = df_match[valid_ev & valid_wb].copy()
df_final.sort_values('Participant_ID', inplace=True)

y = np.where(df_final['Diagnosis'] == POS_LABEL, 1, 0)
groups = df_final['Participant_ID'].values

log(f"  Final samples: {len(df_final)} (PD={np.sum(y)}, CTRL={len(y) - np.sum(y)})")

# [NEW] Check missingness-diagnosis correlation
log("\n" + "=" * 60)
log("MISSING DATA x DIAGNOSIS CORRELATION CHECK")
log("=" * 60)
for circ_name, circ_valid in [('EV_circRNA', valid_ev_circ), ('WB_circRNA', valid_wb_circ)]:
    final_mask = valid_ev & valid_wb
    subset_diag = df_match.loc[final_mask, 'Diagnosis'].values
    subset_has_circ = circ_valid[final_mask].values
    n_pd_has = np.sum((subset_diag == POS_LABEL) & subset_has_circ)
    n_pd_miss = np.sum((subset_diag == POS_LABEL) & ~subset_has_circ)
    n_ctrl_has = np.sum((subset_diag != POS_LABEL) & subset_has_circ)
    n_ctrl_miss = np.sum((subset_diag != POS_LABEL) & ~subset_has_circ)
    total = len(subset_diag)
    pct_miss = (n_pd_miss + n_ctrl_miss) / total * 100
    log(f"  {circ_name}:")
    log(f"    Has data  -> PD: {n_pd_has}, CTRL: {n_ctrl_has}")
    log(f"    Missing   -> PD: {n_pd_miss}, CTRL: {n_ctrl_miss}")
    log(f"    Total missing: {n_pd_miss + n_ctrl_miss} ({pct_miss:.1f}%)")
    if (n_pd_miss + n_ctrl_miss) > 0:
        pd_miss_rate = n_pd_miss / (n_pd_has + n_pd_miss) if (n_pd_has + n_pd_miss) > 0 else 0
        ctrl_miss_rate = n_ctrl_miss / (n_ctrl_has + n_ctrl_miss) if (n_ctrl_has + n_ctrl_miss) > 0 else 0
        log(f"    PD missing rate: {pd_miss_rate:.3f}, CTRL missing rate: {ctrl_miss_rate:.3f}")
        if abs(pd_miss_rate - ctrl_miss_rate) > 0.1:
            log(f"    WARNING: WARNING: Missingness is differentially distributed across groups!")
            log(f"    WARNING: ALLOW_MISSING_INDICATORS is set to {ALLOW_MISSING_INDICATORS}")
        else:
            log(f"    OK Missingness is approximately balanced across groups.")
log("=" * 60 + "\n")

# --- 4.5 Build Matrices with NaN-Fill ---
def extract_with_nanfill(raw_data, sample_ids, features, id_column):
    result = pd.DataFrame(index=range(len(sample_ids)), columns=features, dtype=float)
    missing_count = 0
    for i, sample_id in enumerate(sample_ids):
        if sample_id in raw_data.index:
            result.iloc[i] = raw_data.loc[sample_id, features].values.astype(float)
        else:
            result.iloc[i] = np.nan
            missing_count += 1
    return result.astype(float), missing_count

X_ev_gene = X_ev_raw.loc[df_final['EV_ID'], ev_genes].reset_index(drop=True)
X_wb_gene = X_wb_raw.loc[df_final['WB_ID'], wb_genes].reset_index(drop=True)

X_ev_circ, ev_circ_missing = extract_with_nanfill(
    X_ev_circ_raw, df_final['EV_ID'].values, ev_circ_genes, 'EV_ID')
X_ev_circ = X_ev_circ.add_prefix('EV_circ:')

X_wb_circ, wb_circ_missing = extract_with_nanfill(
    X_wb_circ_raw, df_final['WB_ID'].values, wb_circ_genes, 'WB_ID')
X_wb_circ = X_wb_circ.add_prefix('WB_circ:')

X_ev_combined = pd.concat([X_ev_gene, X_ev_circ], axis=1)
X_wb_combined = pd.concat([X_wb_gene, X_wb_circ], axis=1)

log(f"  EV_Gene:{X_ev_gene.shape[1]} EV_circ:{X_ev_circ.shape[1]} WB_Gene:{X_wb_gene.shape[1]} WB_circ:{X_wb_circ.shape[1]}")
log(f"  NaN-filled: EV_circ={ev_circ_missing}, WB_circ={wb_circ_missing}")

# Save diagnostics
sample_diag = df_final[['Participant_ID', 'EV_ID', 'WB_ID', 'Diagnosis']].copy()
sample_diag['has_EV_circ'] = df_final['EV_ID'].isin(X_ev_circ_raw.index).values
sample_diag['has_WB_circ'] = df_final['WB_ID'].isin(X_wb_circ_raw.index).values
sample_diag.to_csv(os.path.join(OUTPUT_DIR, "Sample_Diagnostics.csv"), index=False)

del X_ev_raw, X_wb_raw, X_ev_circ_raw, X_wb_circ_raw; gc.collect()


# ==========================================
# 5. Nested CV Benchmark
# ==========================================
results = []
curve_data_storage = {}
prediction_history = []
fold_indices_log = []

#  FIXED: Removed best_shap_models (cherry-picking), replaced with shap_fold_collector
best_pca_data = {k: None for k in
    ['EV_Genecode', 'EV_circRNA', 'EV_Combined', 'WB_Genecode', 'WB_circRNA', 'WB_Combined']}
best_pca_auc = {k: -1 for k in best_pca_data}  # Track best AUC for PCA separately

# Feature selection frequency tracker
feature_selection_tracker = {}
missing_indicator_tracker = {}

#  NEW: SHAP fold collector - collects data from ALL folds for aggregation
shap_fold_collector = {}  # {(algo, mod): [fold_data_dict, ...]}

# Modalities
all_modality_data = {
    'EV_Genecode': X_ev_gene, 'EV_circRNA': X_ev_circ,
    'EV_Combined': X_ev_combined, 'WB_Genecode': X_wb_gene,
    'WB_circRNA': X_wb_circ, 'WB_Combined': X_wb_combined,
}

# Pre-compute outer fold indices ONCE
outer_cv = StratifiedGroupKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
precomputed_folds = []
for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X_ev_gene, y, groups)):
    precomputed_folds.append((tr_idx, te_idx))
    fold_indices_log.append({
        'fold': fold_i, 'n_train': len(tr_idx), 'n_test': len(te_idx),
        'train_PD': int(np.sum(y[tr_idx])), 'train_CTRL': int(np.sum(y[tr_idx] == 0)),
        'test_PD': int(np.sum(y[te_idx])), 'test_CTRL': int(np.sum(y[te_idx] == 0)),
        'test_participants': ','.join(groups[te_idx].astype(str))
    })

pd.DataFrame(fold_indices_log).to_csv(os.path.join(OUTPUT_DIR, "Fold_Assignments.csv"), index=False)

# -- Main Loop --
for fold_i, (tr_idx, te_idx) in enumerate(precomputed_folds):
    current_seed = SEED + fold_i
    seed_everything(current_seed)

    y_tr, y_te = y[tr_idx], y[te_idx]
    gr_tr = groups[tr_idx]

    log(f"\n{'=' * 60}")
    log(f"=== OUTER FOLD {fold_i + 1}/{N_OUTER_FOLDS} (seed={current_seed}) ===")
    log(f"    Train: {len(tr_idx)} (PD={np.sum(y_tr)}) | Test: {len(te_idx)} (PD={np.sum(y_te)})")

    for algo in MODELS_TO_RUN:
        probs_storage = {}
        stack_tr_probs = {}

        for mod, X_full in all_modality_data.items():
            X_tr_raw = X_full.iloc[tr_idx].copy()
            X_te_raw = X_full.iloc[te_idx].copy()
            feature_names = list(X_full.columns)
            n_original_features = len(feature_names)

            clf_base, clf_params = get_classifier(algo, current_seed)
            if clf_base is None:
                continue

            full_pipe = build_full_pipeline(clf_base, current_seed, n_original_features)

            inner_cv = StratifiedGroupKFold(
                n_splits=N_INNER_FOLDS, shuffle=True, random_state=current_seed
            )

            try:
                if clf_params:
                    best_pipe, best_params, val_roc = manual_grid_search(
                        full_pipe, clf_params, X_tr_raw, y_tr, gr_tr,
                        inner_cv, scoring_fn=roc_auc_score,
                        n_jobs=GRID_N_JOBS
                    )
                else:
                    best_pipe = clone(full_pipe)
                    best_pipe.fit(X_tr_raw, y_tr, selector__groups=gr_tr)
                    best_params = "Default"
                    val_roc = None
            except Exception as e:
                log(f"    {algo}/{mod}: Pipeline failed: {e}", indent=4)
                traceback.print_exc()
                continue

            try:
                selected_feat_names, n_genes, c_info, n_bio, n_miss, miss_feats = \
                    extract_pipeline_info(best_pipe, feature_names)
            except AssertionError as e:
                log(f"    {algo}/{mod}: Feature alignment error: {e}", indent=4)
                continue

            if n_genes == 0:
                continue

            final_symbols = map_gene_symbols(selected_feat_names, gene_map)
            indicator_note = f" [{n_miss} indicators]" if n_miss > 0 else ""
            log(f"    {algo}/{mod}: {n_genes} features ({c_info}, bio={n_bio}{indicator_note})", indent=4)

            if mod not in feature_selection_tracker:
                feature_selection_tracker[mod] = {}
            ft_key = (fold_i, algo)
            feature_selection_tracker[mod][ft_key] = selected_feat_names

            if n_miss > 0:
                if mod not in missing_indicator_tracker:
                    missing_indicator_tracker[mod] = {}
                missing_indicator_tracker[mod][ft_key] = miss_feats

            # Get preprocessed data for SHAP collection
            preprocess_pipe = Pipeline(best_pipe.steps[:-1])
            X_tr_sel = preprocess_pipe.transform(X_tr_raw)
            X_te_sel = preprocess_pipe.transform(X_te_raw)
            best_clf = best_pipe.named_steps['classifier']

            # Predictions
            if hasattr(best_pipe, "predict_proba"):
                p_te = best_pipe.predict_proba(X_te_raw)[:, 1]
                p_tr_full = best_pipe.predict_proba(X_tr_raw)[:, 1]
            else:
                p_te = best_pipe.decision_function(X_te_raw)
                p_tr_full = best_pipe.decision_function(X_tr_raw)

            # OOF predictions for late fusion
            if mod in ['EV_Genecode', 'WB_Genecode', 'EV_Combined', 'WB_Combined']:
                oof_cv = StratifiedGroupKFold(
                    n_splits=N_INNER_FOLDS, shuffle=True, random_state=current_seed
                )
                p_tr_oof = manual_oof_predict(
                    best_pipe, X_tr_raw, y_tr, gr_tr, oof_cv
                )
                probs_storage[mod] = p_te
                stack_tr_probs[mod] = p_tr_oof

            tr_mets = get_metrics(y_tr, p_tr_full, n_genes, prefix="train")
            te_mets = get_metrics(y_te, p_te, n_genes, prefix="test")
            val_metrics = {'val_roc': val_roc} if val_roc is not None else {}

            row = {
                'Algorithm': algo, 'Modality': mod, 'Fold': fold_i,
                'N_Genes': n_genes, 'N_Bio_Features': n_bio,
                'N_Missing_Indicators': n_miss,
                'Key_Params': str(best_params),
                'LASSO_C_Info': c_info,
                'Selected_Gene_Symbol': ",".join(final_symbols),
                'Selected_Gene_Ensembl': ",".join(selected_feat_names)
            }
            row.update(tr_mets); row.update(val_metrics); row.update(te_mets)
            results.append(row)

            te_ids = groups[te_idx]
            for i, _id in enumerate(te_ids):
                prediction_history.append({
                    'Participant_ID': _id, 'Algorithm': algo, 'Modality': mod,
                    'Fold': fold_i, 'True_Label': y_te[i],
                    'Prob_PD': p_te[i], 'Predicted_Label': 1 if p_te[i] > 0.5 else 0
                })

            curr_key = (algo, mod)
            if curr_key not in curve_data_storage:
                curve_data_storage[curr_key] = []
            curve_data_storage[curr_key].append({
                'y_true': y_te, 'y_prob': p_te, 'auc': te_mets['test_roc']
            })

            #  FIXED: Collect SHAP data for ALL folds (no cherry-picking)
            # Only collect for base modalities (not LateFusion)
            if mod in ['EV_Genecode', 'EV_circRNA', 'EV_Combined',
                        'WB_Genecode', 'WB_circRNA', 'WB_Combined']:
                collector_key = (algo, mod)
                if collector_key not in shap_fold_collector:
                    shap_fold_collector[collector_key] = []
                shap_fold_collector[collector_key].append({
                    'fold': fold_i,
                    'clf': best_clf,
                    'X_tr_sel': X_tr_sel.copy(),
                    'X_te_sel': X_te_sel.copy(),
                    'y_test': y_te.copy(),
                    'features': list(selected_feat_names),
                    'symbols': list(final_symbols),
                    'test_auc': te_mets['test_roc'],
                    'seed': current_seed
                })

            # Track best model for PCA (keep this separate, single-fold is OK for PCA)
            if mod in best_pca_data:
                if te_mets['test_roc'] > best_pca_auc[mod]:
                    best_pca_auc[mod] = te_mets['test_roc']
                    best_pca_data[mod] = {
                        'X_tr': X_tr_sel, 'y_tr': y_tr,
                        'X_te': X_te_sel, 'y_te': y_te, 'algo': algo
                    }

        # --- Late Fusion: Genecode ---
        if 'EV_Genecode' in probs_storage and 'WB_Genecode' in probs_storage:
            ev_tr_o, wb_tr_o = stack_tr_probs['EV_Genecode'], stack_tr_probs['WB_Genecode']
            ev_te, wb_te = probs_storage['EV_Genecode'], probs_storage['WB_Genecode']
            if np.min(ev_tr_o) < 0: ev_tr_o, ev_te = expit(ev_tr_o), expit(ev_te)
            if np.min(wb_tr_o) < 0: wb_tr_o, wb_te = expit(wb_tr_o), expit(wb_te)
            X_st_tr = np.column_stack([ev_tr_o, wb_tr_o])
            X_st_te = np.column_stack([ev_te, wb_te])

            meta = LogisticRegression(random_state=current_seed, n_jobs=1)
            inner_cv_lf = StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=current_seed)
            cv_res = cross_validate(meta, X_st_tr, y_tr, groups=gr_tr, cv=inner_cv_lf, scoring=scoring_dict)
            val_metrics = {f"val_{k.replace('test_', '')}": np.mean(v) for k, v in cv_res.items()}
            meta.fit(X_st_tr, y_tr)
            p_tr_fus = meta.predict_proba(X_st_tr)[:, 1]
            p_fus = meta.predict_proba(X_st_te)[:, 1]

            tr_mets = get_metrics(y_tr, p_tr_fus, n_genes="NA", prefix="train")
            te_mets = get_metrics(y_te, p_fus, n_genes="NA", prefix="test")
            row = {'Algorithm': algo, 'Modality': 'LateFusion_Genecode', 'Fold': fold_i,
                   'N_Genes': 'NA', 'N_Bio_Features': 'NA', 'N_Missing_Indicators': 0,
                   'Key_Params': 'Meta-LR', 'LASSO_C_Info': 'NA',
                   'Selected_Gene_Symbol': 'NA', 'Selected_Gene_Ensembl': 'NA'}
            row.update(tr_mets); row.update(val_metrics); row.update(te_mets)
            results.append(row)
            for i, _id in enumerate(groups[te_idx]):
                prediction_history.append({'Participant_ID': _id, 'Algorithm': algo,
                    'Modality': 'LateFusion_Genecode', 'Fold': fold_i,
                    'True_Label': y_te[i], 'Prob_PD': p_fus[i],
                    'Predicted_Label': 1 if p_fus[i] > 0.5 else 0})
            curr_key = (algo, 'LateFusion_Genecode')
            if curr_key not in curve_data_storage: curve_data_storage[curr_key] = []
            curve_data_storage[curr_key].append({'y_true': y_te, 'y_prob': p_fus, 'auc': te_mets['test_roc']})

        # --- Late Fusion: Combined ---
        if 'EV_Combined' in probs_storage and 'WB_Combined' in probs_storage:
            ev_tr_o, wb_tr_o = stack_tr_probs['EV_Combined'], stack_tr_probs['WB_Combined']
            ev_te, wb_te = probs_storage['EV_Combined'], probs_storage['WB_Combined']
            if np.min(ev_tr_o) < 0: ev_tr_o, ev_te = expit(ev_tr_o), expit(ev_te)
            if np.min(wb_tr_o) < 0: wb_tr_o, wb_te = expit(wb_tr_o), expit(wb_te)
            X_st_tr = np.column_stack([ev_tr_o, wb_tr_o])
            X_st_te = np.column_stack([ev_te, wb_te])

            meta = LogisticRegression(random_state=current_seed, n_jobs=1)
            inner_cv_lf = StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=current_seed)
            cv_res = cross_validate(meta, X_st_tr, y_tr, groups=gr_tr, cv=inner_cv_lf, scoring=scoring_dict)
            val_metrics = {f"val_{k.replace('test_', '')}": np.mean(v) for k, v in cv_res.items()}
            meta.fit(X_st_tr, y_tr)
            p_tr_fus = meta.predict_proba(X_st_tr)[:, 1]
            p_fus = meta.predict_proba(X_st_te)[:, 1]

            tr_mets = get_metrics(y_tr, p_tr_fus, n_genes="NA", prefix="train")
            te_mets = get_metrics(y_te, p_fus, n_genes="NA", prefix="test")
            row = {'Algorithm': algo, 'Modality': 'LateFusion_Combined', 'Fold': fold_i,
                   'N_Genes': 'NA', 'N_Bio_Features': 'NA', 'N_Missing_Indicators': 0,
                   'Key_Params': 'Meta-LR', 'LASSO_C_Info': 'NA',
                   'Selected_Gene_Symbol': 'NA', 'Selected_Gene_Ensembl': 'NA'}
            row.update(tr_mets); row.update(val_metrics); row.update(te_mets)
            results.append(row)
            for i, _id in enumerate(groups[te_idx]):
                prediction_history.append({'Participant_ID': _id, 'Algorithm': algo,
                    'Modality': 'LateFusion_Combined', 'Fold': fold_i,
                    'True_Label': y_te[i], 'Prob_PD': p_fus[i],
                    'Predicted_Label': 1 if p_fus[i] > 0.5 else 0})
            curr_key = (algo, 'LateFusion_Combined')
            if curr_key not in curve_data_storage: curve_data_storage[curr_key] = []
            curve_data_storage[curr_key].append({'y_true': y_te, 'y_prob': p_fus, 'auc': te_mets['test_roc']})


# ==========================================
# 6. Feature Selection Frequency
# ==========================================
log("\nGenerating Feature Selection Frequency Tables...")
for mod in feature_selection_tracker:
    all_feats = set()
    fold_data = feature_selection_tracker[mod]
    for key, feats in fold_data.items():
        all_feats.update(feats)

    if not all_feats:
        continue

    freq = {}
    total_runs = len(fold_data)
    for feat in all_feats:
        count = sum(1 for feats in fold_data.values() if feat in feats)
        freq[feat] = count

    freq_df = pd.DataFrame.from_dict(freq, orient='index', columns=['Selection_Count'])
    freq_df['Selection_Freq'] = freq_df['Selection_Count'] / total_runs
    freq_df['Is_Missing_Indicator'] = freq_df.index.str.endswith('_is_missing')
    freq_df = freq_df.sort_values('Selection_Count', ascending=False)
    freq_df.to_csv(os.path.join(OUTPUT_DIR, f"Feature_Frequency_{mod}.csv"))

    n_total_feat = len(freq_df)
    n_indicator = freq_df['Is_Missing_Indicator'].sum()
    n_high_freq = (freq_df['Selection_Freq'] >= 0.5).sum()
    log(f"  {mod}: {n_total_feat} unique features across folds, "
        f"{n_high_freq} selected >=50% of runs, "
        f"{n_indicator} are _is_missing indicators")

if missing_indicator_tracker:
    log("\nWARNING: Missing Indicator Selection Report:")
    for mod, data in missing_indicator_tracker.items():
        n_runs_with_miss = len(data)
        total_runs = len(feature_selection_tracker.get(mod, {}))
        all_miss_feats = set()
        for feats in data.values():
            all_miss_feats.update(feats)
        log(f"  {mod}: _is_missing features selected in {n_runs_with_miss}/{total_runs} runs")
        log(f"    Unique indicators: {list(all_miss_feats)[:5]}{'...' if len(all_miss_feats) > 5 else ''}")
else:
    log("  OK No _is_missing indicators were selected in any run.")

log("  Feature frequency tables saved.")


# ==========================================
# 7. Final Reporting
# ==========================================
log("\nGenerating Final Reports...")
df_res = pd.DataFrame(results)
df_preds = pd.DataFrame(prediction_history)

if df_res.empty:
    log("CRITICAL: No Results.")
else:
    report_cols = [c for c in df_res.columns if any(x in c for x in ['train_', 'val_', 'test_'])]

    def robust_agg(x):
        clean_x = [v for v in x if v is not None and not np.isnan(v)]
        if not clean_x: return "N/A"
        if len(clean_x) == 1: return f"{clean_x[0]:.3f} (0.000)"
        return f"{np.mean(clean_x):.3f} ({np.std(clean_x, ddof=1):.3f})"

    grouped = df_res.groupby(['Algorithm', 'Modality'])
    agg_df = grouped[report_cols].apply(lambda g: g.apply(robust_agg))
    meta_df = grouped[['N_Genes', 'N_Bio_Features', 'N_Missing_Indicators',
                        'Selected_Gene_Symbol', 'Selected_Gene_Ensembl',
                        'Key_Params', 'LASSO_C_Info']].first()
    final_table = pd.concat([agg_df, meta_df], axis=1)
    final_table.to_csv(os.path.join(OUTPUT_DIR, "Table1_Expanded_Metrics.csv"))
    df_res.to_csv(os.path.join(OUTPUT_DIR, "Table1_PerFold_Raw.csv"), index=False)

    pivot = df_res.pivot_table(index='Algorithm', columns='Modality', values='test_roc', aggfunc='mean')
    pivot_cols = [c for c in pivot.columns if c in [
        'EV_Genecode', 'EV_circRNA', 'EV_Combined', 'WB_Genecode', 'WB_circRNA', 'WB_Combined',
        'LateFusion_Genecode', 'LateFusion_Combined'
    ]]
    pivot['Total_Score'] = pivot[pivot_cols].sum(axis=1)
    algo_order = pivot.sort_values('Total_Score', ascending=False).index.tolist()
    best_algo = algo_order[0]
    log(f"Best Algo for Plots & SHAP: {best_algo}")

    df_preds.to_csv(os.path.join(OUTPUT_DIR, "Table2_Predict_Metrics.csv"), index=False)

    colors = {
        'EV_Genecode': 'green', 'EV_circRNA': 'lightgreen', 'EV_Combined': 'darkgreen',
        'WB_Genecode': 'blue', 'WB_circRNA': 'lightblue', 'WB_Combined': 'darkblue',
        'LateFusion_Genecode': 'red', 'LateFusion_Combined': 'darkred'
    }
    all_mods = list(colors.keys())

    # Figure 1: Benchmark
    plt.figure(figsize=(20, 10))
    sns.boxplot(data=df_res, x='Algorithm', y='test_roc', hue='Modality', order=algo_order, palette='Set3')
    plt.title(f'Benchmark ({N_OUTER_FOLDS}-Fold Nested CV, Fixed Lasso C={FIXED_LASSO_C})')
    plt.ylabel('Test AUROC'); plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "Figure_1_Benchmark.pdf")); plt.close()

    # Figure 2: Mean ROC/PRC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    mean_fpr = np.linspace(0, 1, 100)
    for mod in all_mods:
        key = (best_algo, mod)
        if key in curve_data_storage:
            runs = curve_data_storage[key]
            tprs, aucs, y_real, y_proba = [], [], [], []
            for r in runs:
                fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
                interp_tpr = np.interp(mean_fpr, fpr, tpr); interp_tpr[0] = 0.0
                tprs.append(interp_tpr); aucs.append(r['auc'])
                y_real.extend(r['y_true']); y_proba.extend(r['y_prob'])
            mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr); std_auc = np.std(aucs)
            ax1.plot(mean_fpr, mean_tpr, label=f"{mod} AUC={mean_auc:.3f} +/-{std_auc:.3f}", color=colors.get(mod, 'k'), lw=2)
            prec, rec, _ = precision_recall_curve(y_real, y_proba)
            pr_auc = average_precision_score(y_real, y_proba)
            ax2.plot(rec, prec, label=f"{mod} AUPRC={pr_auc:.3f}", color=colors.get(mod, 'k'), lw=2)

    ax1.plot([0, 1], [0, 1], 'k--'); ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.legend(loc='lower right', fontsize=8); ax1.set_title(f'Mean ROC ({best_algo})')
    ax2.plot([0, 1], [0.5, 0.5], 'k--'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.legend(loc='lower left', fontsize=8); ax2.set_title(f'Aggregate PRC ({best_algo})')
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_ROC_PRC.pdf")); plt.close()

    # Figure 3: Calibration & DCA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    dca_y_max = 0.05
    any_key = list(curve_data_storage.keys())[0]
    y_true_sample = np.concatenate([r['y_true'] for r in curve_data_storage[any_key]])
    thresholds = np.linspace(0.01, 0.99, 100)
    for mod in all_mods:
        key = (best_algo, mod)
        if key in curve_data_storage:
            runs = curve_data_storage[key]
            y_true_all = np.concatenate([r['y_true'] for r in runs])
            y_prob_all = np.concatenate([r['y_prob'] for r in runs])
            color = colors.get(mod, 'k')
            prob_true, prob_pred = calibration_curve(y_true_all, y_prob_all, n_bins=10, strategy='uniform')
            ax1.plot(prob_pred, prob_true, marker='o', label=mod, color=color, alpha=0.8)
            net_benefit = get_net_benefit(y_true_all, y_prob_all, thresholds)
            ax2.plot(thresholds, net_benefit, label=mod, color=color)
            if max(net_benefit) > dca_y_max: dca_y_max = max(net_benefit)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5); ax1.set_xlabel('Mean Predicted Prob')
    ax1.set_ylabel('Fraction Positives'); ax1.set_title(f'Calibration ({best_algo})'); ax1.legend(fontsize=7)
    net_benefit_all = get_net_benefit(y_true_sample, np.ones_like(y_true_sample), thresholds)
    ax2.plot(thresholds, net_benefit_all, color='gray', ls='--', label='Treat All')
    ax2.axhline(0, color='black', lw=1, label='Treat None')
    ax2.set_xlabel('Threshold'); ax2.set_ylabel('Net Benefit'); ax2.set_ylim(-0.05, dca_y_max * 1.2)
    ax2.set_title(f'DCA ({best_algo})'); ax2.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "Figure_3_Calibration_DCA.pdf")); plt.close()

    # Figure 4 & 5: Scatter & Confusion
    plot_modes = ['EV_Genecode', 'EV_Combined', 'WB_Genecode', 'WB_Combined', 'LateFusion_Genecode', 'LateFusion_Combined']
    fig_sc = plt.figure(figsize=(18, 12))
    outer_grid = matplotlib.gridspec.GridSpec(2, 3, wspace=0.25, hspace=0.3)
    fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 10)); axes_cm = axes_cm.flatten()
    rng = np.random.RandomState(SEED)
    cmap = matplotlib.cm.get_cmap('bwr'); color_ctrl = cmap(0.0); color_pd = cmap(1.0)

    for i, mod in enumerate(plot_modes):
        inner_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[i], width_ratios=[4, 1], wspace=0.05)
        ax_main = plt.Subplot(fig_sc, inner_grid[0]); ax_right = plt.Subplot(fig_sc, inner_grid[1])
        fig_sc.add_subplot(ax_main); fig_sc.add_subplot(ax_right)
        subset = df_preds[(df_preds['Algorithm'] == best_algo) & (df_preds['Modality'] == mod)]
        if not subset.empty:
            subset = subset.sort_values('Prob_PD')
            x_vals = np.arange(len(subset))
            x_jitter = x_vals + rng.uniform(-0.45, 0.45, size=len(x_vals))
            ax_main.scatter(x_jitter, subset['Prob_PD'], c=subset['True_Label'], cmap='bwr', s=6, alpha=0.6, edgecolors='none')
            ax_main.axhline(0.5, color='gray', linestyle='--'); ax_main.set_title(f'{mod}'); ax_main.set_ylim(-0.05, 1.05)
            if i >= 3: ax_main.set_xlabel('Sorted Sample Index')
            if i % 3 == 0: ax_main.set_ylabel('Predicted Probability')
            ax_right.set_ylim(ax_main.get_ylim()); ax_right.axis('off')
            y_grid = np.linspace(0, 1, 200)
            for label_val, color_val in [(0, color_ctrl), (1, color_pd)]:
                prob_sub = subset[subset['True_Label'] == label_val]['Prob_PD']
                if len(prob_sub) > 1:
                    kde = gaussian_kde(prob_sub); x_kde = kde(y_grid)
                    ax_right.plot(x_kde, y_grid, color=color_val, lw=1.5)
                    ax_right.fill_betweenx(y_grid, 0, x_kde, color=color_val, alpha=0.3)
            ax_right.axhline(0.5, color='gray', linestyle='--')
            cm_total = confusion_matrix(subset['True_Label'], subset['Predicted_Label'])
            ConfusionMatrixDisplay(confusion_matrix=cm_total, display_labels=['CTRL', 'PD']).plot(cmap='Blues', ax=axes_cm[i], values_format='d', colorbar=False)
            axes_cm[i].set_title(f'{mod}')

    fig_sc.suptitle(f'Prediction Distributions ({best_algo})', fontsize=14, y=0.95)
    from matplotlib.lines import Line2D
    fig_sc.legend(handles=[Line2D([0],[0],marker='o',color='w',label='Control',markerfacecolor=color_ctrl,markersize=8),
                           Line2D([0],[0],marker='o',color='w',label='PD',markerfacecolor=color_pd,markersize=8)],
                  loc='upper right', bbox_to_anchor=(0.98, 0.92))
    plt.figure(fig_sc.number); plt.savefig(os.path.join(OUTPUT_DIR, "Figure_4_Prediction_Scatter.pdf"), bbox_inches='tight'); plt.close(fig_sc)
    fig_cm.suptitle(f'Confusion Matrices ({best_algo})', fontsize=14)
    plt.figure(fig_cm.number); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(OUTPUT_DIR, "Figure_5_Confusion_Matrix.pdf")); plt.close(fig_cm)

    # PCA: Fit on TRAIN only, transform test
    for mod in ['EV_Genecode', 'EV_circRNA', 'EV_Combined', 'WB_Genecode', 'WB_circRNA', 'WB_Combined']:
        pca_data = best_pca_data.get(mod)
        if pca_data:
            X_tr, X_te = pca_data['X_tr'], pca_data['X_te']
            y_tr_p, y_te_p = pca_data['y_tr'], pca_data['y_te']
            pca = PCA(n_components=2, random_state=SEED)
            pca.fit(X_tr)
            X_tr_pca = pca.transform(X_tr)
            X_te_pca = pca.transform(X_te)

            plt.figure(figsize=(8, 8))
            plt.scatter(X_tr_pca[y_tr_p == 0, 0], X_tr_pca[y_tr_p == 0, 1], c='lightblue', marker='o', alpha=0.5, label='Train CTRL')
            plt.scatter(X_tr_pca[y_tr_p == 1, 0], X_tr_pca[y_tr_p == 1, 1], c='salmon', marker='o', alpha=0.5, label='Train PD')
            plt.scatter(X_te_pca[y_te_p == 0, 0], X_te_pca[y_te_p == 0, 1], c='blue', marker='x', alpha=0.8, label='Test CTRL')
            plt.scatter(X_te_pca[y_te_p == 1, 0], X_te_pca[y_te_p == 1, 1], c='red', marker='x', alpha=0.8, label='Test PD')
            ev1, ev2 = pca.explained_variance_ratio_[0]*100, pca.explained_variance_ratio_[1]*100
            plt.xlabel(f'PC1 ({ev1:.1f}%)'); plt.ylabel(f'PC2 ({ev2:.1f}%)'); plt.title(f'PCA: {mod} ({best_algo})'); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, f"Figure_6_PCA_{mod.replace('+','_')}.pdf")); plt.close()


    # ==========================================
    # 8.  FIXED: Scientific SHAP Analysis
    #    - Fixed to best_algo (unified across all modalities)
    #    - Aggregated across ALL outer folds (no cherry-picking)
    #    - Reports uncertainty (std across folds)
    #    - Reports selection frequency
    #    - Outputs detailed TSV for supplementary tables
    # ==========================================

    log("\n" + "=" * 60)
    log(" SCIENTIFIC SHAP ANALYSIS")
    log(f"  Algorithm: {best_algo} (fixed, determined by benchmark)")
    log(f"  Aggregation: across all {N_OUTER_FOLDS} outer folds")
    log("=" * 60)

    def compute_shap_for_one_fold(fold_data, shap_bg_samples, shap_max_test, shap_nsamples):
        """Compute SHAP values for a single fold's classifier and test data."""
        clf = fold_data['clf']
        X_tr = fold_data['X_tr_sel']
        X_te = fold_data['X_te_sel']
        seed = fold_data['seed']

        # Determine explainer type
        needs_kernel = not (hasattr(clf, 'feature_importances_') or
                           (hasattr(clf, 'coef_') and not isinstance(clf, SVC)))

        # Subsample test set for KernelExplainer (slow)
        if needs_kernel and X_te.shape[0] > shap_max_test:
            rng = np.random.RandomState(seed)
            idx = rng.choice(X_te.shape[0], shap_max_test, replace=False)
            X_shap = X_te[idx]
        else:
            X_shap = X_te

        # Compute SHAP values
        if hasattr(clf, 'feature_importances_'):
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_shap)
        elif hasattr(clf, 'coef_') and not isinstance(clf, SVC):
            explainer = shap.LinearExplainer(clf, X_tr)
            shap_vals = explainer.shap_values(X_shap)
        else:
            bg = resample(X_tr, n_samples=shap_bg_samples, random_state=seed)
            pred_fn = clf.predict_proba if hasattr(clf, 'predict_proba') else clf.decision_function
            explainer = shap.KernelExplainer(pred_fn, bg)
            shap_vals = explainer.shap_values(X_shap, nsamples=shap_nsamples, silent=True)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

        # Normalize output format
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        if hasattr(shap_vals, 'values'):
            shap_vals = shap_vals.values
        shap_vals = np.array(shap_vals)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        return shap_vals, X_shap

    def aggregate_shap_across_folds(fold_data_list):
        """
        Aggregate SHAP values across folds.
        Different folds may select different features -> use union + zero-padding.
        """
        # Step 1: Build unified feature space (union of all folds)
        all_symbols = []
        for fd in fold_data_list:
            all_symbols.extend(fd['symbols'])
        unique_symbols = sorted(set(all_symbols))
        symbol_to_idx = {s: i for i, s in enumerate(unique_symbols)}
        n_total_features = len(unique_symbols)

        # Also build Ensembl ID mapping
        symbol_to_ensembl = {}
        for fd in fold_data_list:
            for feat, sym in zip(fd['features'], fd['symbols']):
                if sym not in symbol_to_ensembl:
                    symbol_to_ensembl[sym] = feat

        # Step 2: Map each fold's SHAP to unified space
        fold_mean_abs_shap = []   # (n_folds, n_total_features)
        fold_mean_signed_shap = []  # For direction info
        fold_test_aucs = []

        for fd in fold_data_list:
            shap_vals = fd['shap_vals']  # (n_samples, n_selected_features)
            symbols = fd['symbols']

            mean_abs = np.abs(shap_vals).mean(axis=0)
            mean_signed = shap_vals.mean(axis=0)

            unified_abs = np.zeros(n_total_features)
            unified_signed = np.zeros(n_total_features)
            for j, sym in enumerate(symbols):
                idx = symbol_to_idx[sym]
                unified_abs[idx] = mean_abs[j]
                unified_signed[idx] = mean_signed[j]

            fold_mean_abs_shap.append(unified_abs)
            fold_mean_signed_shap.append(unified_signed)
            fold_test_aucs.append(fd['test_auc'])

        fold_mean_abs_shap = np.array(fold_mean_abs_shap)
        fold_mean_signed_shap = np.array(fold_mean_signed_shap)

        # Step 3: Aggregate across folds
        # mean across ALL folds (unselected features contribute 0 -> lower importance)
        agg_mean_abs = fold_mean_abs_shap.mean(axis=0)
        agg_std_abs = fold_mean_abs_shap.std(axis=0, ddof=1) if len(fold_data_list) > 1 else np.zeros(n_total_features)
        agg_mean_signed = fold_mean_signed_shap.mean(axis=0)

        # Selection count and frequency
        selection_count = (fold_mean_abs_shap > 0).sum(axis=0)
        selection_freq = selection_count / len(fold_data_list)

        # Conditional mean (average only over folds where the feature was selected)
        with np.errstate(divide='ignore', invalid='ignore'):
            agg_conditional_abs = np.where(
                selection_count > 0,
                fold_mean_abs_shap.sum(axis=0) / selection_count,
                0.0
            )

        # Build Ensembl IDs in same order
        ensembl_ids = [symbol_to_ensembl.get(s, 'NA') for s in unique_symbols]

        return {
            'symbols': unique_symbols,
            'ensembl_ids': ensembl_ids,
            'mean_abs_shap': agg_mean_abs,
            'std_abs_shap': agg_std_abs,
            'mean_signed_shap': agg_mean_signed,
            'conditional_mean_abs_shap': agg_conditional_abs,
            'selection_count': selection_count.astype(int),
            'selection_frequency': selection_freq,
            'per_fold_abs_shap': fold_mean_abs_shap,
            'per_fold_signed_shap': fold_mean_signed_shap,
            'per_fold_aucs': fold_test_aucs,
            'n_folds': len(fold_data_list),
        }

    def save_shap_tsv(agg_result, mod_name, algo_name, output_dir):
        """
         Save detailed SHAP results as TSV for supplementary table.
        Columns: Feature_Symbol, Feature_ID, Mean_Abs_SHAP, SD_Abs_SHAP,
                 Mean_Signed_SHAP, Selection_Frequency, N_Folds_Selected,
                 Conditional_Mean_Abs_SHAP, Per_Fold_SHAP (Fold1, Fold2, ...)
        """
        n_folds = agg_result['n_folds']
        per_fold = agg_result['per_fold_abs_shap']

        df = pd.DataFrame({
            'Feature_Symbol': agg_result['symbols'],
            'Feature_ID': agg_result['ensembl_ids'],
            'Mean_Abs_SHAP': agg_result['mean_abs_shap'],
            'SD_Abs_SHAP': agg_result['std_abs_shap'],
            'Mean_Signed_SHAP': agg_result['mean_signed_shap'],
            'Conditional_Mean_Abs_SHAP': agg_result['conditional_mean_abs_shap'],
            'Selection_Frequency': agg_result['selection_frequency'],
            'N_Folds_Selected': agg_result['selection_count'],
            'Is_Missing_Indicator': [s.endswith('_is_missing') for s in agg_result['symbols']],
        })

        # Add per-fold columns
        for f_i in range(n_folds):
            df[f'Fold{f_i+1}_MeanAbsSHAP'] = per_fold[f_i]

        # Add per-fold signed SHAP
        per_fold_signed = agg_result['per_fold_signed_shap']
        for f_i in range(n_folds):
            df[f'Fold{f_i+1}_MeanSignedSHAP'] = per_fold_signed[f_i]

        # Sort by importance
        df = df.sort_values('Mean_Abs_SHAP', ascending=False)

        # Add rank
        df.insert(0, 'Rank', range(1, len(df) + 1))

        # Add metadata
        df['Algorithm'] = algo_name
        df['Modality'] = mod_name
        df['N_CV_Folds'] = n_folds

        # Save
        out_path = os.path.join(output_dir, f"SHAP_Table_{mod_name}_{algo_name}.tsv")
        df.to_csv(out_path, sep='\t', index=False, float_format='%.6f')
        log(f"    Saved: {out_path} ({len(df)} features)")

        return df

    def plot_shap_aggregated_bar(agg_result, mod_name, algo_name, output_dir, top_n=20):
        """
         Aggregated bar plot with error bars and selection frequency annotations.
        """
        symbols = agg_result['symbols']
        mean_shap = agg_result['mean_abs_shap']
        std_shap = agg_result['std_abs_shap']
        sel_freq = agg_result['selection_frequency']
        n_folds = agg_result['n_folds']

        # Sort and take top N
        sort_idx = np.argsort(mean_shap)[::-1][:top_n]
        top_symbols = [symbols[i] for i in sort_idx]
        top_mean = mean_shap[sort_idx]
        top_std = std_shap[sort_idx]
        top_freq = sel_freq[sort_idx]

        n_plot = len(top_symbols)
        fig_height = max(6, n_plot * 0.35)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        y_pos = np.arange(n_plot)
        bars = ax.barh(
            y_pos,
            top_mean[::-1],
            xerr=top_std[::-1],
            color='steelblue',
            edgecolor='black',
            linewidth=0.5,
            capsize=3,
            alpha=0.85
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_symbols[::-1])
        ax.set_xlabel('Mean |SHAP value| (+/- SD across folds)')
        ax.set_title(f'Aggregated Feature Importance - {mod_name}\n'
                     f'({algo_name}, {n_folds}-fold CV)')

        # Annotate selection frequency
        for j, (val, sd, freq) in enumerate(zip(top_mean[::-1], top_std[::-1], top_freq[::-1])):
            x_pos = val + sd + 0.0005
            freq_pct = f'{freq:.0%}'
            ax.text(x_pos, j, freq_pct, va='center', fontsize=8, color='gray')

        ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"Figure_SHAP_{mod_name}_{algo_name}_AggregatedBar.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        log(f"    Saved: {os.path.basename(out_path)}")

    def plot_shap_stability_heatmap(agg_result, mod_name, algo_name, output_dir, top_n=20):
        """
         Per-fold SHAP heatmap showing stability across folds.
        """
        symbols = agg_result['symbols']
        mean_shap = agg_result['mean_abs_shap']
        per_fold = agg_result['per_fold_abs_shap']  # (n_folds, n_features)
        n_folds = agg_result['n_folds']

        # Top N features
        sort_idx = np.argsort(mean_shap)[::-1][:top_n]
        top_symbols = [symbols[i] for i in sort_idx]
        per_fold_top = per_fold[:, sort_idx]  # (n_folds, top_n)

        fig, ax = plt.subplots(figsize=(max(8, n_folds * 1.2), max(6, top_n * 0.35)))
        sns.heatmap(
            per_fold_top.T,
            xticklabels=[f'Fold {i+1}' for i in range(n_folds)],
            yticklabels=top_symbols,
            cmap='YlOrRd',
            ax=ax,
            annot=True if top_n <= 20 else False,
            fmt='.3f' if top_n <= 20 else '',
            linewidths=0.5
        )
        ax.set_title(f'Per-Fold SHAP Stability - {mod_name}\n({algo_name}, {n_folds}-fold CV)')
        ax.set_xlabel('CV Fold')
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"Figure_SHAP_{mod_name}_{algo_name}_StabilityHeatmap.pdf")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        log(f"    Saved: {os.path.basename(out_path)}")

    def plot_shap_beeswarm_best_fold(shap_fold_collector_key, mod_name, algo_name, output_dir):
        """
         Beeswarm plot from the median-AUC fold (representative, not cherry-picked best).
        Uses median fold to avoid selection bias.
        """
        fold_list = shap_fold_collector_key
        aucs = [fd['test_auc'] for fd in fold_list]
        # Use median fold (most representative, not best)
        median_idx = np.argsort(aucs)[len(aucs) // 2]
        fd = fold_list[median_idx]

        if fd.get('shap_vals') is None:
            return

        shap_vals = fd['shap_vals']
        symbols = fd['symbols']
        X_shap = fd.get('X_shap', fd['X_te_sel'])

        n_features = shap_vals.shape[1]
        fig_height = max(6, n_features * 0.25)

        # Build DataFrame with proper column names for beeswarm
        X_df = pd.DataFrame(X_shap[:, :n_features] if X_shap.shape[1] >= n_features else X_shap,
                            columns=symbols[:X_shap.shape[1]])

        plt.figure(figsize=(12, fig_height))
        try:
            shap.summary_plot(
                shap_vals, X_df, show=False,
                max_display=min(30, n_features), plot_size=None
            )
            plt.title(f'SHAP Beeswarm - {mod_name}\n'
                      f'({algo_name}, Fold {fd["fold"]+1}, '
                      f'AUC={fd["test_auc"]:.3f}, median fold)')
            plt.tight_layout()
            out_path = os.path.join(output_dir,
                                    f"Figure_SHAP_{mod_name}_{algo_name}_Beeswarm_MedianFold.pdf")
            plt.savefig(out_path, bbox_inches='tight')
            log(f"    Saved: {os.path.basename(out_path)}")
        except Exception as e:
            log(f"    Beeswarm failed: {e}")
        finally:
            plt.close()


    # -- Run SHAP for each modality using best_algo --
    shap_modalities = ['EV_Genecode', 'EV_circRNA', 'EV_Combined',
                       'WB_Genecode', 'WB_circRNA', 'WB_Combined']

    all_shap_tables = {}  # Store for cross-modality comparison

    for mod in shap_modalities:
        key = (best_algo, mod)
        if key not in shap_fold_collector or not shap_fold_collector[key]:
            log(f"  Skipping SHAP for {mod}: no data for {best_algo}")
            continue

        fold_list = shap_fold_collector[key]
        log(f"\n  Computing SHAP: {mod} ({best_algo}, {len(fold_list)} folds)...")

        # Step 1: Compute SHAP for each fold
        n_success = 0
        for fd in fold_list:
            try:
                np.random.seed(fd['seed'])  #  Reset RNG for reproducibility
                shap_vals, X_shap = compute_shap_for_one_fold(
                    fd, SHAP_BG_SAMPLES, SHAP_MAX_TEST, SHAP_NSAMPLES
                )
                fd['shap_vals'] = shap_vals
                fd['X_shap'] = X_shap
                n_success += 1
                log(f"    Fold {fd['fold']+1}: OK, shape={shap_vals.shape}, "
                    f"{len(fd['symbols'])} features, AUC={fd['test_auc']:.3f}")
            except Exception as e:
                log(f"    Fold {fd['fold']+1}: FAILED - {e}")
                traceback.print_exc()
                fd['shap_vals'] = None

        # Filter successful folds
        valid_folds = [fd for fd in fold_list if fd.get('shap_vals') is not None]
        if not valid_folds:
            log(f"  Skipping {mod}: all SHAP computations failed")
            continue

        log(f"    {n_success}/{len(fold_list)} folds computed successfully")

        # Step 2: Aggregate across folds
        agg_result = aggregate_shap_across_folds(valid_folds)

        # Step 3: Save detailed TSV ( for supplementary table)
        df_shap = save_shap_tsv(agg_result, mod, best_algo, OUTPUT_DIR)
        all_shap_tables[mod] = df_shap

        # Step 4: Generate plots
        plot_shap_aggregated_bar(agg_result, mod, best_algo, OUTPUT_DIR, top_n=20)
        plot_shap_stability_heatmap(agg_result, mod, best_algo, OUTPUT_DIR, top_n=20)

        # Step 5: Beeswarm from median fold
        plot_shap_beeswarm_best_fold(valid_folds, mod, best_algo, OUTPUT_DIR)

        # Step 6: Separate annotated vs novel gene plots
        annotated_mask = [not (s.startswith('ENSG') or ((':' in s) and s.split(':')[-1].startswith('ENSG')))
                         for s in agg_result['symbols']]
        novel_mask = [not m for m in annotated_mask]

        if any(annotated_mask):
            ann_idx = [i for i, m in enumerate(annotated_mask) if m]
            ann_result = {
                'symbols': [agg_result['symbols'][i] for i in ann_idx],
                'mean_abs_shap': agg_result['mean_abs_shap'][ann_idx],
                'std_abs_shap': agg_result['std_abs_shap'][ann_idx],
                'selection_frequency': agg_result['selection_frequency'][ann_idx],
                'n_folds': agg_result['n_folds']
            }
            plot_shap_aggregated_bar(ann_result, f"{mod}_Annotated", best_algo, OUTPUT_DIR, top_n=20)

        if any(novel_mask):
            nov_idx = [i for i, m in enumerate(novel_mask) if m]
            nov_result = {
                'symbols': [agg_result['symbols'][i] for i in nov_idx],
                'mean_abs_shap': agg_result['mean_abs_shap'][nov_idx],
                'std_abs_shap': agg_result['std_abs_shap'][nov_idx],
                'selection_frequency': agg_result['selection_frequency'][nov_idx],
                'n_folds': agg_result['n_folds']
            }
            plot_shap_aggregated_bar(nov_result, f"{mod}_Novel", best_algo, OUTPUT_DIR, top_n=20)

    # -- Cross-modality comparison table --
    if all_shap_tables:
        log("\n  Generating cross-modality SHAP comparison...")
        comparison_rows = []
        for mod, df in all_shap_tables.items():
            for _, row in df.iterrows():
                comparison_rows.append({
                    'Modality': mod,
                    'Rank_in_Modality': row['Rank'],
                    'Feature_Symbol': row['Feature_Symbol'],
                    'Feature_ID': row['Feature_ID'],
                    'Mean_Abs_SHAP': row['Mean_Abs_SHAP'],
                    'SD_Abs_SHAP': row['SD_Abs_SHAP'],
                    'Mean_Signed_SHAP': row['Mean_Signed_SHAP'],
                    'Selection_Frequency': row['Selection_Frequency'],
                    'N_Folds_Selected': row['N_Folds_Selected'],
                    'Is_Missing_Indicator': row['Is_Missing_Indicator'],
                })
        df_cross = pd.DataFrame(comparison_rows)
        cross_path = os.path.join(OUTPUT_DIR, f"SHAP_CrossModality_Comparison_{best_algo}.tsv")
        df_cross.to_csv(cross_path, sep='\t', index=False, float_format='%.6f')
        log(f"    Saved: {cross_path} ({len(df_cross)} rows)")

        # Summary: top 5 per modality
        log("\n   Top 5 features per modality:")
        for mod, df in all_shap_tables.items():
            top5 = df.head(5)
            feats = ', '.join(f"{r['Feature_Symbol']}({r['Selection_Frequency']:.0%})" for _, r in top5.iterrows())
            log(f"    {mod}: {feats}")

    log("\n   SHAP analysis complete.")


log("\n" + "=" * 60)
log(" SUMMARY - Fixed LASSO Pipeline + Scientific SHAP")
log("=" * 60)
log(f"Samples: {len(y)} | CV: {N_OUTER_FOLDS}-Fold Nested")
log(f"Pipeline: Imputer(median) -> IndicatorFilter(allow={ALLOW_MISSING_INDICATORS}) "
    f"-> Scaler -> SimpleLassoSelector(C={FIXED_LASSO_C}) -> Classifier")
log(f" Feature Selection: Standard LogisticRegression L1, C={FIXED_LASSO_C}")
log(f" SHAP: Fixed to best_algo, aggregated across ALL {N_OUTER_FOLDS} outer folds")
log(f" SHAP Outputs: TSV tables + Aggregated bars + Stability heatmaps + Beeswarm (median fold)")
log(f"Grid search: Manual group-aware implementation")
log(f"Output: {OUTPUT_DIR}")
log("=" * 60)
log("Done.")
