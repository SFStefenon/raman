"""Patient-level benchmark: Flow + ANOVA256 + PLS-DA vs real-only classifiers.

Proposed:
    SNV -> class-conditional Latent Flow augmentation (training fold only)
    -> ANOVA top-256 wavelengths -> PLS-DA.
Baselines:
    Original real patients only -> full spectral channels -> classifier.
    No synthetic samples, ANOVA, feature selection, or feature construction.
    By default they share the same SNV normalization as the proposed pipeline.
    Set BASELINE_SNV=False for a strictly unnormalized raw-spectrum comparison.

Outer evaluation: leave-one-original-patient-out (LOOCV). No held-out
patient contributes to Flow, ANOVA, preprocessing fit, or classifier training.
Every prediction is made for a REAL patient. This is exploratory, because
ANOVA256 + PLS-DA was selected previously on this same cohort; external
validation on unseen patients is needed before clinical claims.

Install:
    pip install numpy pandas scipy scikit-learn torch matplotlib xgboost
Place saudaveis.zip and doentes.zip alongside this file and run:
    python flow_anova256_vs_raw_benchmark.py
"""
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F
import torch

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

ROOT = Path(globals().get('ROOT', Path(__file__).resolve().parent))
HEALTHY_ZIP = Path(globals().get('HEALTHY_ZIP', ROOT / 'saudaveis.zip'))
SICK_ZIP = Path(globals().get('SICK_ZIP', ROOT / 'doentes.zip'))
OUTPUT_DIR = Path(globals().get('OUTPUT_DIR', ROOT / 'flow_anova256_vs_raw_results'))
SEED = int(globals().get('SEED', 0))
ANOVA_K = int(globals().get('ANOVA_K', 256))
BASELINE_SNV = bool(globals().get('BASELINE_SNV', True))
LATENT_DIM = int(globals().get('LATENT_DIM', 4))
FLOW_TRAIN_STEPS = int(globals().get('FLOW_TRAIN_STEPS', 350))
FLOW_BATCH_SIZE = int(globals().get('FLOW_BATCH_SIZE', 64))
FLOW_ODE_STEPS = int(globals().get('FLOW_ODE_STEPS', 32))
SYNTHETIC_PER_CLASS = int(globals().get('SYNTHETIC_PER_CLASS', 30))
NEURAL_EPOCHS = int(globals().get('NEURAL_EPOCHS', 35))
NEURAL_BATCH_SIZE = int(globals().get('NEURAL_BATCH_SIZE', 32))
BOOTSTRAP_REPS = int(globals().get('BOOTSTRAP_REPS', 3000))
INCLUDE_NEURAL = bool(globals().get('INCLUDE_NEURAL', True))
INCLUDE_XGBOOST = bool(globals().get('INCLUDE_XGBOOST', True))
DEBUG_FOLDS = globals().get('DEBUG_FOLDS', None)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSIFIERS = globals().get('CLASSIFIERS', [
    'PLS-DA', 'Linear SVM', 'RBF SVM', 'Elastic-net LR', 'Random Forest'
] + (['XGBoost'] if INCLUDE_XGBOOST else [])
  + (['SE-ResNet1D', 'Patch Transformer'] if INCLUDE_NEURAL else []))
PROPOSED = 'Flow + ANOVA256 + PLS-DA'
BASELINE_NAMES = [f'{c}' for c in CLASSIFIERS]
PIPELINES = [PROPOSED] + BASELINE_NAMES
NEURAL_MODELS = ('SE-ResNet1D', 'Patch Transformer')

if not HAVE_XGBOOST and 'XGBoost' in CLASSIFIERS:
    raise ImportError('XGBoost requested: pip install xgboost or set INCLUDE_XGBOOST=False')
if 'PLS-DA' not in CLASSIFIERS:
    raise ValueError('Keep PLS-DA as a real-only control for the proposed PLS-DA pipeline')
if len(CLASSIFIERS) != len(set(CLASSIFIERS)):
    raise ValueError('Repeated classifier names')
if ANOVA_K < 1 or SYNTHETIC_PER_CLASS < 1:
    raise ValueError('ANOVA_K and SYNTHETIC_PER_CLASS must be positive')
if BOOTSTRAP_REPS < 100:
    raise ValueError('Set BOOTSTRAP_REPS >= 100')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(min(3, torch.get_num_threads()))
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'savefig.facecolor': 'white', 'axes.grid': False,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 9, 'pdf.fonttype': 42})

def load_original_spectra():
    rows = []
    for zip_path, cls in [(HEALTHY_ZIP, 0), (SICK_ZIP, 1)]:
        if not zip_path.is_file():
            raise FileNotFoundError(f'Missing archive: {zip_path}')
        with ZipFile(zip_path) as archive:
            for path in sorted(archive.namelist()):
                path_l = path.lower()
                if (not path_l.endswith('.csv') or '.ipynb_checkpoints' in path_l
                        or '__macosx' in path_l):
                    continue
                raw = archive.read(path)
                first_line = raw.splitlines()[0]
                delimiter = ';' if first_line.count(b';') > first_line.count(b',') else ','
                df = pd.read_csv(BytesIO(raw), sep=delimiter, encoding='utf-8-sig')
                if df.shape[1] != 2:
                    raise ValueError(f'Expected two columns in {path}')
                w = pd.to_numeric(df.iloc[:, 0], errors='raise').to_numpy(float)
                x = pd.to_numeric(df.iloc[:, 1], errors='raise').to_numpy(float)
                if not (np.isfinite(w).all() and np.isfinite(x).all()):
                    raise ValueError(f'Nonfinite data in {path}')
                rows.append((Path(path).stem, cls, w, x))
    if not rows:
        raise ValueError('No usable CSV spectra found')
    wn = rows[0][2]
    for name, _, w, x in rows:
        if len(w) != len(wn) or not np.allclose(wn, w, atol=1e-6, rtol=0):
            raise ValueError(f'Spectral grid differs: {name}')
    names = np.array([r[0] for r in rows], dtype=str)
    if len(names) != len(set(names)):
        raise ValueError('Duplicate original patient identifiers')
    X = np.asarray([r[3] for r in rows], dtype=np.float64)
    y = np.asarray([r[1] for r in rows], dtype=int)
    if min(np.bincount(y, minlength=2)) < 3:
        raise ValueError('At least three original patients per class are required')
    return wn, X, y, names


def snv(X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return (X - X.mean(axis=1, keepdims=True)) / np.maximum(X.std(axis=1, keepdims=True), 1e-10)


class ConditionalLatentFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim + 7, 64), nn.SiLU(), nn.Linear(64, 64),
            nn.SiLU(), nn.Linear(64, dim))

    def forward(self, x, label, time):
        tau = 2 * np.pi * time
        t_emb = [time, torch.sin(tau), torch.cos(tau),
                 torch.sin(2 * tau), torch.cos(2 * tau)]
        condition = F.one_hot(label.long(), 2).float()
        return self.network(torch.cat([x, condition, *t_emb], dim=1))


def fit_and_generate_flow(X_train, y_train, seed):
    """Fit generator using ONLY the original training patients of this outer fold."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    real = snv(X_train)
    dim = min(LATENT_DIM, len(X_train) - 2, real.shape[1])
    pca = PCA(n_components=dim, random_state=seed)
    scaler = StandardScaler()
    latent = scaler.fit_transform(pca.fit_transform(real)).astype(np.float32)
    data = torch.as_tensor(latent, device=DEVICE)
    labels = torch.as_tensor(y_train, dtype=torch.long, device=DEVICE)
    class_ids = [torch.where(labels == c)[0] for c in (0, 1)]
    model = ConditionalLatentFlow(dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(FLOW_TRAIN_STEPS):
        c = torch.randint(0, 2, (FLOW_BATCH_SIZE,), device=DEVICE)
        idx = torch.empty_like(c)
        for class_id in (0, 1):
            mask = (c == class_id)
            idx[mask] = class_ids[class_id][torch.randint(len(class_ids[class_id]),
                                                       (int(mask.sum()),), device=DEVICE)]
        x1 = data[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand((len(c), 1), device=DEVICE)
        xt = (1 - t) * x0 + t * x1
        loss = F.mse_loss(model(xt, c, t), x1 - x0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    y_syn = np.repeat([0, 1], SYNTHETIC_PER_CLASS).astype(int)
    c_syn = torch.as_tensor(y_syn, dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        x = torch.randn((len(y_syn), dim), device=DEVICE)
        dt = 1.0 / FLOW_ODE_STEPS
        for i in range(FLOW_ODE_STEPS):
            t0 = torch.full((len(y_syn), 1), i * dt, device=DEVICE)
            t1 = torch.full((len(y_syn), 1), (i+1) * dt, device=DEVICE)
            v0 = model(x, c_syn, t0)
            v1 = model(x + dt*v0, c_syn, t1)
            x += 0.5 * dt * (v0 + v1)
        output = pca.inverse_transform(scaler.inverse_transform(x.cpu().numpy()))
    X_syn = snv(output)
    if not np.isfinite(X_syn).all():
        raise ValueError('Flow generated nonfinite spectra')
    return X_syn, y_syn


class PLSDA:
    def fit(self, X, y):
        d = min(4, X.shape[1], len(X)-2)
        self.model = PLSRegression(n_components=d, scale=True)
        self.model.fit(X, y)
        return self
    def decision_function(self, X):
        return np.asarray(self.model.predict(X), dtype=float).reshape(-1) - 0.5


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 4)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(channels, hidden), nn.ReLU(),
                                  nn.Linear(hidden, channels), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2), nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 5, padding=2), nn.BatchNorm1d(channels), SEBlock(channels))
    def forward(self, x):
        return F.gelu(x + self.block(x))

class SEResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv1d(1, 24, 5, padding=2, stride=2)
        self.encoder = nn.Sequential(ResBlock(24), ResBlock(24))
        self.head = nn.Linear(24, 1)
    def forward(self, x):
        return self.head(self.encoder(self.entry(x.unsqueeze(1))).mean(dim=-1)).squeeze(-1)


class PatchTransformer(nn.Module):
    def __init__(self, length):
        super().__init__()
        patch = min(16, max(1, length))
        self.patch = nn.Conv1d(1, 32, kernel_size=patch, stride=patch)
        tokens = (length - patch) // patch + 1
        self.pos = nn.Parameter(torch.zeros((1, tokens, 32)))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                           dropout=0.1, batch_first=True,
                                           activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1,
                                                 enable_nested_tensor=False)
        self.head = nn.Linear(32, 1)
    def forward(self, x):
        tokens = self.patch(x.unsqueeze(1)).transpose(1, 2)
        return self.head(self.transformer(tokens + self.pos).mean(dim=1)).squeeze(-1)


def make_classical(method, n_real):
    if method == 'PLS-DA':
        return PLSDA()
    if method == 'Linear SVM':
        return make_pipeline(StandardScaler(), SVC(C=1, kernel='linear',
                                                   class_weight='balanced'))
    if method == 'RBF SVM':
        return make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf',
                                                   gamma='scale', class_weight='balanced'))
    if method == 'Elastic-net LR':
        return make_pipeline(StandardScaler(), LogisticRegression(
            solver='saga', penalty='elasticnet', l1_ratio=0.5, C=0.2,
            class_weight='balanced', max_iter=3000, tol=0.01, random_state=SEED))
    if method == 'Random Forest':
        return RandomForestClassifier(n_estimators=120, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       n_jobs=1, random_state=SEED)
    if method == 'XGBoost':
        return XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                reg_lambda=5.0, objective='binary:logistic',
                                eval_metric='logloss', random_state=SEED, n_jobs=1)
    raise KeyError(method)


def train_nn(method, X_train, y_train, seed, length):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    factory = {'SE-ResNet1D': SEResNet1D,
               'Patch Transformer': lambda: PatchTransformer(length)}
    model = factory[method]().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    x_t = torch.as_tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.as_tensor(y_train, dtype=torch.float32, device=DEVICE)
    freqs = np.bincount(y_train, minlength=2).astype(float)
    class_weights = torch.as_tensor([len(y_train)/(2*freqs[0]),
                                     len(y_train)/(2*freqs[1])],
                                    dtype=torch.float32, device=DEVICE)
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    for _ in range(NEURAL_EPOCHS):
        model.train()
        for ids in torch.randperm(len(y_train), generator=rng).split(NEURAL_BATCH_SIZE):
            ids = ids.to(DEVICE)
            logits = model(x_t[ids])
            loss = F.binary_cross_entropy_with_logits(logits, y_t[ids], reduction='none')
            loss = (loss * class_weights[y_t[ids].long()]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def score_classifier(method, model, X_test):
    if method in ('SE-ResNet1D', 'Patch Transformer'):
        with torch.no_grad():
            return model(torch.as_tensor(X_test, dtype=torch.float32, device=DEVICE)).cpu().numpy().ravel()
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(X_test), dtype=float).ravel()
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1].ravel() - 0.5
    return np.asarray(model.predict(X_test), dtype=float).ravel() - 0.5


def score_table(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, scores) if len(set(y_true)) == 2 else np.nan
    return {'N_patients': len(y_true), 'correct': int(np.sum(y_true == y_pred)),
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'sensitivity': tp/max(1, tp+fn), 'specificity': tn/max(1, tn+fp),
            'roc_auc': auc, 'TP': int(tp), 'FN': int(fn), 'TN': int(tn), 'FP': int(fp)}


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    corrected = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(corrected, 1.0)
    return adjusted


def stratified_bootstrap_indices(y, B, rng):
    healthy = np.where(y == 0)[0]
    sick = np.where(y == 1)[0]
    return np.column_stack((rng.choice(healthy, size=(B, len(healthy)), replace=True),
                            rng.choice(sick, size=(B, len(sick)), replace=True)))


def bootstrap_ba(y, pred, indices):
    correct = np.asarray(pred == y, dtype=float)
    split = int(np.sum(y == 0))
    return 0.5 * (correct[indices[:, :split]].mean(axis=1)
                  + correct[indices[:, split:]].mean(axis=1))




# -------- Strictly matched patient-level benchmark --------
_, X, y, names = load_original_spectra()
if X.shape[1] < ANOVA_K:
    raise ValueError(f'ANOVA_K={ANOVA_K} requires at least that many channels; got {X.shape[1]}')
if DEBUG_FOLDS is not None:
    selected_folds = set(DEBUG_FOLDS)
    if not selected_folds or not selected_folds <= set(range(1, len(X)+1)):
        raise ValueError('DEBUG_FOLDS must contain valid 1-based original patient indices')
else:
    selected_folds = None

rows = []
for fold, (tr, te) in enumerate(LeaveOneOut().split(X), start=1):
    if selected_folds is not None and fold not in selected_folds:
        continue
    seed = SEED + 1031 * fold

    # Proposed only: use SNV, training-fold-only Flow, and training-fold-only ANOVA.
    X_real_snv = snv(X[tr])
    X_synth, y_synth = fit_and_generate_flow(X[tr], y[tr], seed)
    X_aug = np.vstack([X_real_snv, X_synth])
    y_aug = np.concatenate([y[tr], y_synth])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        selector = SelectKBest(f_classif, k=ANOVA_K).fit(X_aug, y_aug)
    XA = selector.transform(X_aug)
    XB = selector.transform(snv(X[te]))
    proposed_model = PLSDA().fit(XA, y_aug)
    proposed_score = float(score_classifier('PLS-DA', proposed_model, XB)[0])
    rows.append({'fold': fold, 'patient': names[te[0]], 'true': int(y[te[0]]),
                 'pipeline': PROPOSED, 'classifier': 'PLS-DA',
                 'augmentation': 'Latent Flow', 'feature_engineering': f'ANOVA top {ANOVA_K}',
                 'snv': True, 'n_real_train': len(tr), 'n_synthetic_train': len(y_synth),
                 'n_input_features': XA.shape[1], 'score': proposed_score,
                 'pred': int(proposed_score >= 0)})

    # Controls: no Flow, no synthetic subjects, no ANOVA, no extracted features.
    # BASELINE_SNV only standardizes each existing spectrum: spectral channels
    # remain unchanged and there is no feature selection or construction.
    X_train = snv(X[tr]) if BASELINE_SNV else X[tr].copy()
    X_test = snv(X[te]) if BASELINE_SNV else X[te].copy()
    assert X_train.shape[1] == X.shape[1] and X_test.shape[1] == X.shape[1]

    for i, classifier in enumerate(CLASSIFIERS):
        if classifier in NEURAL_MODELS:
            # StandardScaler is fitted on this fold's REAL training patients only.
            # This is numerical scaling, not feature engineering or augmentation.
            scaler = StandardScaler().fit(X_train)
            A = np.clip(scaler.transform(X_train), -10, 10).astype('float32')
            B = np.clip(scaler.transform(X_test), -10, 10).astype('float32')
            model = train_nn(classifier, A, y[tr], seed + (i+1)*37, A.shape[1])
        else:
            A, B = X_train, X_test
            model = make_classical(classifier, len(tr))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(A, y[tr])
        score = float(score_classifier(classifier, model, B)[0])
        rows.append({'fold': fold, 'patient': names[te[0]], 'true': int(y[te[0]]),
                     'pipeline': f'{classifier}', 'classifier': classifier,
                     'augmentation': 'None', 'feature_engineering': 'None',
                     'snv': BASELINE_SNV, 'n_real_train': len(tr), 'n_synthetic_train': 0,
                     'n_input_features': X.shape[1], 'score': score,
                     'pred': int(score >= 0)})
    print(f'Fold {fold:02d}/{len(X)}: 1 proposed + {len(CLASSIFIERS)} real-only baselines', flush=True)

predictions = pd.DataFrame(rows)
if predictions.empty:
    raise RuntimeError('No patient predictions generated')
if len(predictions) != len(PIPELINES) * predictions.fold.nunique():
    raise RuntimeError('Not every method was evaluated on every selected fold')
if predictions.groupby('pipeline')['patient'].nunique().ne(predictions.fold.nunique()).any():
    raise RuntimeError('Duplicated or missing held-out patients')
predictions.to_csv(OUTPUT_DIR / 'patient_predictions.csv', index=False)

ordered = {}
metrics_rows = []
for name in PIPELINES:
    d = predictions.loc[predictions.pipeline == name].sort_values('fold')
    yy, pp, ss = d.true.to_numpy(int), d.pred.to_numpy(int), d.score.to_numpy(float)
    ordered[name] = (yy, pp, ss)
    if len(np.unique(yy)) == 2:
        performance = score_table(yy, pp, ss)
    else:
        performance = {'N_patients': len(yy), 'correct': int((yy==pp).sum()),
                       'accuracy': accuracy_score(yy, pp), 'balanced_accuracy': np.nan,
                       'sensitivity': np.nan, 'specificity': np.nan, 'roc_auc': np.nan,
                       'TP': np.nan, 'FN': np.nan, 'TN': np.nan, 'FP': np.nan}
    metrics_rows.append({'pipeline': name, 'classifier': d.classifier.iloc[0],
                         'augmentation': d.augmentation.iloc[0],
                         'feature_engineering': d.feature_engineering.iloc[0],
                         'snv': bool(d.snv.iloc[0]),
                         'n_features': int(d.n_input_features.iloc[0]), **performance})
metrics = pd.DataFrame(metrics_rows)
ref_y, ref_pred, ref_score = ordered[PROPOSED]
all_folds = DEBUG_FOLDS is None

# Paired, stratified patient bootstrap; SAME resampled patients for all methods.
indices = None
if len(ref_y) >= 6 and len(np.unique(ref_y)) == 2:
    indices = stratified_bootstrap_indices(ref_y, BOOTSTRAP_REPS,
                                           np.random.default_rng(SEED + 799))
    ref_boot = bootstrap_ba(ref_y, ref_pred, indices)
    for i, name in enumerate(PIPELINES):
        b = bootstrap_ba(ref_y, ordered[name][1], indices)
        metrics.loc[i, 'ba_ci95_low'] = float(np.quantile(b, .025))
        metrics.loc[i, 'ba_ci95_high'] = float(np.quantile(b, .975))
else:
    metrics['ba_ci95_low'] = np.nan
    metrics['ba_ci95_high'] = np.nan
metrics['delta_ba_proposed_minus_baseline_pp'] = 100 * (
    metrics.loc[0, 'balanced_accuracy'] - metrics.balanced_accuracy)
metrics.to_csv(OUTPUT_DIR / 'benchmark_metrics.csv', index=False)

pairs = []
if all_folds:
    for name in BASELINE_NAMES:
        yi, pi, si = ordered[name]
        if not np.array_equal(yi, ref_y):
            raise ValueError(f'Patient order mismatch for {name}')
        wins = int(np.sum((ref_pred == ref_y) & (pi != yi)))
        losses = int(np.sum((ref_pred != ref_y) & (pi == yi)))
        discordant = wins + losses
        mcnemar = float(binomtest(wins, discordant, p=.5).pvalue) if discordant else 1.0
        d_ba = 100*(balanced_accuracy_score(ref_y, ref_pred) - balanced_accuracy_score(yi, pi))
        d_acc = 100*(accuracy_score(ref_y, ref_pred) - accuracy_score(yi, pi))
        if indices is not None:
            diffs = 100*(ref_boot - bootstrap_ba(yi, pi, indices))
            low, high = [float(z) for z in np.quantile(diffs, [.025, .975])]
        else:
            low, high = np.nan, np.nan
        pairs.append({'reference': PROPOSED, 'competitor': name,
                      'proposed_only_correct': wins, 'competitor_only_correct': losses,
                      'discordant_patients': discordant,
                      'delta_accuracy_pp': d_acc, 'delta_balanced_accuracy_pp': d_ba,
                      'delta_ba_ci95_low_pp': low, 'delta_ba_ci95_high_pp': high,
                      'exact_mcnemar_p': mcnemar})

tests = pd.DataFrame(pairs)
if not tests.empty:
    tests['holm_p'] = holm_adjust(tests.exact_mcnemar_p.to_numpy())
tests.to_csv(OUTPUT_DIR / 'paired_statistical_tests.csv', index=False)

config = {'total_real_patients': int(len(X)), 'n_healthy': int(np.sum(y == 0)),
          'n_sick': int(np.sum(y == 1)), 'n_spectral_channels': int(X.shape[1]),
          'evaluated_folds': int(predictions.fold.nunique()),
          'n_competing_baselines': len(CLASSIFIERS), 'classifiers': CLASSIFIERS,
          'reference': PROPOSED, 'proposed_snv': True,
          'baseline_snv': BASELINE_SNV,
          'baselines': 'Real only, complete spectrum, no augmentation, no feature engineering',
          'normalization_note': 'SVM/LR and neural models perform training-fold-only numerical scaling',
          'flow_steps': FLOW_TRAIN_STEPS, 'flow_ode_steps': FLOW_ODE_STEPS,
          'latent_dim': LATENT_DIM, 'synthetic_per_class_per_fold': SYNTHETIC_PER_CLASS,
          'anovak': ANOVA_K, 'neural_epochs': NEURAL_EPOCHS,
          'bootstrap_replications': BOOTSTRAP_REPS,
          'random_seed': SEED, 'partial_debug': not all_folds,
          'interpretation': 'Exploratory on same patient cohort used to select ANOVA256 + PLS-DA; external validation required',
          'statistical_caveat': 'With few patients, bootstrap CIs and exact McNemar tests have limited resolution; correction across all 9 comparisons applied.'}
(OUTPUT_DIR / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

# -------- Publication-ready PDF figures, white background, no grids --------
order = metrics.sort_values(['balanced_accuracy', 'roc_auc'], ascending=False)
plot_names = order.pipeline.tolist()
fig, ax = plt.subplots(figsize=(10, max(4.6, len(plot_names)*.47)))
ypos = np.arange(len(plot_names))
vals = order.balanced_accuracy.to_numpy(float)*100
low = order.ba_ci95_low.to_numpy(float)*100
high = order.ba_ci95_high.to_numpy(float)*100
err = np.vstack((np.maximum(0, vals-low), np.maximum(0, high-vals)))
colors = ['.15' if n == PROPOSED else '.65' for n in plot_names]
ax.barh(ypos, vals, color=colors, height=.65)
if np.all(np.isfinite(err)):
    ax.errorbar(vals, ypos, xerr=err, fmt='none', ecolor='.25', capsize=3, lw=1)
ax.set_yticks(ypos, plot_names)
ax.invert_yaxis()
ax.set_xlim(0, 105)
ax.set_xlabel('Balanced accuracy (%) with 95% patient-bootstrap CI')
for yi, v in zip(ypos, vals):
    if np.isfinite(v):
        ax.text(min(v+1, 101.0), yi, f'{v:.1f}', va='center', fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / '01_benchmark_accuracy.pdf', bbox_inches='tight')
plt.close(fig)

if all_folds and not tests.empty:
    t = tests.sort_values('delta_balanced_accuracy_pp', ascending=False).iloc[::-1]
    xx = t.delta_balanced_accuracy_pp.to_numpy(float)
    ci = np.vstack((np.maximum(0, xx-t.delta_ba_ci95_low_pp.to_numpy(float)),
                    np.maximum(0, t.delta_ba_ci95_high_pp.to_numpy(float)-xx)))
    fig, ax = plt.subplots(figsize=(10, max(4.6, len(t)*.47)))
    ypos = np.arange(len(t))
    ax.axvline(0, color='.45', lw=1)
    ax.errorbar(xx, ypos, xerr=ci, fmt='o', color='.2', capsize=3, ms=4)
    ax.set_yticks(ypos, t.competitor)
    ax.set_xlabel('Proposed minus real-only baseline: balanced accuracy (percentage points)')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / '02_paired_differences.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    shades = np.linspace(.3, .75, len(BASELINE_NAMES))
    for i, name in enumerate([PROPOSED] + BASELINE_NAMES):
        yi, pi, scores = ordered[name]
        fpr, tpr, _ = roc_curve(yi, scores)
        auc = roc_auc_score(yi, scores)
        is_ref = name == PROPOSED
        ax.plot(fpr, tpr, lw=2.6 if is_ref else 1.0,
                color='black' if is_ref else str(shades[i-1]),
                linestyle='-' if is_ref else ['-', '--', ':', '-.'][i%4],
                label=f'{name} (AUC {auc:.2f})')
    ax.plot([0, 1], [0, 1], '--', lw=.7, color='.7')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / '03_roc_curves.pdf', bbox_inches='tight')
    plt.close(fig)

    displayed = [PROPOSED] + [n for n in plot_names if n != PROPOSED]
    correct_matrix = np.stack([(ordered[n][1] == ref_y).astype(int)
                               for n in displayed])
    fig, ax = plt.subplots(figsize=(max(8, len(ref_y)*.5), 5.6))
    ax.imshow(correct_matrix, aspect='auto', cmap='Greys', vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(displayed)), displayed, fontsize=8)
    ax.set_xticks(np.arange(len(ref_y)), np.arange(1, len(ref_y)+1))
    ax.set_xlabel('Held-out original patient (LOOCV fold)')
    ax.set_xticks(np.arange(-.5, len(ref_y), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(displayed), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=.6)
    ax.tick_params(which='minor', length=0)
    fig.text(.96, .04, 'Black = correct; white = incorrect', ha='right', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(OUTPUT_DIR / '04_patient_error_map.pdf', bbox_inches='tight')
    plt.close(fig)

print(f'Saved {len(PIPELINES)} pipelines, {predictions.fold.nunique()} folds to {OUTPUT_DIR}', flush=True)
