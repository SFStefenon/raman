"""Sequential three-stage spectral ablation with nested patient-level validation.

ONE SOURCE FILE; switch RUN_MODE = 'quick' / 'complete' below.
Input folders: ROOT/saudaveis/*.csv and ROOT/doentes/*.csv.
Each CSV corresponds to ONE original patient, 2 columns: wavenumber, signal.

At each outer held-out original patient:
  Step 1: inner patient-level CV chooses the classifier on SNV real spectra.
  Step 2: with classifier fixed, inner CV selects real-only data or a
          class-conditional augmentation (three Latent Flow counts, optionally
          Spectral Mixup and PCA-Gaussian historical controls).
  Step 3: with classifier and augmentation fixed, inner CV chooses Full spectra,
          three mRMR sizes, two stability sizes, two PCA-peak sizes, and
          optionally two ANOVA historical references.
Each chosen intermediate and final configuration is then refitted using ONLY
outer training patients and predicts ONLY the one held-out original patient.
The outer patient is NEVER used in any of the three configuration selections.

Outputs: stage-specific CSVs, LaTeX booktabs tables, 11 PDF figures,
held-out-patient comparisons, reference benchmarks, stage-wise ROC/PR,
selected wavelength frequencies and path diagrams, all inside OUTPUT_DIR.
Mode quick = smoke check on 4 held-out patients, small inner CV; don't publish.
Mode complete = all patient-level outer and inner leave-one-out folds.
The full model grid is computationally intensive (repeated nested flow fits).
Reference controls make comparisons more interpretable but do not contribute
novel methods. Current aeon private transform attributes are used only to
obtain continuous MultiRocket-Hydra margins, not calibrated probabilities.

Py3.9: pip install numpy pandas scipy scikit-learn matplotlib torch 'aeon==1.0.0'
Py>=3.11: pip install numpy pandas scipy scikit-learn matplotlib torch aeon

Methods based on prior manuscript/code. This does not validate a clinical test.
"""
from __future__ import annotations
from pathlib import Path
import json
import random
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from scipy.signal import find_peaks
from scipy.stats import rankdata
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score, roc_curve,
                             f1_score, matthews_corrcoef, precision_recall_curve,
                             average_precision_score)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import torch
from torch import nn
from torch.nn import functional as F

# USER SETTINGS: change only RUN_MODE to switch between the two workflows.
# In Spyder, explicit assignment prevents previous console variables from
# silently overriding the selected mode on repeated runs.
RUN_MODE = 'complete'                    # 'quick' or 'complete'
RUN_AEON = True                      # needs aeon; False for dependency-free smoke test
INCLUDE_REFERENCE_CONTROLS = True   # SVM, ANOVA, Mixup, PCA-Gaussian controls
RUN_FIXED_BENCHMARKS = True         # three prespecified external benchmark recipes
PRETRAINED_ENCODER_PATH = None       # optional externally pretrained TorchScript
OVERRIDES = {}                       # e.g. {'ROOT': '/path/to/paper 2'}
ROOT = Path(OVERRIDES.get('ROOT', Path(__file__).resolve().parent))
HEALTHY_DIR = Path(OVERRIDES.get('HEALTHY_DIR', ROOT / 'saudaveis'))
SICK_DIR = Path(OVERRIDES.get('SICK_DIR', ROOT / 'doentes'))
if RUN_MODE not in ('quick', 'complete'):
    raise ValueError('RUN_MODE must be quick or complete')
FAST = RUN_MODE == 'quick'
OUTPUT_DIR = Path(OVERRIDES.get('OUTPUT_DIR', ROOT / ('sequential_ablation_' + RUN_MODE)))
SEED = int(OVERRIDES.get('SEED', 42))
DEVICE = torch.device(OVERRIDES.get('DEVICE', 'cpu'))
OUTER_QUICK_PER_CLASS = int(OVERRIDES.get('OUTER_QUICK_PER_CLASS', 2))
INNER_QUICK_PER_CLASS = int(OVERRIDES.get('INNER_QUICK_PER_CLASS', 2))
FLOW_STEPS = int(OVERRIDES.get('FLOW_STEPS', 30 if FAST else 350))
FLOW_BATCH_SIZE = int(OVERRIDES.get('FLOW_BATCH_SIZE', 32 if FAST else 64))
FLOW_ODE_STEPS = int(OVERRIDES.get('FLOW_ODE_STEPS', 8 if FAST else 32))
LATENT_DIM = int(OVERRIDES.get('LATENT_DIM', 4))
AUG_COUNTS = tuple(OVERRIDES.get('AUG_COUNTS', (4, 8, 12) if FAST else (15, 30, 60)))
MRMR_K = int(OVERRIDES.get('MRMR_K', 32 if FAST else 256))
FEATURE_K = tuple(OVERRIDES.get('FEATURE_K', (12, 20, 32) if FAST else (64, 128, 256)))
MRMR_BINS = int(OVERRIDES.get('MRMR_BINS', 3))
STABILITY_K = int(OVERRIDES.get('STABILITY_K', 20 if FAST else 128))
STABILITY_BOOTSTRAPS = int(OVERRIDES.get('STABILITY_BOOTSTRAPS', 3 if FAST else 24))
STABILITY_ALPHA = float(OVERRIDES.get('STABILITY_ALPHA', 0.015))
PEAK_K = int(OVERRIDES.get('PEAK_K', 20 if FAST else 128))
QUANT_DEPTH = int(OVERRIDES.get('QUANT_DEPTH', 2 if FAST else 4))
HYDRA_GROUPS = int(OVERRIDES.get('HYDRA_GROUPS', 8 if FAST else 64))
HYDRA_KERNELS = int(OVERRIDES.get('HYDRA_KERNELS', 4 if FAST else 8))
N_BOOTSTRAP = int(OVERRIDES.get('N_BOOTSTRAP', 200 if FAST else 2000))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.grid': False, 'font.size': 10,
    'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': '#29374A', 'text.color': '#29374A',
    'xtick.color': '#566477', 'ytick.color': '#566477',
    'axes.edgecolor': '#D0D8E3', 'font.size': 9.5})
PALETTE = {'navy':'#24456B', 'teal':'#117D83','blue':'#468AC2',
           'amber':'#D5A04B','coral':'#C46B66', 'muted':'#8495A6',
           'ink':'#26364A','light':'#E6EDF3','bg':'#F5F8FB'}

def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snv(x):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return (x - x.mean(axis=1, keepdims=True)) / np.maximum(
        x.std(axis=1, keepdims=True), 1e-10)


def load_data():
    """Read one patient per CSV directly from each class folder (no ZIPs)."""
    records = []
    for folder, label in ((HEALTHY_DIR, 0), (SICK_DIR, 1)):
        if not folder.is_dir():
            raise FileNotFoundError(
                f'Missing class folder: {folder}\n'
                'Expected directories saudaveis/ and doentes/ beside this script, '
                'each containing original patient CSV files. '
                'Alternatively set HEALTHY_DIR and SICK_DIR at the top.')
        files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() == '.csv'
                       and not any(part.startswith('.') or part.lower() == '__macosx'
                                   for part in p.relative_to(folder).parts))
        if not files:
            raise FileNotFoundError(f'No patient CSV files found in {folder}')
        for path in files:
            # A header row is normally present. If first row is fully numeric,
            # preserve it as data instead of silently dropping one wavenumber.
            with path.open('r', encoding='utf-8-sig') as f:
                first = f.readline().strip()
            sep = ';' if first.count(';') > first.count(',') else ','
            values = [part.strip().strip('\"') for part in first.split(sep)]
            def _is_number(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False
            numeric_header = len(values) == 2 and all(map(_is_number, values))
            df = pd.read_csv(path, encoding='utf-8-sig', sep=sep,
                             header=None if numeric_header else 0)
            if df.shape[1] != 2:
                raise ValueError(f'Expected exactly 2 CSV columns (wavenumber, signal): {path}')
            wn = pd.to_numeric(df.iloc[:, 0], errors='raise').to_numpy(float)
            x = pd.to_numeric(df.iloc[:, 1], errors='raise').to_numpy(float)
            if len(wn) < 8:
                raise ValueError(f'Expected a complete spectrum, got {len(wn)} channels: {path}')
            if not (np.isfinite(wn).all() and np.isfinite(x).all()):
                raise ValueError('Nonfinite spectrum: ' + str(path))
            # Keep original labels and ensure patient IDs are unique across folders.
            patient_id = str(path.relative_to(folder).with_suffix('')).replace('/', '_')
            records.append((patient_id, int(label), wn, x))
    ref = records[0][2]
    for patient, _, wn, _ in records:
        if len(wn) != len(ref) or not np.allclose(wn, ref, atol=1e-6, rtol=0):
            raise ValueError(f'Inconsistent wavenumber grid: {patient}')
    ids = np.array([f'{"healthy" if label == 0 else "sick"}/{name}'
                    for name, label, _, _ in records])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate original patient identifiers')
    X = np.stack([r[3] for r in records])
    y = np.array([r[1] for r in records], dtype=int)
    if min(np.bincount(y, minlength=2)) < 3:
        raise ValueError('Need at least 3 original patients per class')
    return ref, X, y, ids


class FlowVelocity(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim + 7, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, latent_dim))

    def forward(self, z, cls, t):
        angle = 2 * np.pi * t
        time = [t, torch.sin(angle), torch.cos(angle),
                torch.sin(2 * angle), torch.cos(2 * angle)]
        return self.net(torch.cat([z, F.one_hot(cls.long(), 2).float()] + time, 1))


def generate_latent_flow(X_train_real, y_train_real, seed, per_class):
    """Train PCA + conditional flow ONLY on original outer-training spectra."""
    seed_everything(seed)
    real = snv(X_train_real)
    d = min(LATENT_DIM, len(real) - 2, real.shape[1])
    pca = PCA(n_components=d, random_state=seed)
    scaler = StandardScaler()
    latent = scaler.fit_transform(pca.fit_transform(real)).astype(np.float32)
    data = torch.tensor(latent, device=DEVICE)
    y_t = torch.as_tensor(y_train_real, dtype=torch.long, device=DEVICE)
    pool = [torch.where(y_t == k)[0] for k in (0, 1)]
    if min(len(p) for p in pool) < 2:
        raise ValueError('Each training fold must contain >=2 real patients/class')
    model = FlowVelocity(d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(FLOW_STEPS):
        c = torch.randint(0, 2, (FLOW_BATCH_SIZE,), device=DEVICE)
        idx = torch.empty_like(c)
        for k in (0, 1):
            mask = c == k
            idx[mask] = pool[k][torch.randint(0, len(pool[k]),
                                        (int(mask.sum().item()),), device=DEVICE)]
        z1 = data[idx]
        z0 = torch.randn_like(z1)
        t = torch.rand((len(c), 1), device=DEVICE)
        zt = z0 * (1 - t) + z1 * t
        loss = F.mse_loss(model(zt, c, t), z1 - z0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    y_syn = np.repeat([0, 1], per_class).astype(int)
    c_syn = torch.as_tensor(y_syn, dtype=torch.long, device=DEVICE)
    model.eval()
    with torch.no_grad():
        z = torch.randn((len(y_syn), d), device=DEVICE)
        dt = 1.0 / FLOW_ODE_STEPS
        for k in range(FLOW_ODE_STEPS):
            t0 = torch.full((len(y_syn), 1), k * dt, device=DEVICE)
            t1 = torch.full((len(y_syn), 1), (k + 1) * dt, device=DEVICE)
            v0 = model(z, c_syn, t0)
            v1 = model(z + dt * v0, c_syn, t1)
            z = z + (dt / 2.0) * (v0 + v1)
        X_syn = pca.inverse_transform(scaler.inverse_transform(z.cpu().numpy()))
    X_syn = snv(X_syn)
    if not np.isfinite(X_syn).all():
        raise FloatingPointError('Generator emitted nonfinite spectra')
    return X_syn, y_syn


def discrete_mutual_information(one_hot_a, one_hot_b):
    """MI in nats between all columns of a and one-hot b (n_samples,n_states)."""
    n = one_hot_a.shape[1]
    a = np.moveaxis(one_hot_a, 0, 1).astype(np.float64)  # (n, bins_a, features)
    b = one_hot_b.astype(np.float64)                    # (n, bins_b)
    counts = np.einsum('naf,nb->abf', a, b, optimize=True)
    joint = counts / n
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    den = pa * pb
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(joint > 0, joint * np.log(
            np.maximum(joint, 1e-16) / np.maximum(den, 1e-16)), 0.0)
    return terms.sum(axis=(0, 1))


def mrmr_indices(X_real, y_real, k, n_bins=3, sort_result=True):
    """Exact empirical MI-difference mRMR on quantile-discretized real spectra.

    All discretization boundaries and relevance/redundancy statistics are fitted
    on original TRAINING patients; synthetic data does not influence selection.
    """
    X = snv(X_real)
    n, p = X.shape
    k = min(k, p)
    q = np.quantile(X, np.arange(1, n_bins) / n_bins, axis=0)
    bins = (X[None, :, :] > q[:, None, :]).sum(axis=0).astype(np.int8)
    hot = np.asarray([bins == b for b in range(n_bins)], dtype=np.float32)
    label_hot = np.eye(2, dtype=np.float32)[np.asarray(y_real, int)]
    relevance = discrete_mutual_information(hot, label_hot)
    totals = np.zeros(p, float)
    selected = []
    forbidden = np.zeros(p, bool)
    for i in range(k):
        score = relevance - totals / max(i, 1)
        score[forbidden] = -np.inf
        j = int(np.argmax(score))
        selected.append(j)
        forbidden[j] = True
        totals += discrete_mutual_information(hot, hot[:, :, j].T)
    # Return in spectral order; keep local adjacency meaningful for interpretability.
    ranked = np.array(selected, dtype=int)
    return np.sort(ranked) if sort_result else ranked


def stability_indices(X_real, y_real, k, seed, sort_result=True):
    """Bootstrap stability selection using elastic-net-penalized logistic SGD."""
    X = snv(X_real)
    n, p = X.shape
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y_real == c) for c in (0, 1)]
    count = np.zeros(p, dtype=int)
    strength = np.zeros(p, dtype=float)
    for b in range(STABILITY_BOOTSTRAPS):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        model = SGDClassifier(loss='log_loss', penalty='elasticnet',
            l1_ratio=0.8, alpha=STABILITY_ALPHA, max_iter=1500,
            tol=1e-4, random_state=seed + b)
        model.fit(Z[ix], y_real[ix])
        coef = np.abs(model.coef_[0])
        count += (coef > 1e-7)
        strength += coef
    # Sort first by selection frequency, then by magnitude, then by index.
    best = np.lexsort((np.arange(p), -strength, -count))[:min(k, p)]
    return (np.sort(best) if sort_result else best), count / STABILITY_BOOTSTRAPS


class PLSDA:
    def fit(self, X, y):
        n = min(4, X.shape[1], X.shape[0] - 2)
        self.reg = PLSRegression(n_components=n, scale=True)
        self.reg.fit(X, y)
        return self

    def decision_function(self, X):
        return self.reg.predict(X).reshape(-1) - 0.5


def make_elasticnet(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.8, C=0.2,
        class_weight='balanced', max_iter=5000, tol=1e-3, random_state=seed))


def array_from_aeon(z):
    if hasattr(z, 'detach'):
        z = z.detach().cpu().numpy()
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError('Expected a 2D feature matrix from aeon')
    if not np.isfinite(z).all():
        raise FloatingPointError('aeon transform produced nonfinite values')
    return z


def normalize_margin(test_margin, train_margin):
    """Scale raw margins by training dispersion, preserving zero threshold.

    This helps compare ROC scores across different classifiers/folds; it is
    NOT probability calibration. ROC-AUC remains descriptive for heterogeneous
    models fitted independently inside outer folds.
    """
    scale = max(float(np.std(np.asarray(train_margin, float))), 1e-6)
    return np.asarray(test_margin, float).reshape(-1) / scale


def predict_quant(X_train, y_train, X_test, classifier='ridge', seed=42):
    from aeon.transformations.collection.interval_based import QUANTTransformer
    transformer = QUANTTransformer(interval_depth=QUANT_DEPTH, quantile_divisor=4)
    train = array_from_aeon(transformer.fit_transform(X_train[:, None, :]))
    test = array_from_aeon(transformer.transform(X_test[:, None, :]))
    if classifier == 'ridge':
        head = RidgeClassifier(alpha=1.0, class_weight='balanced')
    elif classifier == 'svm':
        head = LinearSVC(C=0.1, class_weight='balanced', max_iter=6000,
                         random_state=int(seed))
    else:
        raise ValueError(classifier)
    clf = make_pipeline(StandardScaler(), head)
    clf.fit(train, y_train)
    return normalize_margin(clf.decision_function(test), clf.decision_function(train))


def predict_multirocket_hydra(X_train, y_train, X_test, seed):
    from aeon.classification.convolution_based import MultiRocketHydraClassifier
    model = MultiRocketHydraClassifier(n_kernels=HYDRA_KERNELS,
        n_groups=HYDRA_GROUPS, n_jobs=1, class_weight='balanced', random_state=int(seed))
    A = X_train[:, None, :].astype(np.float32)
    B = X_test[:, None, :].astype(np.float32)
    model.fit(A, y_train)
    # A RidgeClassifierCV-backed aeon classifier can expose hard labels through
    # predict_proba. Use its fitted continuous decision function instead.
    if hasattr(model, '_transform_hydra') and hasattr(model, 'classifier'):
        def margin(x):
            h = model._scale_hydra.transform(model._transform_hydra.transform(x))
            r = model._scale_multirocket.transform(model._transform_multirocket.transform(x))
            if hasattr(h, 'detach'): h = h.detach().cpu().numpy()
            if hasattr(r, 'detach'): r = r.detach().cpu().numpy()
            if hasattr(h, 'toarray'): h = h.toarray()
            if hasattr(r, 'toarray'): r = r.toarray()
            return np.asarray(model.classifier.decision_function(
                np.concatenate([np.asarray(h), np.asarray(r)], axis=1)), float).ravel()
        return normalize_margin(margin(B), margin(A))
    raise RuntimeError('aeon MultiRocket-Hydra internals changed. '
                       'Its continuous ridge margin is required for ROC-AUC.')


def load_pretrained_encoder():
    if PRETRAINED_ENCODER_PATH is None:
        return None
    path = Path(PRETRAINED_ENCODER_PATH)
    if not path.is_file():
        raise FileNotFoundError('Externally pretrained TorchScript missing: ' + str(path))
    module = torch.jit.load(str(path), map_location='cpu')
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def frozen_embeddings(model, X):
    with torch.no_grad():
        z = model(torch.as_tensor(X[:, None, :], dtype=torch.float32))
    if not torch.is_tensor(z):
        raise ValueError('Pretrained encoder must return a single tensor')
    result = z.cpu().numpy()
    if result.ndim != 2 or len(result) != len(X) or not np.isfinite(result).all():
        raise ValueError('Pretrained model output must be finite [N, embedding_dim]')
    return result.astype(float)


def predict_model(name, X_train, y_train, X_test, seed, encoder=None):
    if name == 'QUANT + Ridge':
        return predict_quant(X_train, y_train, X_test, 'ridge', seed)
    if name == 'QUANT + Linear SVM':
        return predict_quant(X_train, y_train, X_test, 'svm', seed)
    if name == 'MultiRocket-Hydra':
        return predict_multirocket_hydra(X_train, y_train, X_test, seed)
    if name == 'PLS-DA':
        model = PLSDA().fit(X_train, y_train)
    elif name == 'Elastic-net LR':
        model = make_elasticnet(int(seed))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=FutureWarning,
                                     message="'penalty' was deprecated.*")
            model.fit(X_train, y_train)
    elif name == 'Linear SVM':
        model = make_pipeline(StandardScaler(), LinearSVC(
            class_weight='balanced', C=0.1, max_iter=6000, random_state=int(seed)))
        model.fit(X_train, y_train)
    elif name == 'RBF SVM':
        model = make_pipeline(StandardScaler(), SVC(
            C=1.0, kernel='rbf', class_weight='balanced', gamma='scale'))
        model.fit(X_train, y_train)
    elif name == 'Frozen encoder + Linear SVM':
        if encoder is None:
            raise ValueError('Pretrained encoder is not supplied')
        model = make_pipeline(StandardScaler(), LinearSVC(
            class_weight='balanced', C=0.1, max_iter=15000, random_state=int(seed)))
        Xt = frozen_embeddings(encoder, X_train)
        Xv = frozen_embeddings(encoder, X_test)
        model.fit(Xt, y_train)
        return normalize_margin(model.decision_function(Xv), model.decision_function(Xt))
    else:
        raise ValueError('Unexpected model: ' + name)
    return normalize_margin(model.decision_function(X_test), model.decision_function(X_train))


# Names are deterministic for selection ties and paper outputs.
small_k, mid_k, large_k = FEATURE_K
SELECTORS = ['Full spectrum',
             f'mRMR-{small_k}', f'mRMR-{mid_k}', f'mRMR-{large_k}',
             f'Stability-{small_k}', f'Stability-{mid_k}',
             f'PCA peaks-{small_k}', f'PCA peaks-{mid_k}']
if INCLUDE_REFERENCE_CONTROLS:
    SELECTORS += [f'ANOVA-{small_k}', f'ANOVA-{large_k}']
AUGMENTATIONS = ['Real only'] + [f'Latent Flow {n}/class' for n in AUG_COUNTS]
if INCLUDE_REFERENCE_CONTROLS:
    AUGMENTATIONS += [f'Spectral Mixup {AUG_COUNTS[1]}/class',
                      f'PCA-Gaussian {AUG_COUNTS[1]}/class']
MODELS = (['QUANT + Ridge', 'QUANT + Linear SVM', 'MultiRocket-Hydra']
          if RUN_AEON else []) + ['PLS-DA', 'Elastic-net LR']
if INCLUDE_REFERENCE_CONTROLS:
    MODELS += ['Linear SVM', 'RBF SVM']
if PRETRAINED_ENCODER_PATH:
    MODELS += ['Frozen encoder + Linear SVM']

def peak_indices(x_real, k, seed):
    """PCA-loading peak selector: uses training real spectra, not labels."""
    xx = snv(x_real)
    pcs = PCA(n_components=min(4, len(xx) - 2), random_state=seed).fit(xx)
    influence = np.sqrt(np.sum((pcs.components_ ** 2) *
                              pcs.explained_variance_[:, None], axis=0))
    peaks, _ = find_peaks(influence, distance=5)
    priority = sorted(peaks, key=lambda j: (-influence[j], int(j)))
    selection = list(priority[:min(k, len(priority))])
    # Deterministic fallback if fewer than k local maxima exist.
    for j in np.argsort(-influence, kind='mergesort'):
        if len(selection) == min(k, xx.shape[1]):
            break
        if int(j) not in selection:
            selection.append(int(j))
    return np.array(sorted(selection), dtype=int)


def feature_indices(selector, x_real, y_real, seed):
    """Selector is fitted on REAL training patients, without synthetic-label inflation."""
    if selector == 'Full spectrum':
        return None
    base, kstr = selector.rsplit('-', 1)
    k = min(int(kstr), x_real.shape[1])
    if base == 'mRMR':
        return mrmr_indices(x_real, y_real, k, MRMR_BINS)
    if base == 'Stability':
        return stability_indices(x_real, y_real, k, seed)[0]
    if base == 'PCA peaks':
        return peak_indices(x_real, k, seed)
    if base == 'ANOVA':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            f, _ = f_classif(snv(x_real), y_real)
        f = np.nan_to_num(f, nan=-np.inf, posinf=np.finfo(float).max,
                          neginf=-np.inf)
        return np.sort(np.argsort(-f, kind='mergesort')[:k])
    raise KeyError(selector)

def representation(x, indices, model):
    if indices is None:
        return x
    if model in ('QUANT + Ridge', 'MultiRocket-Hydra', 'Frozen encoder + Linear SVM'):
        # Convolution and interval models need a regularly sampled ordered signal;
        # linear interpolation restores the original positions and length.
        original = np.arange(x.shape[1])
        return np.stack([np.interp(original, indices, spectrum[indices]) for spectrum in x])
    return x[:, indices]


def balanced_selection_scores(y, pred, margin):
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    margin = np.asarray(margin, float)
    if len(set(y)) < 2:
        raise ValueError('An inner evaluation must include both original classes')
    return dict(BA=float(balanced_accuracy_score(y, pred)),
                AUC=float(roc_auc_score(y, margin)),
                ACC=float(accuracy_score(y, pred)))


def select_winner(frame, names):
    # frame has one row/candidate, and exactly one entry per inner test patient.
    by = {}
    for name in names:
        dd = frame.loc[frame.candidate == name].sort_values('inner_fold')
        if len(dd) == 0:
            raise RuntimeError('No inner CV results for ' + name)
        by[name] = balanced_selection_scores(dd.y_true, dd.y_pred, dd.score)
    # Priority: balanced accuracy > ROC-AUC > accuracy > predefined order.
    selected = max(names, key=lambda n: (by[n]['BA'], by[n]['AUC'],
                                         by[n]['ACC'], -names.index(n)))
    return selected, by


def evaluate_predictions(y, pred, score):
    y = np.asarray(y, int); pred = np.asarray(pred, int); score = np.asarray(score, float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(N=len(y), Correct=int((y == pred).sum()),
                Accuracy=float(accuracy_score(y, pred)),
                BalancedAccuracy=float(balanced_accuracy_score(y, pred)),
                Sensitivity=float(tp / max(1, tp + fn)),
                Specificity=float(tn / max(1, tn + fp)),
                AUC=float(roc_auc_score(y, score)),
                AP=float(average_precision_score(y, score)),
                F1=float(f1_score(y, pred, zero_division=0)),
                MCC=float(matthews_corrcoef(y, pred)),
                TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


def freeze_encoder():
    if not PRETRAINED_ENCODER_PATH:
        return None
    model = torch.jit.load(str(PRETRAINED_ENCODER_PATH), map_location='cpu')
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def train_and_predict(model, xtr, ytr, xte, seed, encoder=None):
    seed_everything(seed)
    return float(predict_model(model, xtr, ytr, xte, seed, encoder)[0])


def inner_indices(n, y_all, original_outer_tr, outer_fold):
    if not FAST:
        return list(original_outer_tr)
    rng = np.random.default_rng(SEED + outer_fold * 137)
    chosen = []
    for cls in (0, 1):
        choices = original_outer_tr[y_all[original_outer_tr] == cls]
        chosen.extend(rng.choice(choices, size=min(INNER_QUICK_PER_CLASS, len(choices)),
                                 replace=False).tolist())
    return sorted(chosen)


def parse_augmentation(aug):
    if aug == 'Real only':
        return 'Real only', 0
    method, count = aug.rsplit(' ', 1)
    return method, int(count.split('/')[0])


def sample_reference_augmentation(xtr, ytr, method, count, seed):
    """Class-conditional reference methods; use original training spectra only."""
    rng = np.random.default_rng(int(seed))
    labels = np.repeat([0, 1], count).astype(int)
    if method == 'Spectral Mixup':
        synthetic = []
        for c in labels:
            pool = xtr[ytr == c]
            a, b = rng.integers(0, len(pool), 2)
            mix = rng.beta(0.4, 0.4)
            synthetic.append(mix * pool[a] + (1 - mix) * pool[b] +
                             rng.normal(0, 0.008, size=xtr.shape[1]))
        return snv(np.asarray(synthetic)), labels
    if method == 'PCA-Gaussian':
        pc = PCA(n_components=min(LATENT_DIM, len(xtr)-2, xtr.shape[1]),
                 random_state=int(seed))
        std = StandardScaler()
        z = std.fit_transform(pc.fit_transform(xtr))
        fake = np.zeros((len(labels), z.shape[1]), dtype=float)
        for c in (0, 1):
            zc = z[ytr == c]
            cov = LedoitWolf().fit(zc).covariance_ + np.eye(z.shape[1]) * 1e-6
            fake[labels == c] = rng.multivariate_normal(zc.mean(axis=0), cov,
                                                         size=int(np.sum(labels == c)))
        return snv(pc.inverse_transform(std.inverse_transform(fake))), labels
    raise KeyError(method)


def aug_dataset(record, aug):
    if aug == 'Real only':
        return record['xtr'], record['ytr']
    method, count = parse_augmentation(aug)
    if method not in record['synthetic_cache']:
        if method == 'Latent Flow':
            total = max(AUG_COUNTS)
            data = generate_latent_flow(record['xtr'], record['ytr'],
                                        record['seed'], total)
        else:
            total = count
            data = sample_reference_augmentation(record['xtr'], record['ytr'],
                                                 method, total, record['seed']+337)
        record['synthetic_cache'][method] = (data, total)
    (xs, ys), nmax = record['synthetic_cache'][method]
    chosen = np.r_[0:count, nmax:nmax+count]
    return np.vstack([record['xtr'], xs[chosen]]), np.r_[record['ytr'], ys[chosen]]

def make_inner_records(x, y, outer_tr, outer_fold):
    result = []
    for j, inner_test in enumerate(inner_indices(len(x), y, outer_tr, outer_fold)):
        inner_tr = outer_tr[outer_tr != inner_test]
        result.append(dict(inner_fold=j, xtr=snv(x[inner_tr]),
            ytr=y[inner_tr], xte=snv(x[inner_test:inner_test+1]),
            y_true=int(y[inner_test]), synthetic_cache={},
            seed=int(SEED + (outer_fold + 1) * 100003 + (inner_test + 1) * 101),
            selector_cache={}, score_cache={}))
    return result


def evaluate_candidates(records, candidates, model, aug, selector, step, encoder, candidates_axis):
    rows=[]
    for rec in records:
        # Cache selector fitted ONLY on original inner training patients.
        if selector is not None and selector not in rec['selector_cache']:
            rec['selector_cache'][selector] = feature_indices(
                selector, rec['xtr'], rec['ytr'], rec['seed'])
        for name in candidates:
            mod = name if candidates_axis == 'model' else model
            ag = name if candidates_axis == 'augmentation' else aug
            sel = name if candidates_axis == 'selector' else selector
            if sel not in rec['selector_cache']:
                if sel.startswith('mRMR-'):
                    if '__mrmr_rank__' not in rec['selector_cache']:
                        rec['selector_cache']['__mrmr_rank__'] = mrmr_indices(
                            rec['xtr'], rec['ytr'], max(FEATURE_K), MRMR_BINS, False)
                    rec['selector_cache'][sel] = np.sort(
                        rec['selector_cache']['__mrmr_rank__'][:int(sel.rsplit('-',1)[1])])
                elif sel.startswith('Stability-'):
                    if '__stable_rank__' not in rec['selector_cache']:
                        ranked, _ = stability_indices(rec['xtr'], rec['ytr'],
                                      max(FEATURE_K[1],STABILITY_K), rec['seed'], False)
                        rec['selector_cache']['__stable_rank__'] = ranked
                    rec['selector_cache'][sel] = np.sort(
                        rec['selector_cache']['__stable_rank__'][:int(sel.rsplit('-',1)[1])])
                else:
                    rec['selector_cache'][sel] = feature_indices(
                        sel, rec['xtr'], rec['ytr'], rec['seed'])
            idx = rec['selector_cache'][sel]
            a, lab = aug_dataset(rec, ag)
            a = representation(a, idx, mod)
            xt = representation(rec['xte'], idx, mod)
            cache_key = (mod, ag, sel)
            if cache_key in rec['score_cache']:
                score = rec['score_cache'][cache_key]
            else:
                score = train_and_predict(mod, a, lab, xt, rec['seed'], encoder)
                rec['score_cache'][cache_key] = score
            rows.append(dict(step=step, candidate=name, inner_fold=rec['inner_fold'],
                y_true=rec['y_true'], score=score, y_pred=int(score >= 0)))
    return pd.DataFrame(rows)


def outer_predict(raw, y, train, test, model, aug, selector, seed, encoder,
                  synthetic_cache, selector_cache):
    xtr = snv(raw[train]); xte = snv(raw[test:test+1]); ytr = y[train]
    if selector not in selector_cache:
        selector_cache[selector] = feature_indices(selector, xtr, ytr, seed)
    ids = selector_cache[selector]
    rec = dict(xtr=xtr, ytr=ytr, seed=seed,
               synthetic_cache=synthetic_cache)
    xx, yy = aug_dataset(rec, aug)
    score = train_and_predict(model, representation(xx, ids, model), yy,
                              representation(xte, ids, model), seed, encoder)
    return score

def latex_escape(s):
    return (str(s).replace('\\', r'\textbackslash{}').replace('&', r'\&')
                  .replace('%', r'\%').replace('_',r'\_').replace('#', r'\#'))


def write_latex_table(frame, columns, path, caption, label, numeric=None):
    numeric = numeric or {}
    n = len(columns)
    align = 'l' + 'r'*(n-1)
    env = 'table*' if n >= 6 else 'table'
    lines = ['\\begin{' + env + '}[t]', '\\centering',
             '\\caption{' + caption + '}', '\\label{' + label + '}',
             '\\footnotesize', '\\setlength{\\tabcolsep}{4pt}',
             '\\begin{tabular}{' + align + '}', '\\toprule',
             ' & '.join(latex_escape(h) for _, h in columns) + ' \\\\', '\\midrule']
    for _,row in frame.iterrows():
        out=[]
        for key, _ in columns:
            v=row[key]
            if pd.isna(v): text='--'
            elif key in numeric: text=format(v, numeric[key])
            elif isinstance(v, (float,np.floating)): text=f'{v:.3f}'
            else: text=str(v)
            out.append(latex_escape(text))
        lines.append(' & '.join(out) + ' \\\\')
    lines += ['\\bottomrule','\\end{tabular}','\\end{' + env + '}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def save_stage_tables(candidate_scores, selections, preds):
    stage_labels = {1:'classifier', 2:'augmentation', 3:'feature selection'}
    table_outputs = []
    refs = {1:'PLS-DA', 2:'Real only', 3:'Full spectrum'}
    for step in (1,2,3):
        data = candidate_scores[candidate_scores.step == step].copy()
        grouped = data.groupby('candidate', sort=False).agg(
            ValBA=('BA','mean'), SD_BA=('BA','std'),
            ValAUC=('AUC','mean'), ValAcc=('ACC','mean'),
            NOuter=('outer_fold','nunique')).reset_index()
        baseline = data.loc[data.candidate == refs[step],
                            ['outer_fold','BA']].rename(columns={'BA':'ReferenceBA'})
        pair = data.merge(baseline, on='outer_fold', how='left')
        pair['DeltaBA_pp'] = 100 * (pair.BA - pair.ReferenceBA)
        delta = pair.groupby('candidate',sort=False).agg(
            MeanDeltaBA_pp=('DeltaBA_pp','mean'),
            Wins=('DeltaBA_pp',lambda z:int((z > 1e-12).sum()))).reset_index()
        grouped = grouped.merge(delta, on='candidate',how='left')
        chosen = selections[selections.step == step].candidate.value_counts()
        grouped['Selected'] = grouped.candidate.map(chosen).fillna(0).astype(int)
        grouped = grouped.sort_values(['Selected','ValBA','ValAUC'],ascending=False)
        grouped.to_csv(OUTPUT_DIR/f'0{step}_candidate_summary.csv',index=False)
        tex = OUTPUT_DIR/f'0{step}_candidate_summary.tex'
        write_latex_table(grouped,
            [('candidate','Configuration'),('ValBA','Inner BA'),
             ('SD_BA','BA SD'),('MeanDeltaBA_pp','Delta BA (pp)'),
             ('ValAUC','Inner AUC'),('Selected','Selections')],tex,
            caption=f'Step {step} ({stage_labels[step]}): nested inner-validation '
                    f'candidate comparison; mean paired differences versus {refs[step]}. '
                    'Selections count the outer folds choosing each candidate. '
                    'Inner validation means are descriptive, not outer-test estimates.',
            label=f'tab:step{step}_candidates',numeric={
                'ValBA':'.3f','SD_BA':'.3f','MeanDeltaBA_pp':'+.1f','ValAUC':'.3f'})
        table_outputs.append(tex.name)
    rows=[]
    for step in (1,2,3):
        d=preds[preds.step==step].sort_values('outer_fold')
        rows.append(dict(Step=f'Step {step}', **evaluate_predictions(
            d.y_true, d.y_pred, d.score)))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR/'nested_stage_metrics.csv', index=False)
    write_latex_table(summary,[('Step','Stage'),('Accuracy','Acc.'),
        ('BalancedAccuracy','BA'),('Sensitivity','Sens.'),('Specificity','Spec.'),
        ('AUC','AUC'),('AP','AP'),('F1','F1'),('MCC','MCC'),('Correct','Correct')],
        OUTPUT_DIR/'nested_stage_metrics.tex',
        caption='Patient-level outer-fold results after nested sequential selection. '
             + ('Quick-run diagnostics, not inferential.' if FAST else
                'All original patients are held out once. Accuracy-related '
                'metrics are based only on original patients.'),
        label='tab:nested_stages', numeric={key:'.3f' for key in
         ('Accuracy','BalancedAccuracy','Sensitivity','Specificity','AUC','AP','F1','MCC')})
    table_outputs.append('nested_stage_metrics.tex')
    table_outputs.append('paired_stage_comparisons.tex')
    (OUTPUT_DIR/'paper_tables.tex').write_text(
        '\n'.join('\\input{' + name + '}' for name in table_outputs)+'\n',
        encoding='utf-8')
    return summary

def paired_comparisons(pred):
    result=[]
    groups={s:d.sort_values('outer_fold') for s,d in pred.groupby('step')}
    for a,b in [(1,2),(2,3),(1,3)]:
        z=groups[a]; w=groups[b]
        assert list(z.patient)==list(w.patient)
        y=z.y_true.to_numpy(int); p=z.y_pred.to_numpy(int); q=w.y_pred.to_numpy(int)
        wins=int(((p!=y)&(q==y)).sum()); losses=int(((p==y)&(q!=y)).sum())
        rng=np.random.default_rng(SEED+101*a+11*b)
        n=np.flatnonzero(y==0); pos=np.flatnonzero(y==1)
        draw_n=rng.choice(n, (N_BOOTSTRAP,len(n)),replace=True)
        draw_p=rng.choice(pos,(N_BOOTSTRAP,len(pos)),replace=True)
        diffs=50*((q[draw_n]==y[draw_n]).mean(axis=1)-(p[draw_n]==y[draw_n]).mean(axis=1)+
                  (q[draw_p]==y[draw_p]).mean(axis=1)-(p[draw_p]==y[draw_p]).mean(axis=1))
        exact=binomtest(min(wins,losses),n=wins+losses,p=0.5).pvalue if wins+losses else 1.0
        result.append(dict(Reference=f'Step {a}',Candidate=f'Step {b}',
                           DeltaBA_pp=float(100*(balanced_accuracy_score(y,q)-balanced_accuracy_score(y,p))),
                           CI_low_pp=float(np.quantile(diffs,.025)),
                           CI_high_pp=float(np.quantile(diffs,.975)),
                           DeltaSensitivity_pp=float(100*(np.mean(q[y==1]==1)-np.mean(p[y==1]==1))),
                           DeltaSpecificity_pp=float(100*(np.mean(q[y==0]==0)-np.mean(p[y==0]==0))),
                           DeltaAUC=float(roc_auc_score(y,w.score)-roc_auc_score(y,z.score)),
                           Corrected=wins,NewErrors=losses,
                           ExactMcNemarP=np.nan if FAST else exact))
    result=pd.DataFrame(result)
    if not FAST:
        x=result.ExactMcNemarP.to_numpy(); order=np.argsort(x)
        out=np.empty_like(x);out[order]=np.minimum(1,np.maximum.accumulate(x[order]*(len(x)-np.arange(len(x)))))
        result['HolmP']=out
    else:
        result['HolmP']=np.nan
    result.to_csv(OUTPUT_DIR/'paired_stage_comparisons.csv',index=False)
    write_latex_table(result, [('Reference','Reference'),('Candidate','Candidate'),
        ('DeltaBA_pp','Delta BA (pp)'),('CI_low_pp','CI low'),
        ('CI_high_pp','CI high'),('Corrected','Recovered'),('NewErrors','New errors'),
        ('DeltaAUC','Delta AUC'),('HolmP','Holm p')],OUTPUT_DIR/'paired_stage_comparisons.tex',
        caption='Paired out-of-fold change in balanced accuracy for sequential stages.',
        label='tab:paired_stage_changes',
        numeric={s:'.3f' for s in ('DeltaBA_pp','CI_low_pp','CI_high_pp','DeltaAUC','HolmP')})
    return result

def fixed_benchmark_comparisons(raw, y, outer_folds):
    """Evaluate fixed, prespecified recipes on the exact same held-out patients.

    These recipes DO NOT participate in stage selection. The original pipeline
    was historically chosen using this cohort: its performance remains exploratory.
    """
    rows=[]
    for outer_fold, train, test in outer_folds:
        seed=SEED + outer_fold * 100003
        cache, selection_cache = {}, {}
        for model_name, mod, aug, sel in [
             ('Real-only PLS-DA', 'PLS-DA','Real only','Full spectrum'),
             ('Real-only Elastic-net', 'Elastic-net LR','Real only','Full spectrum'),
             ('Flow + ANOVA256 + PLS-DA','PLS-DA',
               f'Latent Flow {min(30,max(AUG_COUNTS))}/class','ANOVA-256')]:
            margin=outer_predict(raw,y,train,test,mod,aug,sel,seed,None,
                                 cache, selection_cache)
            rows.append(dict(outer_fold=outer_fold,patient=int(test),
                benchmark=model_name,y_true=int(y[test]),
                score=margin,y_pred=int(margin>=0)))
    d=pd.DataFrame(rows)
    d.to_csv(OUTPUT_DIR/'fixed_benchmark_predictions.csv',index=False)
    metrics=[]
    for name,sub in d.groupby('benchmark',sort=False):
        metrics.append(dict(Benchmark=name,**evaluate_predictions(
            sub.y_true,sub.y_pred,sub.score)))
    metrics=pd.DataFrame(metrics)
    metrics.to_csv(OUTPUT_DIR/'fixed_benchmark_metrics.csv',index=False)
    write_latex_table(metrics,[('Benchmark','Fixed benchmark'),
        ('Accuracy','Acc.'),('BalancedAccuracy','BA'),('Sensitivity','Sens.'),
        ('Specificity','Spec.'),('AUC','AUC'),('F1','F1'),('MCC','MCC'),
        ('Correct','Correct')],OUTPUT_DIR/'fixed_benchmark_metrics.tex',
        caption='Prespecified fixed benchmark recipes under the same patient-level '
        'outer holdouts. The historical Flow + ANOVA-256 recipe was chosen on '
        'this cohort in earlier work and is therefore not an independent comparator.',
        label='tab:fixed_benchmarks',numeric={key:'.3f' for key in
         ('Accuracy','BalancedAccuracy','Sensitivity','Specificity','AUC','F1','MCC')})
    return d,metrics


def compare_against_fixed(pred, basepred):
    final=pred[pred.step==3].sort_values('outer_fold')
    rows=[]
    for name,b in basepred.groupby('benchmark'):
        b=b.sort_values('outer_fold')
        if list(final.outer_fold)!=list(b.outer_fold):
            raise AssertionError('Misaligned held-out patient folds')
        y=final.y_true.to_numpy(int)
        a=b.y_pred.to_numpy(int); z=final.y_pred.to_numpy(int)
        wins=int(np.sum((a!=y)&(z==y)))
        losses=int(np.sum((a==y)&(z!=y)))
        rng=np.random.default_rng(SEED+len(rows)*17+832)
        neg=np.where(y==0)[0];pos=np.where(y==1)[0]
        neg_draw=rng.choice(neg,(N_BOOTSTRAP,len(neg)),replace=True)
        pos_draw=rng.choice(pos,(N_BOOTSTRAP,len(pos)),replace=True)
        diff=50*((z[neg_draw]==y[neg_draw]).mean(1)-(a[neg_draw]==y[neg_draw]).mean(1)+
                 (z[pos_draw]==y[pos_draw]).mean(1)-(a[pos_draw]==y[pos_draw]).mean(1))
        pval=binomtest(min(wins,losses),n=wins+losses,p=.5).pvalue if wins+losses else 1.
        rows.append(dict(Benchmark=name,
            DeltaBA_pp=100*(balanced_accuracy_score(y,z)-balanced_accuracy_score(y,a)),
            CI_low_pp=np.percentile(diff,2.5),CI_high_pp=np.percentile(diff,97.5),
            DeltaAUC=roc_auc_score(y,final.score)-roc_auc_score(y,b.score),
            Recovered=wins,NewErrors=losses,
            ExactMcNemarP=float(pval) if not FAST else np.nan))
    out=pd.DataFrame(rows)
    if FAST:
        out['HolmP']=np.nan
    else:
        x=out.ExactMcNemarP.to_numpy();ix=np.argsort(x)
        adj=np.maximum.accumulate(x[ix]*(len(x)-np.arange(len(x))))
        out['HolmP']=0.
        out.loc[ix,'HolmP']=np.minimum(1,adj)
    out.to_csv(OUTPUT_DIR/'fixed_benchmark_comparisons.csv',index=False)
    write_latex_table(out,[('Benchmark','Reference'),('DeltaBA_pp','Delta BA (pp)'),
        ('CI_low_pp','95% low'),('CI_high_pp','95% high'),('DeltaAUC','Delta AUC'),
        ('Recovered','Recovered'),('NewErrors','New errors'),('HolmP','Holm p')],
        OUTPUT_DIR/'fixed_benchmark_comparisons.tex',
        caption='Paired differences: nested Step 3 minus each fixed benchmark. '
        'Confidence intervals are stratified patient bootstraps; multiplicity '
        'correction uses Holm-adjusted exact McNemar tests. Given 17 patients, '
        'all inferences remain exploratory.',
        label='tab:fixed_benchmark_delta',numeric={k:'.3f' for k in
             ('DeltaBA_pp','CI_low_pp','CI_high_pp','DeltaAUC','HolmP')})
    return out

def tidy(ax):
    ax.spines['left'].set_color(PALETTE['light'])
    ax.spines['bottom'].set_color(PALETTE['light'])
    ax.tick_params(length=0, pad=5)
    ax.set_facecolor('white')


def descriptive_ci(y, preds, reps=800):
    """Stratified patient bootstrap CI, not an independent validation CI."""
    yy=np.asarray(y,int); pp=np.asarray(preds,int)
    neg=np.flatnonzero(yy==0);pos=np.flatnonzero(yy==1)
    rng=np.random.default_rng(SEED+1981)
    n=rng.choice(neg,(reps,len(neg)),replace=True)
    p=rng.choice(pos,(reps,len(pos)),replace=True)
    result=.5*((pp[n]==yy[n]).mean(axis=1)+(pp[p]==yy[p]).mean(axis=1))
    return np.percentile(result*100,[2.5,97.5])


def plot_stage_results(scores, selections, metrics, pred):
    ref={1:'PLS-DA',2:'Real only',3:'Full spectrum'}
    labels={1:'Classifier',2:'Augmentation',3:'Feature selection'}
    colors={1:PALETTE['navy'],2:PALETTE['teal'],3:PALETTE['blue']}
    for stage in (1,2,3):
        s=scores[scores.step==stage].copy()
        summary=s.groupby('candidate',sort=False).agg(
           mean=('BA','mean'),sd=('BA','std')).reset_index()
        freq=selections[selections.step==stage].candidate.value_counts()
        summary['freq']=summary.candidate.map(freq).fillna(0).astype(int)
        summary=summary.sort_values(['freq','mean'],ascending=False).reset_index(drop=True)
        selected=summary.candidate.to_list()
        mat=s.pivot(index='candidate',columns='outer_fold',values='BA').loc[selected]
        delta=mat.sub(mat.loc[ref[stage]],axis=1)*100
        n=len(selected); y=np.arange(n)
        fig=plt.figure(figsize=(13.0,max(8.1, 4.8 + n*0.37)),facecolor='white')
        gs=fig.add_gridspec(2,2,height_ratios=[1.0,1.02],width_ratios=[1.55,1.0],
                            left=.195,right=.945,top=.905,bottom=.09,hspace=.30,wspace=.23)
        a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);
        c=fig.add_subplot(gs[1,0]);d=fig.add_subplot(gs[1,1])
        for ax in [a,b,d]:tidy(ax)
        for i,row in summary.iterrows():
            points=mat.loc[row.candidate].to_numpy()*100
            a.plot(points,np.full_like(points,i,dtype=float), 'o', ms=3.3,
                   color=colors[stage],alpha=.24,zorder=2)
            sd=0. if np.isnan(row.sd) else row.sd*100
            a.plot([max(0,100*row['mean']-sd),min(100,100*row['mean']+sd)],
                   [i,i],color=PALETTE['muted'],lw=2.3,zorder=3)
            a.scatter(100*row['mean'],i,color=colors[stage],s=50,zorder=4,
                      edgecolor='white',linewidth=.9)
        a.set_yticks(y);a.set_yticklabels(selected,fontsize=9)
        a.set_ylim(n-.5,-.5);a.set_xlim(0,105)
        a.set_xlabel('Inner-CV balanced accuracy (%)')
        a.text(0,1.035,'A  Candidate accuracy',transform=a.transAxes,
               fontsize=10,weight='bold',color=PALETTE['ink'])
        for i,row in summary.iterrows():
            b.plot([0,row.freq],[i,i],color=PALETTE['light'],lw=3,zorder=1)
            b.scatter(row.freq,i,s=54,color=colors[stage],zorder=2)
            b.text(row.freq+.13,i,str(row.freq),fontsize=9,va='center')
        b.set_ylim(n-.5,-.5);b.set_xlim(0,max(1,summary.freq.max())+2.4)
        b.set_yticks(y);b.set_yticklabels([])
        b.set_xlabel('Outer folds selecting this candidate')
        b.text(0,1.035,'B  Selection frequency',transform=b.transAxes,
               fontsize=10,weight='bold',color=PALETTE['ink'])
        lim=max(3.,float(np.nanmax(abs(delta.to_numpy()))))
        from matplotlib.colors import TwoSlopeNorm
        cmap=plt.get_cmap('RdBu')
        image=c.imshow(delta.to_numpy(),cmap=cmap,
                       norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
        c.set_yticks(y);c.set_yticklabels(selected,fontsize=8.3)
        c.set_xticks(np.arange(mat.shape[1]));c.set_xticklabels(
            [str(z) for z in mat.columns],fontsize=7.7)
        c.set_xlabel('Outer-fold index (each entry is an inner-CV result)')
        c.text(0,1.035,'C  Paired gain relative to '+ref[stage],transform=c.transAxes,
               fontsize=10,weight='bold',color=PALETTE['ink'])
        cb=fig.colorbar(image,ax=c,orientation='horizontal',fraction=.07,pad=.19,shrink=.75)
        cb.set_label('Change in inner balanced accuracy (pp)',fontsize=8.6)
        cb.outline.set_visible(False)
        diffs=delta.mean(axis=1).to_numpy()
        d.axvline(0,color=PALETTE['muted'],lw=1)
        d.barh(y,diffs,color=[PALETTE['teal'] if v>=0 else PALETTE['coral'] for v in diffs],
               height=.62,alpha=.85)
        d.set_ylim(n-.5,-.5)
        d.set_yticks(y);d.set_yticklabels([])
        d.set_xlabel('Mean paired inner BA change (pp)')
        spread=max(3.5,max(abs(diffs)) *1.25)
        d.set_xlim(-spread,spread)
        for i,v in enumerate(diffs):
            d.text(v+(.3 if v>=0 else -.3),i,f'{v:+.1f}',
                   ha='left' if v>=0 else 'right',va='center',fontsize=8)
        d.text(0,1.035,'D  Average paired change',transform=d.transAxes,
               fontsize=10,weight='bold',color=PALETTE['ink'])
        fig.suptitle(f'Step {stage}  |  {labels[stage]} ablation',
                     x=.195,ha='left',fontsize=14,weight='bold',color=PALETTE['ink'])
        fig.text(.195,.026,'Descriptive inner cross-validation summaries; the selected candidate '
                 'is refitted before the outer patient is evaluated.',
                 fontsize=8,color=PALETTE['muted'])
        filename = f'0{stage}_' + {1:'classifier_ablation',
            2:'augmentation_ablation',3:'feature_selection_ablation'}[stage] + '.pdf'
        fig.savefig(OUTPUT_DIR / filename,bbox_inches='tight')
        plt.close(fig)

    # Nested progression with class-specific sensitivity/specificity trade-offs.
    fig,ax=plt.subplots(figsize=(9.0,4.9))
    x=np.arange(3)
    metric_specs=[('BalancedAccuracy','Balanced accuracy',PALETTE['navy']),
        ('AUC','ROC-AUC',PALETTE['blue']),
        ('Sensitivity','Sensitivity',PALETTE['coral']),
        ('Specificity','Specificity',PALETTE['teal']),
        ('AP','Average precision',PALETTE['amber'])]
    for key,name,color in metric_specs:
        v=metrics[key].to_numpy()*100
        ax.plot(x,v,'o-',color=color,lw=2.1,ms=6.5,label=name,
                markeredgecolor='white',markeredgewidth=.9)
        ax.text(x[-1]+.09,v[-1],f'{v[-1]:.1f}',color=color,va='center',fontsize=8.7)
    ax.set_xticks(x);ax.set_xticklabels(['Step 1: Classifier','Step 2: Augmentation',
                                         'Step 3: Features'])
    ax.set_xlim(-.15,2.39);ax.set_ylim(0,106)
    ax.set_ylabel('Held-out patient metric (%)')
    ax.legend(loc='upper left',ncol=3,frameon=False,fontsize=8.5)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/'04_nested_stage_performance.pdf',bbox_inches='tight')
    plt.close(fig)

    # ROC and precision-recall use held-out continuous margins only.
    fig,(a,b)=plt.subplots(1,2,figsize=(10.2,4.7),sharey=False)
    for stage in (1,2,3):
        d=pred[pred.step==stage].sort_values('outer_fold')
        fpr,tpr,_=roc_curve(d.y_true,d.score)
        precision,recall,_=precision_recall_curve(d.y_true,d.score)
        col=colors[stage]
        a.step(fpr,tpr,where='post',color=col,lw=2.1,
               label=f'Step {stage}: AUC {roc_auc_score(d.y_true,d.score):.3f}')
        b.step(recall,precision,where='post',color=col,lw=2.1,
               label=f'Step {stage}: AP {average_precision_score(d.y_true,d.score):.3f}')
    a.plot([0,1],[0,1],ls='--',lw=1,color=PALETTE['muted'])
    base=float(np.mean(pred[pred.step==1].y_true))
    b.axhline(base,color=PALETTE['muted'],ls='--',lw=1)
    a.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',
          ylabel='True positive rate')
    b.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall (diseased)',
          ylabel='Precision (diseased)')
    a.legend(loc='upper left',frameon=False,fontsize=8.5)
    b.legend(loc='upper left',frameon=False,fontsize=8.5)
    for ax,t in [(a,'A  Receiver operating characteristic'),
                 (b,'B  Precision–recall')]:
        tidy(ax)
        ax.set_title(t,loc='left',weight='bold',fontsize=10)
    fig.tight_layout(w_pad=3)
    fig.savefig(OUTPUT_DIR/'05_stage_roc.pdf',bbox_inches='tight')
    plt.close(fig)

    # Patient-level correctness matrix with patient names and legend.
    truth=pred[pred.step==1].sort_values('outer_fold')
    ids=truth.outer_fold.to_numpy()
    yp=truth.y_true.to_numpy(int)
    mat=[]
    for st in (1,2,3):
        d=pred[pred.step==st].sort_values('outer_fold')
        mat.append((d.y_pred.to_numpy(int)==yp).astype(int))
    mat=np.array(mat)
    from matplotlib.colors import ListedColormap
    fig,ax=plt.subplots(figsize=(max(8, .61*len(ids)+3.2),3.0))
    ax.imshow(mat,cmap=ListedColormap([PALETTE['coral'],PALETTE['teal']]),
              vmin=0,vmax=1,aspect='auto')
    ax.set_yticks(range(3));ax.set_yticklabels(['Step 1','Step 2','Step 3'])
    ax.set_xticks(np.arange(len(ids)))
    ax.set_xticklabels([str(int(i)) for i in ids],fontsize=8)
    ax.set_xlabel('Outer-fold index (original held-out patient)')
    for i in range(3):
        for j in range(len(ids)):
            ax.text(j,i,'✓' if mat[i,j] else '×',ha='center',va='center',
                    color='white',weight='bold',fontsize=11)
    ax.set_yticks(np.arange(-.5,3,.5),minor=False) if False else None
    ax.tick_params(length=0)
    ax.set_title('Patient-level correctness across sequential stages',
                 loc='left',fontsize=11,weight='bold',pad=16)
    fig.text(.095,.02,'Teal = correct  |  Coral = incorrect; held-out patients are never synthetic.',
             fontsize=8.5,color=PALETTE['muted'])
    fig.tight_layout(rect=[0,.055,1,1])
    fig.savefig(OUTPUT_DIR/'06_patient_errors.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_selection_paths(selections):
    df=selections.pivot(index='outer_fold',columns='step',values='candidate').sort_index()
    vals=sorted(set(df.stack()))
    colormap=plt.get_cmap('tab20')
    colors={z:colormap(i%20) for i,z in enumerate(vals)}
    n=len(df)
    fig,ax=plt.subplots(figsize=(11.8,max(4,1.9+.49*n)))
    for j,col in enumerate((1,2,3)):
        for i,txt in enumerate(df[col]):
            from matplotlib.patches import FancyBboxPatch
            xx=j*1.04
            patch=FancyBboxPatch((xx,i-.36),.98,.75,
                boxstyle='round,pad=0.02,rounding_size=0.09',
                fc=colors[txt],ec='white',lw=.8,alpha=.88)
            ax.add_patch(patch)
            clr='white' if np.mean(colors[txt][:3])<.62 else PALETTE['ink']
            name=(txt.replace('Latent Flow','Flow')
                     .replace('Spectral Mixup','Mixup')
                     .replace('PCA-Gaussian','PCA-G')
                     .replace('MultiRocket-Hydra','Rocket+Hydra')
                     .replace('Stability Elastic-Net','Stability'))
            ax.text(xx+.49,i+.02,name,ha='center',va='center',
                    fontsize=7.6,color=clr,clip_on=True)
    ax.set_xlim(-.04,3.1);ax.set_ylim(n-.45,-.55)
    ax.set_yticks(np.arange(n));ax.set_yticklabels(df.index.astype(str))
    ax.set_ylabel('Outer-fold index')
    ax.set_xticks([.49,1.53,2.57]);ax.set_xticklabels(['Classifier','Augmentation','Feature selection'],
                                                      weight='bold')
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=5)
    for z in ax.spines.values():z.set_visible(False)
    ax.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/'07_sequential_selection_paths.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_feature_frequency(wn,raw,y,feature_rows,nouter):
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(10.8,6.3),sharex=True,
                            gridspec_kw={'height_ratios':[1.4,1.0]})
    x=snv(raw)
    ax.plot(wn,x[y==0].mean(axis=0),lw=1.5,color=PALETTE['navy'],label='Healthy mean')
    ax.plot(wn,x[y==1].mean(axis=0),lw=1.5,color=PALETTE['coral'],label='Diseased mean')
    ax.set_ylabel('SNV spectral intensity')
    ax.legend(loc='upper left',frameon=False,ncol=2)
    if len(feature_rows):
        d=pd.DataFrame(feature_rows).groupby('channel').agg(
            freq=('outer_fold','nunique'),wavenumber=('wavenumber','first')).reset_index()
        d['freq']=d.freq/nouter
        ax2.vlines(d.wavenumber,0,d.freq,color=PALETTE['teal'],linewidth=.8,alpha=.8)
        top=d.nlargest(12,'freq').sort_values('wavenumber')
        ax2.scatter(top.wavenumber,top.freq,color=PALETTE['navy'],s=15,zorder=3)
        d.to_csv(OUTPUT_DIR/'selected_wavenumber_frequency.csv',index=False)
    else:
        ax2.text(.5,.5,'All selected pipelines retained the full spectrum',
                 ha='center',va='center',transform=ax2.transAxes)
        pd.DataFrame(columns=['channel','freq','wavenumber']).to_csv(
            OUTPUT_DIR/'selected_wavenumber_frequency.csv',index=False)
    ax2.set_ylim(0,1.03)
    ax2.set_ylabel('Outer-fold selection frequency')
    ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
    for a in (ax,ax2):tidy(a)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(OUTPUT_DIR/'08_wavenumber_selection_frequency.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_paired_stage_forest(paired):
    x=np.arange(len(paired))[::-1]
    means=paired.DeltaBA_pp.to_numpy(float)
    left=means-paired.CI_low_pp.to_numpy(float)
    right=paired.CI_high_pp.to_numpy(float)-means
    fig,ax=plt.subplots(figsize=(8.8,3.3))
    ax.axvline(0,color=PALETTE['muted'],lw=1,ls='--')
    ax.errorbar(means,x,xerr=np.maximum(np.vstack([left,right]),0),
                fmt='o',color=PALETTE['navy'],ecolor=PALETTE['muted'],
                ms=6,capsize=3,lw=1.4)
    ax.set_yticks(x)
    ax.set_yticklabels((paired.Candidate+' vs '+paired.Reference).to_list(),fontsize=9)
    ax.set_xlabel('Change in held-out balanced accuracy (pp)')
    ax.set_ylim(-.6,len(x)-.4)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR/'09_paired_stage_forest.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_stage_confusion(pred):
    from matplotlib.colors import ListedColormap
    fig,axs=plt.subplots(1,3,figsize=(9.2,3.4))
    for st,ax in enumerate(axs,1):
        d=pred[pred.step==st]
        cm=confusion_matrix(d.y_true,d.y_pred,labels=[0,1])
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(2,cm.max()))
        ax.set_xticks([0,1]);ax.set_xticklabels(['Healthy','Diseased'],fontsize=8)
        ax.set_yticks([0,1]);ax.set_yticklabels(['Healthy','Diseased'],fontsize=8)
        ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
        ax.set_title(f'Step {st}',fontsize=10,weight='bold')
        for i in (0,1):
            for j in (0,1):
                ax.text(j,i,str(cm[i,j]),fontsize=14,weight='bold',ha='center',
                        va='center',color='white' if cm[i,j]>cm.max()*.55 else PALETTE['ink'])
        ax.tick_params(length=0)
        for sp in ax.spines.values():sp.set_visible(False)
    fig.tight_layout(w_pad=2.)
    fig.savefig(OUTPUT_DIR/'10_stage_confusion_matrices.pdf',bbox_inches='tight')
    plt.close(fig)


def plot_fixed_benchmarks(stage_metrics,base_metrics,changes):
    combined=pd.concat([base_metrics.assign(Group='Fixed'),
                       stage_metrics[stage_metrics.Step=='Step 3'].rename(
                           columns={'Step':'Benchmark'}).assign(Group='Sequential')],
                       ignore_index=True)
    combined=combined.sort_values('BalancedAccuracy',ascending=False)
    colors=[PALETTE['teal'] if x=='Sequential' else PALETTE['muted']
            for x in combined.Group]
    n=len(combined)
    fig,(a,b)=plt.subplots(1,2,figsize=(11.5,4.5),
                           gridspec_kw={'width_ratios':[1.25,.92]})
    yi=np.arange(n)
    a.barh(yi,100*combined.BalancedAccuracy,color=colors,height=.62)
    a.set_yticks(yi);a.set_yticklabels(combined.Benchmark,fontsize=8.5)
    a.set_xlim(0,106);a.invert_yaxis()
    a.set_xlabel('Outer-test balanced accuracy (%)')
    for i,v in enumerate(combined.BalancedAccuracy):
        a.text(100*v+1,i,f'{v*100:.1f}',va='center',fontsize=8.2)
    d=changes.sort_values('DeltaBA_pp')
    yb=np.arange(len(d))
    b.axvline(0,color=PALETTE['muted'],lw=1,ls='--')
    for i,r in d.iterrows():
        j=list(d.index).index(i)
        b.plot([r.CI_low_pp,r.CI_high_pp],[j,j],lw=2,color=PALETTE['muted'])
        b.scatter(r.DeltaBA_pp,j,color=PALETTE['teal'],s=50,zorder=2)
    b.set_yticks(yb);b.set_yticklabels(d.Benchmark,fontsize=8)
    b.set_xlabel('Sequential Step 3 minus reference (BA pp)')
    b.set_ylim(-.5,len(d)-.5)
    tidy(a);tidy(b)
    a.set_title('Fixed benchmarks vs selected pipeline',loc='left',fontsize=10,weight='bold')
    b.set_title('Paired differences (95% bootstrap CI)',loc='left',fontsize=10,weight='bold')
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUTPUT_DIR/'11_fixed_benchmark_comparison.pdf',bbox_inches='tight')
    plt.close(fig)


def run_experiment():
    if RUN_AEON:
        try:
            from aeon.transformations.collection.interval_based import QUANTTransformer
            from aeon.classification.convolution_based import MultiRocketHydraClassifier
        except ImportError as err:
            raise ImportError('aeon is required for QUANT and MultiRocket-Hydra; '
                'install aeon==1.0.0 for Python 3.9. To only smoke-test the '
                'non-aeon models, set RUN_AEON=False.') from err
    seed_everything(SEED)
    torch.set_num_threads(min(3, torch.get_num_threads()))
    wn, raw, y, names=load_data()
    if len(AUG_COUNTS)<2 or min(AUG_COUNTS)<=0 or list(AUG_COUNTS)!=sorted(set(AUG_COUNTS)):
        raise ValueError('AUG_COUNTS must contain at least two increasing positive integers')
    if len(FEATURE_K)!=3 or list(FEATURE_K)!=sorted(set(FEATURE_K)):
        raise ValueError('FEATURE_K must contain three increasing values')
    encoder=freeze_encoder()
    if FAST:
        rng=np.random.default_rng(SEED)
        outer_test=np.concatenate([rng.choice(np.flatnonzero(y==k),
                    size=min(OUTER_QUICK_PER_CLASS,int((y==k).sum())),replace=False)
                    for k in (0,1)])
    else:
        outer_test=np.arange(len(y))
    candidate_scores=[]; selected_rows=[]; predictions=[]; features=[]
    all_outer_folds=[]
    outer_test_set=set(outer_test.tolist())
    for outer_fold, (train,held) in enumerate(LeaveOneOut().split(raw),1):
        test=int(held[0])
        if test not in outer_test_set: continue
        all_outer_folds.append((outer_fold,train.copy(),test))
        inner=make_inner_records(raw,y,train,outer_fold)
        held_labels=[r['y_true'] for r in inner]
        if min(np.bincount(held_labels,minlength=2)) < 1:
            raise ValueError('Inner CV must contain both healthy and sick held-out patients')
        outer_seed=SEED+outer_fold*100003
        outer_flow={};outer_selectors={}
        # Step 1: model choice on REAL spectra, no selection.
        a=evaluate_candidates(inner,MODELS,None,'Real only','Full spectrum',1,
                              encoder,'model')
        model, by=select_winner(a,MODELS)
        # Step 2: model fixed, choose augmentation.
        b=evaluate_candidates(inner,AUGMENTATIONS,model,None,'Full spectrum',2,
                              encoder,'augmentation')
        aug, by2=select_winner(b,AUGMENTATIONS)
        # Step 3: classifier and augmentation fixed, choose feature selector.
        c=evaluate_candidates(inner,SELECTORS,model,aug,None,3,encoder,'selector')
        selector,by3=select_winner(c,SELECTORS)
        for step, chosen, scores, byscore in [(1,model,a,by),(2,aug,b,by2),
                                              (3,selector,c,by3)]:
            selected_rows.append(dict(outer_fold=outer_fold,held_out=names[test],
                                      step=step,candidate=chosen,inner_BA=byscore[chosen]['BA'],
                                      inner_AUC=byscore[chosen]['AUC']))
            for cand,vals in byscore.items():
                candidate_scores.append(dict(outer_fold=outer_fold,step=step,
                                             candidate=cand,**vals))
        for step, mod, ag, sel in [(1,model,'Real only','Full spectrum'),
                                    (2,model,aug,'Full spectrum'),
                                    (3,model,aug,selector)]:
            score=outer_predict(raw,y,train,test,mod,ag,sel,outer_seed,encoder,
                                outer_flow,outer_selectors)
            predictions.append(dict(outer_fold=outer_fold,patient=names[test],
                                    step=step,model=mod,augmentation=ag,selector=sel,
                                    y_true=int(y[test]),y_pred=int(score>=0),score=score))
        if selector!='Full spectrum':
            ids=outer_selectors[selector]
            for j in ids:
                features.append(dict(outer_fold=outer_fold,selector=selector,
                                     channel=int(j),wavenumber=float(wn[j])))
        if not FAST:
            print(f'outer fold {len(set(row["outer_fold"] for row in selected_rows))}/'
                  f'{len(outer_test)}: {names[test]} => {model} | {aug} | {selector}',flush=True)
    sel=pd.DataFrame(selected_rows); val=pd.DataFrame(candidate_scores)
    pred=pd.DataFrame(predictions)
    sel.to_csv(OUTPUT_DIR/'selected_configurations_by_outer_fold.csv',index=False)
    val.to_csv(OUTPUT_DIR/'inner_cv_candidate_scores.csv',index=False)
    pred.to_csv(OUTPUT_DIR/'nested_patient_predictions.csv',index=False)
    pd.DataFrame(features,columns=['outer_fold','selector','channel','wavenumber']).to_csv(
        OUTPUT_DIR/'selected_features_by_outer_fold.csv',index=False)
    metrics=save_stage_tables(val,sel,pred)
    paired=paired_comparisons(pred)
    plot_stage_results(val,sel,metrics,pred)
    plot_selection_paths(sel)
    plot_feature_frequency(wn,raw,y,features,len(outer_test))
    plot_paired_stage_forest(paired)
    plot_stage_confusion(pred)
    if RUN_FIXED_BENCHMARKS:
        basepred,basemetrics=fixed_benchmark_comparisons(raw,y,all_outer_folds)
        basedelta=compare_against_fixed(pred,basepred)
        plot_fixed_benchmarks(metrics,basemetrics,basedelta)
        with (OUTPUT_DIR/'paper_tables.tex').open('a',encoding='utf-8') as f:
            f.write('\\input{fixed_benchmark_metrics.tex}\n')
            f.write('\\input{fixed_benchmark_comparisons.tex}\n')
    config=dict(mode=RUN_MODE,n_total_patients=len(y),n_tested_patients=len(outer_test),
        n_healthy=int((y==0).sum()),n_sick=int((y==1).sum()),channels=raw.shape[1],
        root=str(ROOT), healthy_directory=str(HEALTHY_DIR),sick_directory=str(SICK_DIR),
        classifiers=MODELS,augmentations=AUGMENTATIONS,selectors=SELECTORS,
        include_reference_controls=INCLUDE_REFERENCE_CONTROLS,
        run_fixed_benchmarks=RUN_FIXED_BENCHMARKS,feature_k=FEATURE_K,
        selection_criterion='inner LOOCV balanced_accuracy -> ROC_AUC -> accuracy',
        outer_protocol='patient-level LOOCV with nested sequential inner selection',
        results_status='SMOKE_TEST_NOT_FOR_PUBLICATION' if FAST else 'EXPLORATORY_NESTED_LOOCV',
        auc_note='Across fold-selected models, AUC uses training-std-standardized margins; not calibrated probabilities.',
        selected_wavenumber_note='For interval/convolution/encoder models selected coordinates are linearly interpolated back to the fixed spectral grid; these comparisons assess reconstructed signal representations, not direct variable restriction.',
        flow_steps=FLOW_STEPS,flow_ode_steps=FLOW_ODE_STEPS,
        augmentation_counts=AUG_COUNTS,seed=SEED,python=sys.version,
        aeon_version=getattr(sys.modules.get('aeon'),'__version__',None))
    (OUTPUT_DIR/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Finished {RUN_MODE}: {len(outer_test)} outer patients, outputs in {OUTPUT_DIR}')
    return pred,metrics,sel,val,paired


if __name__ == '__main__':
    run_experiment()
