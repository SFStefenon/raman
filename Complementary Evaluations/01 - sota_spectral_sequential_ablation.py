"""Sequential three-stage spectral ablation with nested patient-level validation.

ONE SOURCE FILE; switch RUN_MODE = 'quick' / 'complete' below.
Input folders: ROOT/saudaveis/*.csv and ROOT/doentes/*.csv.
Each CSV corresponds to ONE original patient, 2 columns: wavenumber, signal.

At each outer held-out original patient:
  Step 1: inner patient-level CV chooses the classifier on SNV real spectra.
  Step 2: with classifier fixed, inner CV chooses No augmentation or Latent Flow
          with a low/high per-class synthetic spectrum count.
  Step 3: with classifier and augmentation fixed, inner CV chooses Full spectra,
          mRMR, stability-selected Elastic Net, or PCA-loading peaks.
Each chosen intermediate and final configuration is then refitted using ONLY
outer training patients and predicts ONLY the one held-out original patient.
The outer patient is NEVER used in any of the three configuration selections.

Outputs: stage-specific CSVs, LaTeX booktabs tables, one PDF figure per step,
additional ROC, paired comparison and per-patient plots, all inside OUTPUT_DIR.
Mode quick = smoke check on 4 held-out patients, small inner CV; don't publish.
Mode complete = all patient-level outer and inner leave-one-out folds.

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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
from torch import nn
from torch.nn import functional as F

# USER SETTINGS: change only RUN_MODE to switch between the two workflows.
# In Spyder, explicit assignment prevents previous console variables from
# silently overriding the selected mode on repeated runs.
RUN_MODE = 'complete'                    # 'quick' or 'complete'
RUN_AEON = True                      # requires aeon; keep True for paper
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
AUG_COUNTS = tuple(OVERRIDES.get('AUG_COUNTS', (4, 8) if FAST else (15, 30)))
MRMR_K = int(OVERRIDES.get('MRMR_K', 20 if FAST else 256))
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
    'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})

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


def mrmr_indices(X_real, y_real, k, n_bins=3):
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
    return np.sort(np.array(selected, dtype=int))


def stability_indices(X_real, y_real, k, seed):
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
    return np.sort(best), count / STABILITY_BOOTSTRAPS


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


def predict_quant(X_train, y_train, X_test):
    from aeon.transformations.collection.interval_based import QUANTTransformer
    transformer = QUANTTransformer(interval_depth=QUANT_DEPTH, quantile_divisor=4)
    train = array_from_aeon(transformer.fit_transform(X_train[:, None, :]))
    test = array_from_aeon(transformer.transform(X_test[:, None, :]))
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0,
                                                       class_weight='balanced'))
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
        return predict_quant(X_train, y_train, X_test)
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


# Ordered names used to break exact cross-validation ties deterministically.
SELECTORS = ['Full spectrum', 'mRMR', 'Stability Elastic-Net', 'PCA peaks']
AUGMENTATIONS = ['Real only'] + [f'Latent Flow {n}/class' for n in AUG_COUNTS]
MODELS = (['QUANT + Ridge', 'MultiRocket-Hydra'] if RUN_AEON else []) + [
    'PLS-DA', 'Elastic-net LR'] + ([ 'Frozen encoder + Linear SVM'] if PRETRAINED_ENCODER_PATH else [])


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
    if selector == 'Full spectrum':
        return None
    if selector == 'mRMR':
        return mrmr_indices(x_real, y_real, MRMR_K, MRMR_BINS)
    if selector == 'Stability Elastic-Net':
        return stability_indices(x_real, y_real, STABILITY_K, seed)[0]
    if selector == 'PCA peaks':
        return peak_indices(x_real, PEAK_K, seed)
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


def aug_dataset(record, aug):
    if aug == 'Real only':
        return record['xtr'], record['ytr']
    n = int(aug.split()[2].split('/')[0])
    if record['synthetic'] is None:
        record['synthetic'] = generate_latent_flow(
            record['xtr'], record['ytr'], record['seed'], max(AUG_COUNTS))
    syn, lab = record['synthetic']
    nmax = max(AUG_COUNTS)
    ii = np.r_[0:n, nmax:nmax+n]
    return np.vstack([record['xtr'], syn[ii]]), np.r_[record['ytr'], lab[ii]]


def make_inner_records(x, y, outer_tr, outer_fold):
    result = []
    for j, inner_test in enumerate(inner_indices(len(x), y, outer_tr, outer_fold)):
        inner_tr = outer_tr[outer_tr != inner_test]
        result.append(dict(inner_fold=j, xtr=snv(x[inner_tr]),
            ytr=y[inner_tr], xte=snv(x[inner_test:inner_test+1]),
            y_true=int(y[inner_test]), synthetic=None,
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
                rec['selector_cache'][sel] = feature_indices(sel, rec['xtr'], rec['ytr'], rec['seed'])
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
    xtr = snv(raw[train]); xte = snv(raw[test:test+1]); ytr=y[train]
    if selector not in selector_cache:
        selector_cache[selector] = feature_indices(selector, xtr, ytr, seed)
    idx = selector_cache[selector]
    if aug == 'Real only':
        xx, yy = xtr, ytr
    else:
        if synthetic_cache['pair'] is None:
            synthetic_cache['pair'] = generate_latent_flow(xtr, ytr, seed, max(AUG_COUNTS))
        syn, labs = synthetic_cache['pair']
        count = int(aug.split()[2].split('/')[0]); m=max(AUG_COUNTS)
        use = np.r_[0:count, m:m+count]
        xx, yy = np.vstack([xtr,syn[use]]), np.r_[ytr,labs[use]]
    return train_and_predict(model, representation(xx, idx, model), yy,
                             representation(xte, idx, model), seed, encoder)


def latex_escape(s):
    return (str(s).replace('\\', r'\textbackslash{}').replace('&', r'\&')
                  .replace('%', r'\%').replace('_',r'\_').replace('#', r'\#'))


def write_latex_table(frame, columns, path, caption, label, numeric=None):
    numeric = numeric or {}
    n = len(columns)
    align = 'l' + 'r'*(n-1)
    lines = ['\\begin{table}[t]', '\\centering',
             '\\caption{' + caption + '}', '\\label{' + label + '}',
             '\\small', '\\begin{tabular}{' + align + '}', '\\toprule',
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
    lines += ['\\bottomrule','\\end{tabular}','\\end{table}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def save_stage_tables(candidate_scores, selections, preds):
    stage_labels={1:'Classifier',2:'Augmentation',3:'Feature selection'}
    table_outputs=[]
    for step in (1,2,3):
        data = candidate_scores[candidate_scores.step == step].copy()
        grouped = (data.groupby('candidate', sort=False)
                   .agg(ValBA=('BA','mean'), ValAUC=('AUC','mean'),
                        NOuter=('outer_fold','nunique')).reset_index())
        chosen = selections[selections.step==step].candidate.value_counts()
        grouped['Selected'] = grouped.candidate.map(chosen).fillna(0).astype(int)
        grouped = grouped.sort_values(['Selected','ValBA','ValAUC'], ascending=False)
        path = OUTPUT_DIR / f'0{step}_candidate_summary.csv'
        grouped.to_csv(path, index=False)
        latexpath=OUTPUT_DIR / f'0{step}_candidate_summary.tex'
        write_latex_table(grouped, [('candidate','Configuration'), ('ValBA','Inner BA'),
            ('ValAUC','Inner AUC'), ('Selected','Selected folds')], latexpath,
            caption=f'Step {step}: nested inner-validation comparison for {stage_labels[step].lower()}. '
                    'Inner CV statistics are averaged over outer folds.',
            label=f'tab:step{step}_candidates', numeric={'ValBA':'.3f','ValAUC':'.3f'})
        table_outputs.append(latexpath.name)
    summary=[]
    for step in (1,2,3):
        d=preds[preds.step==step].sort_values('outer_fold')
        summary.append(dict(Step=f'Step {step}', **evaluate_predictions(
            d.y_true, d.y_pred, d.score)))
    summary=pd.DataFrame(summary)
    summary.to_csv(OUTPUT_DIR/'nested_stage_metrics.csv', index=False)
    # A paper-ready table of unbiased outer patient predictions.
    write_latex_table(summary, [('Step','Stage'),('Accuracy','Acc.'),
        ('BalancedAccuracy','BA'),('Sensitivity','Sens.'),('Specificity','Spec.'),
        ('AUC','AUC'),('Correct','Correct')], OUTPUT_DIR/'nested_stage_metrics.tex',
        caption='Patient-level outer-fold results after sequential inner-fold selection. '
                + ('Quick-run diagnostic, not inferential.' if FAST else 'All original patients are held out once.'),
        label='tab:nested_stages', numeric={key:'.3f' for key in
            ('Accuracy','BalancedAccuracy','Sensitivity','Specificity','AUC')})
    table_outputs.append('nested_stage_metrics.tex')
    (OUTPUT_DIR/'paper_tables.tex').write_text('\n'.join('\\input{' + name + '}'
                                                for name in table_outputs)+'\n',encoding='utf-8')
    return summary, grouped


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
        ('HolmP','Holm p')],OUTPUT_DIR/'paired_stage_comparisons.tex',
        caption='Paired out-of-fold change in balanced accuracy for sequential stages.',
        label='tab:paired_stage_changes',
        numeric={s:'.3f' for s in ('DeltaBA_pp','CI_low_pp','CI_high_pp','HolmP')})
    return result


def plot_stage_results(scores, selections, metrics, pred):
    for step, title, filename in [
        (1, 'Classifier selection', '01_classifier_ablation.pdf'),
        (2, 'Augmentation selection', '02_augmentation_ablation.pdf'),
        (3, 'Feature-selection ablation','03_feature_selection_ablation.pdf')]:
        dd=scores[scores.step==step]
        g=(dd.groupby('candidate',sort=False).agg(BA=('BA','mean'),
            SD=('BA','std')).reset_index())
        c=selections[selections.step==step].candidate.value_counts()
        g['frequency']=g.candidate.map(c).fillna(0).astype(int)
        g=g.sort_values(['frequency','BA'],ascending=False)
        fig, (a,b)=plt.subplots(1,2,figsize=(11, max(3.5,len(g)*.55)),
                                gridspec_kw={'width_ratios':[1.3,1]})
        y=np.arange(len(g)); a.barh(y,100*g.BA,color='0.45',height=.65)
        a.set_yticks(y);a.set_yticklabels(g.candidate.tolist());a.invert_yaxis()
        a.set_xlabel('Mean inner-CV balanced accuracy (%)'); a.set_xlim(0,105)
        for i,v in enumerate(g.BA): a.text(min(101,100*v+1),i,f'{100*v:.1f}',va='center',fontsize=9)
        b.barh(y,g.frequency,color='0.25',height=.65)
        b.set_yticks(y);b.set_yticklabels([]);b.invert_yaxis()
        b.set_xlabel('Selected outer folds'); b.set_xlim(0,max(1,int(g.frequency.max()))+1)
        for i,v in enumerate(g.frequency): b.text(v+.1,i,str(v),va='center')
        fig.suptitle(f'Step {step}: {title}',fontsize=11)
        fig.tight_layout();fig.savefig(OUTPUT_DIR/filename,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4))
    x=np.arange(3); ba=100*metrics.BalancedAccuracy.to_numpy()
    ax.bar(x,ba,color=['0.75','0.5','0.25'],width=.65)
    ax.set_xticks(x);ax.set_xticklabels(['Step 1','Step 2','Step 3']);ax.set_ylim(0,107)
    ax.set_ylabel('Outer-test balanced accuracy (%)')
    for i,z in enumerate(ba):ax.text(i,z+1,f'{z:.1f}',ha='center')
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'04_nested_stage_performance.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    for step in (1,2,3):
        d=pred[pred.step==step].sort_values('outer_fold')
        fpr,tpr,_=roc_curve(d.y_true,d.score)
        auc=roc_auc_score(d.y_true,d.score)
        ax.plot(fpr,tpr,label=f'Step {step} (AUC {auc:.3f})')
    ax.plot([0,1],[0,1],'--',color='0.7',lw=1)
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate')
    ax.legend(loc='lower right',frameon=False)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'05_stage_roc.pdf',bbox_inches='tight');plt.close(fig)
    piv=pred.pivot(index='step',columns='outer_fold',values='y_pred').sort_index()
    true=pred[pred.step==1].set_index('outer_fold').y_true
    hits=piv.eq(true,axis=1).astype(int)
    fig,ax=plt.subplots(figsize=(max(7,0.5*len(true)),2.5))
    ax.imshow(hits,cmap='Greys',vmin=0,vmax=1,aspect='auto')
    ax.set_yticks(range(3));ax.set_yticklabels(['Step 1','Step 2','Step 3'])
    ax.set_xticks(range(len(hits.columns)));ax.set_xticklabels(hits.columns)
    ax.set_xlabel('Outer-held-out patient fold')
    for i in range(hits.shape[0]):
        for j in range(hits.shape[1]):
            ax.text(j,i,'✓' if hits.iloc[i,j] else '×',ha='center',va='center',
                    color='white' if hits.iloc[i,j] else 'black',fontsize=9)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'06_patient_errors.pdf',bbox_inches='tight');plt.close(fig)


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
    if not len(AUG_COUNTS)==2 or min(AUG_COUNTS)<=0 or AUG_COUNTS[0]>=AUG_COUNTS[1]:
        raise ValueError('AUG_COUNTS must contain two increasing positive integers')
    encoder=freeze_encoder()
    if FAST:
        rng=np.random.default_rng(SEED)
        outer_test=np.concatenate([rng.choice(np.flatnonzero(y==k),
                    size=min(OUTER_QUICK_PER_CLASS,int((y==k).sum())),replace=False)
                    for k in (0,1)])
    else:
        outer_test=np.arange(len(y))
    candidate_scores=[]; selected_rows=[]; predictions=[]; features=[]
    outer_test_set=set(outer_test.tolist())
    for outer_fold, (train,held) in enumerate(LeaveOneOut().split(raw),1):
        test=int(held[0])
        if test not in outer_test_set: continue
        inner=make_inner_records(raw,y,train,outer_fold)
        held_labels=[r['y_true'] for r in inner]
        if min(np.bincount(held_labels,minlength=2)) < 1:
            raise ValueError('Inner CV must contain both healthy and sick held-out patients')
        outer_seed=SEED+outer_fold*100003
        outer_flow={'pair':None};outer_selectors={}
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
    metrics,_=save_stage_tables(val,sel,pred)
    paired=paired_comparisons(pred)
    plot_stage_results(val,sel,metrics,pred)
    config=dict(mode=RUN_MODE,n_total_patients=len(y),n_tested_patients=len(outer_test),
        n_healthy=int((y==0).sum()),n_sick=int((y==1).sum()),channels=raw.shape[1],
        root=str(ROOT), healthy_directory=str(HEALTHY_DIR),sick_directory=str(SICK_DIR),
        classifiers=MODELS,augmentations=AUGMENTATIONS,selectors=SELECTORS,
        selection_criterion='inner LOOCV balanced_accuracy -> ROC_AUC -> accuracy',
        outer_protocol='patient-level LOOCV with nested sequential inner selection',
        results_status='SMOKE_TEST_NOT_FOR_PUBLICATION' if FAST else 'EXPLORATORY_NESTED_LOOCV',
        auc_note='Across fold-selected models, AUC uses training-std-standardized margins; not calibrated probabilities.',
        flow_steps=FLOW_STEPS,flow_ode_steps=FLOW_ODE_STEPS,
        augmentation_counts=AUG_COUNTS,seed=SEED,python=sys.version,
        aeon_version=getattr(sys.modules.get('aeon'),'__version__',None))
    (OUTPUT_DIR/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Finished {RUN_MODE}: {len(outer_test)} outer patients, outputs in {OUTPUT_DIR}')
    return pred,metrics,sel,val,paired


if __name__ == '__main__':
    run_experiment()
