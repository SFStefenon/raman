"""Three-stage nested spectral classification, augmentation and feature study.

Set RUN_MODE to 'quick', 'normal' or 'complete'.
Data: saudaveis/*.csv and doentes/*.csv, one original patient per CSV.
Two columns (wavenumber, signal), consistent wavelength grid.

Stage 1: classifier selection (real spectra; SNV; full channels).
Stage 2: augmentation selection (selected classifier; full channels).
Stage 3: feature selection/extraction (selected classifier and augmentation).
The entire selection is nested inside an outer original-patient LOOCV loop.
Synthetic data, feature statistics and scalers are never fitted using inner or
outer held-out patients. All reported *outer test* metrics use real patients.

quick: two outer patients, two inner validation patients and tiny grids; fast
       smoke test only, intentionally omits most candidates and PDF figures.
normal: 6 outer patients, 4 inner validation patients and a broad candidate set.
complete: all outer patients, all inner patients, ALL configured methods/counts,
          expanded diagnostics, saved quality comparisons, final seed robustness.

'complete' is COMPLETE SEQUENTIAL evaluation, not the Cartesian product of all
classifier x augmentation x feature combinations. RUN_FACTORIAL_AUDIT enables
an OPTIONAL expensive full cross-factor analysis on nested inner folds.

Python 3.9+ dependencies: numpy pandas scipy scikit-learn matplotlib torch.
Only a small clinical dataset is available; findings require independent
external validation. Timing is machine-specific and not formal complexity.
"""

from __future__ import annotations
from pathlib import Path
import json
import time
import resource
import random
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from scipy.signal import find_peaks, savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
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

# Modes specify BOTH experimental coverage and optimization budgets.
RUN_MODE = 'complete'        # 'quick', 'normal', or 'complete'
OVERRIDES = {}           # e.g. {'ROOT': '/path/to/paper 2', 'DEVICE':'cuda'}
ROOT = Path(OVERRIDES.get('ROOT', Path(__file__).resolve().parent))
HEALTHY_DIR = Path(OVERRIDES.get('HEALTHY_DIR', ROOT / 'saudaveis'))
SICK_DIR = Path(OVERRIDES.get('SICK_DIR', ROOT / 'doentes'))
PROFILES = {
 'quick': dict(outer_per_class=1, inner_per_class=1,
      aug_counts=(3,), feature_k=(12,20,32),
      flow_steps=5, diffusion_steps=5, gan_steps=5, vae_steps=5, wgan_steps=5,
      diffusion_t=6, flow_ode_steps=4, batch_size=12,
      stability_bootstraps=2, bootstrap=50, latent_dim=3,
      classifier_subset=('PLS-DA','Linear SVM'),
      augmentation_subset=('Spectral Mixup','PCA-Gaussian','Latent Flow'),
      feature_subset=('Full spectrum','PCA-4','ANOVA-12'),
      plots=False, quality=False, final_seed_repeats=1),
 'normal': dict(outer_per_class=3, inner_per_class=2,
      aug_counts=(10,25), feature_k=(32,128,256),
      flow_steps=100, diffusion_steps=120, gan_steps=140,
      vae_steps=120, wgan_steps=120, diffusion_t=30,
      flow_ode_steps=12, batch_size=32,
      stability_bootstraps=8, bootstrap=400, latent_dim=4,
      classifier_subset=None,
      augmentation_subset=None,
      feature_subset=None,
      plots=True, quality=True, final_seed_repeats=1),
 'complete': dict(outer_per_class=None, inner_per_class=None,
      aug_counts=(10,25,50), feature_k=(64,128,256),
      flow_steps=500, diffusion_steps=650, gan_steps=700,
      vae_steps=500, wgan_steps=700, diffusion_t=100,
      flow_ode_steps=32, batch_size=64,
      stability_bootstraps=32, bootstrap=3000, latent_dim=6,
      classifier_subset=None, augmentation_subset=None, feature_subset=None,
      plots=True, quality=True, final_seed_repeats=3)
}
if RUN_MODE not in PROFILES:
    raise ValueError('RUN_MODE must be quick, normal or complete')
P = PROFILES[RUN_MODE]
OUTPUT_DIR = Path(OVERRIDES.get('OUTPUT_DIR', ROOT / ('three_stage_pipeline_' + RUN_MODE)))
SEED = int(OVERRIDES.get('SEED', 1))
DEVICE = torch.device(OVERRIDES.get('DEVICE', 'cpu'))
FAST = RUN_MODE == 'quick'
SUBSET = RUN_MODE != 'complete'
OUTER_PER_CLASS = int(P['outer_per_class']) if P['outer_per_class'] else None
INNER_PER_CLASS = int(P['inner_per_class']) if P['inner_per_class'] else None
FLOW_STEPS = int(OVERRIDES.get('FLOW_STEPS', P['flow_steps']))
DIFFUSION_STEPS = int(OVERRIDES.get('DIFFUSION_STEPS', P['diffusion_steps']))
GAN_STEPS = int(OVERRIDES.get('GAN_STEPS', P['gan_steps']))
VAE_STEPS = int(OVERRIDES.get('VAE_STEPS', P['vae_steps']))
WGAN_STEPS = int(OVERRIDES.get('WGAN_STEPS', P['wgan_steps']))
DIFFUSION_T = int(OVERRIDES.get('DIFFUSION_T', P['diffusion_t']))
FLOW_BATCH_SIZE = int(OVERRIDES.get('BATCH_SIZE', P['batch_size']))
FLOW_ODE_STEPS = int(OVERRIDES.get('FLOW_ODE_STEPS', P['flow_ode_steps']))
LATENT_DIM = int(OVERRIDES.get('LATENT_DIM', P['latent_dim']))
AUG_COUNTS = tuple(OVERRIDES.get('AUG_COUNTS', P['aug_counts']))
FEATURE_K = tuple(OVERRIDES.get('FEATURE_K', P['feature_k']))
STABILITY_BOOTSTRAPS = int(OVERRIDES.get('STABILITY_BOOTSTRAPS', P['stability_bootstraps']))
MRMR_BINS = 3
STABILITY_ALPHA = 0.015
N_BOOTSTRAP = int(OVERRIDES.get('N_BOOTSTRAP', P['bootstrap']))
RUN_PLOTS = bool(OVERRIDES.get('RUN_PLOTS', P['plots']))
RUN_QUALITY = bool(OVERRIDES.get('RUN_QUALITY', P['quality']))
FINAL_SEED_REPEATS = int(OVERRIDES.get('FINAL_SEED_REPEATS', P['final_seed_repeats']))
# A completely crossed factorial audit is far more expensive than sequential CV.
# Enable only if the 3-way interaction analysis is specifically needed.
RUN_FACTORIAL_AUDIT = bool(OVERRIDES.get('RUN_FACTORIAL_AUDIT', False))
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

def sync_device():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE)


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


class ConditionalDenoiser(nn.Module):
    """Class-conditional epsilon network for a DDPM latent diffusion generator."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 7, 96), nn.SiLU(),
                                 nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, d))

    def forward(self, z, labels, timesteps):
        t = timesteps.reshape(-1, 1).float()
        emb = [t, torch.sin(2 * np.pi * t), torch.cos(2 * np.pi * t),
               torch.sin(4 * np.pi * t), torch.cos(4 * np.pi * t)]
        return self.net(torch.cat([z, F.one_hot(labels.long(), 2).float()] + emb, dim=1))


class ConditionalGANGenerator(nn.Module):
    def __init__(self, d, noise_dim=16):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(nn.Linear(noise_dim + 2, 96), nn.LeakyReLU(0.2),
                                 nn.Linear(96, 96), nn.LeakyReLU(0.2), nn.Linear(96, d))

    def forward(self, noise, labels):
        return self.net(torch.cat([noise, F.one_hot(labels.long(), 2).float()], dim=1))


class ConditionalGANDiscriminator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 2, 96), nn.LeakyReLU(0.2),
                                 nn.Linear(96, 48), nn.LeakyReLU(0.2), nn.Linear(48, 1))

    def forward(self, z, labels):
        return self.net(torch.cat([z, F.one_hot(labels.long(), 2).float()], dim=1))


class ConditionalVAE(nn.Module):
    """Conditional variational autoencoder with a class-conditional prior."""
    def __init__(self, d, hidden=64):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(nn.Linear(d+2, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.mean = nn.Linear(hidden, d)
        self.logvar = nn.Linear(hidden, d)
        self.decoder = nn.Sequential(nn.Linear(d+2, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden), nn.SiLU(),
                                    nn.Linear(hidden, d))

    def forward(self, x, y):
        h = self.encoder(torch.cat([x, F.one_hot(y.long(),2).float()], 1))
        mu, logv = self.mean(h), self.logvar(h).clamp(-10, 10)
        z = mu + torch.exp(.5*logv)*torch.randn_like(mu)
        reconstruct = self.decoder(torch.cat([z,F.one_hot(y.long(),2).float()],1))
        return reconstruct, mu, logv

    def generate(self, z, y):
        return self.decoder(torch.cat([z,F.one_hot(y.long(),2).float()],1))


def conditional_gradient_penalty(disc, real, fake, labels):
    mix = torch.rand((len(real),1),device=real.device)
    interpolated = (mix*real + (1-mix)*fake).requires_grad_(True)
    result = disc(interpolated,labels)
    grad = torch.autograd.grad(result,interpolated,
        grad_outputs=torch.ones_like(result),create_graph=True)[0]
    return ((grad.reshape(len(real),-1).norm(2,dim=1)-1)**2).mean()


class ClassConditionalLatentGenerator:
    """Train-fold-only PCA latent space; compare Flow, cDDPM, and cGAN fairly.

    All three models train with identical latent dimension, real source patients,
    class labels, batch size, and CPU/GPU device. Training never includes held-out
    original patients. This evaluates *latent* generative methods, not full-spectrum
    DDPM/GAN architectures. Sampled spectra are projected back to spectral space.
    """
    def __init__(self, method, seed):
        self.method = method
        self.seed = int(seed)
        self.params = 0

    def fit(self, Xreal, yreal):
        seed_everything(self.seed)
        real = snv(Xreal)
        d = max(1, min(LATENT_DIM, len(real) - 2, real.shape[1]))
        self.pca = PCA(n_components=d, random_state=self.seed)
        self.scaler = StandardScaler()
        data = torch.as_tensor(self.scaler.fit_transform(
            self.pca.fit_transform(real)), dtype=torch.float32, device=DEVICE)
        labels = torch.as_tensor(yreal, dtype=torch.long, device=DEVICE)
        pools = [torch.where(labels == c)[0] for c in (0, 1)]
        if any(len(z) < 2 for z in pools):
            raise ValueError('Generative models require at least 2 ORIGINAL training patients/class.')
        self.d = d
        def minibatch():
            y = torch.randint(0, 2, (FLOW_BATCH_SIZE,), device=DEVICE)
            ix = torch.empty_like(y)
            for c in (0, 1):
                mask = y == c
                ix[mask] = pools[c][torch.randint(len(pools[c]), (int(mask.sum()),), device=DEVICE)]
            return data[ix], y

        if self.method == 'Latent Flow':
            self.net = FlowVelocity(d).to(DEVICE)
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
            self.net.train()
            for _ in range(FLOW_STEPS):
                z1, y = minibatch()
                z0 = torch.randn_like(z1)
                t = torch.rand((len(y), 1), device=DEVICE)
                zt = (1 - t) * z0 + t * z1
                loss = F.mse_loss(self.net(zt, y, t), z1 - z0)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            self.params = sum(p.numel() for p in self.net.parameters())

        elif self.method == 'Conditional Diffusion':
            self.net = ConditionalDenoiser(d).to(DEVICE)
            # High enough terminal noise that the sampling prior is near Gaussian.
            end_beta = 0.25 if FAST else 0.12
            self.beta = torch.linspace(1e-4, end_beta, DIFFUSION_T, device=DEVICE)
            self.alpha = 1 - self.beta
            self.ab = torch.cumprod(self.alpha, dim=0)
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
            self.net.train()
            for _ in range(DIFFUSION_STEPS):
                z, y = minibatch()
                k = torch.randint(DIFFUSION_T, (len(y),), device=DEVICE)
                noise = torch.randn_like(z)
                zt = self.ab[k, None].sqrt() * z + (1 - self.ab[k, None]).sqrt() * noise
                eps = self.net(zt, y, k.float() / max(1, DIFFUSION_T - 1))
                loss = F.mse_loss(eps, noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            self.params = sum(p.numel() for p in self.net.parameters())

        elif self.method == 'Conditional GAN':
            self.net = ConditionalGANGenerator(d).to(DEVICE)
            self.disc = ConditionalGANDiscriminator(d).to(DEVICE)
            go = torch.optim.Adam(self.net.parameters(), lr=2e-4, betas=(0.5, 0.999))
            do = torch.optim.Adam(self.disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
            self.net.train(); self.disc.train()
            for _ in range(GAN_STEPS):
                zr, y = minibatch()
                noise = torch.randn((len(y), self.net.noise_dim), device=DEVICE)
                zf = self.net(noise, y)
                dr, df = self.disc(zr, y), self.disc(zf.detach(), y)
                dloss = (F.binary_cross_entropy_with_logits(dr, torch.full_like(dr, 0.9)) +
                         F.binary_cross_entropy_with_logits(df, torch.zeros_like(df)))
                do.zero_grad(set_to_none=True); dloss.backward(); do.step()
                zf = self.net(torch.randn_like(noise), y)
                gpred = self.disc(zf, y)
                gloss = F.binary_cross_entropy_with_logits(gpred, torch.ones_like(gpred))
                go.zero_grad(set_to_none=True); gloss.backward(); go.step()
            self.params = sum(p.numel() for p in self.net.parameters()) + sum(
                p.numel() for p in self.disc.parameters())
        elif self.method == 'Conditional VAE':
            self.net = ConditionalVAE(d).to(DEVICE)
            opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3)
            self.net.train()
            for _ in range(VAE_STEPS):
                zr, y = minibatch()
                xr, mu, logvar = self.net(zr,y)
                reconstruction = F.mse_loss(xr,zr)
                kl = -.5*(1 + logvar - mu.square() - logvar.exp()).sum(1).mean()/d
                loss = reconstruction + .03*kl
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            self.params = sum(p.numel() for p in self.net.parameters())

        elif self.method == 'Conditional WGAN-GP':
            self.net = ConditionalGANGenerator(d).to(DEVICE)
            self.disc = ConditionalGANDiscriminator(d).to(DEVICE)
            gopt = torch.optim.Adam(self.net.parameters(),lr=1e-4,betas=(0.,.9))
            dopt = torch.optim.Adam(self.disc.parameters(),lr=1e-4,betas=(0.,.9))
            self.net.train(); self.disc.train()
            for _ in range(WGAN_STEPS):
                for _critic in range(2):
                    zr, y = minibatch()
                    zf = self.net(torch.randn((len(y),self.net.noise_dim),device=DEVICE),y)
                    penalty = conditional_gradient_penalty(self.disc,zr,zf.detach(),y)
                    dloss = self.disc(zf.detach(),y).mean()-self.disc(zr,y).mean()+10*penalty
                    dopt.zero_grad(set_to_none=True);dloss.backward();dopt.step()
                zg = self.net(torch.randn((len(y),self.net.noise_dim),device=DEVICE),y)
                gloss = -self.disc(zg,y).mean()
                gopt.zero_grad(set_to_none=True);gloss.backward();gopt.step()
            self.params = sum(p.numel() for p in self.net.parameters()) + sum(
                p.numel() for p in self.disc.parameters())
        else:
            raise KeyError(self.method)
        self.net.eval()
        return self

    def sample(self, per_class):
        seed_everything(self.seed + 1049 + 17 * int(per_class))
        n = int(per_class)
        labels = np.repeat([0, 1], n).astype(int)
        y = torch.as_tensor(labels, dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            if self.method == 'Latent Flow':
                z = torch.randn((len(y), self.d), device=DEVICE)
                dt = 1. / FLOW_ODE_STEPS
                for k in range(FLOW_ODE_STEPS):
                    t0 = torch.full((len(y), 1), k * dt, device=DEVICE)
                    t1 = torch.full((len(y), 1), (k + 1) * dt, device=DEVICE)
                    v0 = self.net(z, y, t0)
                    v1 = self.net(z + dt * v0, y, t1)
                    z += 0.5 * dt * (v0 + v1)
            elif self.method == 'Conditional Diffusion':
                z = torch.randn((len(y), self.d), device=DEVICE)
                for k in range(DIFFUSION_T - 1, -1, -1):
                    ts = torch.full((len(y),), k / max(1, DIFFUSION_T - 1), device=DEVICE)
                    eps = self.net(z, y, ts)
                    mean = (z - self.beta[k] * eps / (1 - self.ab[k]).sqrt()) / self.alpha[k].sqrt()
                    if k:
                        prev_ab = self.ab[k - 1]
                        posterior_var = self.beta[k] * (1 - prev_ab) / (1 - self.ab[k])
                        z = mean + posterior_var.clamp_min(0).sqrt() * torch.randn_like(z)
                    else:
                        z = mean
            elif self.method == 'Conditional VAE':
                z = self.net.generate(torch.randn((len(y),self.d),device=DEVICE),y)
            else:
                noise = torch.randn((len(y), self.net.noise_dim), device=DEVICE)
                z = self.net(noise, y)
            lat = z.cpu().numpy().clip(-7, 7)
            xs = self.pca.inverse_transform(self.scaler.inverse_transform(lat))
            xs = snv(xs)
        if not np.isfinite(xs).all():
            raise FloatingPointError(self.method + ' produced invalid spectral values')
        return xs, labels


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




def normalize_margin(test_margin, train_margin):
    """Scale raw margins by training dispersion, preserving zero threshold.

    This helps compare ROC scores across different classifiers/folds; it is
    NOT probability calibration. ROC-AUC remains descriptive for heterogeneous
    models fitted independently inside outer folds.
    """
    scale = max(float(np.std(np.asarray(train_margin, float))), 1e-6)
    return np.asarray(test_margin, float).reshape(-1) / scale




def predict_model(name, X_train, y_train, X_test, seed, encoder=None):
    """Return calibrated-to-training-dispersion margins and operation timing."""
    fit_start = time.perf_counter()
    if name == 'PLS-DA':
        model = PLSDA().fit(X_train, y_train)
    elif name == 'Elastic-net LR':
        model = make_elasticnet(int(seed))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            model.fit(X_train, y_train)
    elif name == 'Linear SVM':
        model = make_pipeline(StandardScaler(), LinearSVC(
            class_weight='balanced', C=0.1, max_iter=6000, random_state=int(seed)))
        model.fit(X_train, y_train)
    elif name == 'RBF SVM':
        model = make_pipeline(StandardScaler(), SVC(
            C=1.0, kernel='rbf', class_weight='balanced', gamma='scale'))
        model.fit(X_train, y_train)
    elif name == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, max_features='sqrt',
            class_weight='balanced', n_jobs=1, random_state=int(seed))
        model.fit(X_train, y_train)
    elif name == 'Extra Trees':
        model = ExtraTreesClassifier(n_estimators=100, max_features='sqrt',
            class_weight='balanced', n_jobs=1, random_state=int(seed))
        model.fit(X_train, y_train)
    elif name == 'KNN':
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=min(3,len(y_train)), weights='distance'))
        model.fit(X_train, y_train)
    else:
        raise ValueError('Unexpected classifier method: ' + str(name))
    fit_seconds = time.perf_counter() - fit_start
    predict_start = time.perf_counter()
    if hasattr(model,'decision_function'):
        test_score = model.decision_function(X_test)
    else:
        test_score = 2 * model.predict_proba(X_test)[:,1] - 1
    test_predict_seconds = time.perf_counter() - predict_start
    if hasattr(model,'decision_function'):
        train_score = model.decision_function(X_train)
    else:
        train_score = 2 * model.predict_proba(X_train)[:,1] - 1
    return (normalize_margin(test_score,train_score),
            fit_seconds, test_predict_seconds)


def sample_statistical_augmentation(xtr, ytr, method, count, seed):
    """Class-respecting spectral augmentation with *training-fold* parameters.

    The transformed spectra are re-normalized by SNV before training. Spectral
    perturbation strengths are deliberately conservative and tunable in code.
    They are not guaranteed label-invariant for every spectroscopy modality.
    """
    rng = np.random.default_rng(int(seed))
    real = snv(xtr)
    labels = np.repeat([0, 1], int(count)).astype(int)
    n, p = real.shape
    grid = np.arange(p)
    residual = real - savgol_filter(real,window_length=min(21,p if p%2 else p-1),
                                    polyorder=2,axis=1)
    noise_scale = max(float(np.median(residual.std(axis=1))), .005)
    samples=[]
    for c in labels:
        pool = real[ytr==c]
        a = pool[int(rng.integers(len(pool)))].copy()
        b = pool[int(rng.integers(len(pool)))].copy()
        if method == 'Spectral Mixup':
            lam = rng.beta(.4,.4)
            x = lam*a + (1-lam)*b
        elif method == 'Spectral SMOTE':
            lam = rng.uniform(.1,.9)
            x = a + lam*(b-a)
        elif method == 'Noise Injection':
            x = a + rng.normal(0,.35*noise_scale,size=p)
        elif method == 'Spectral Shift':
            shift = rng.uniform(-3,3)
            x = np.interp(np.clip(grid-shift,0,p-1),grid,a)
        elif method == 'Baseline Perturbation':
            t = np.linspace(-1,1,p)
            x = a + rng.uniform(-.08,.08)*t + rng.uniform(-.04,.04)*(t*t-.3)
        elif method == 'Smooth Warping':
            knots = np.linspace(0,p-1,9)
            displacement = np.interp(grid, knots, rng.normal(0,2.0,len(knots)))
            warped = np.maximum.accumulate(np.clip(grid+displacement,0,p-1))
            x = np.interp(warped,grid,a)
        elif method == 'Fourier Perturbation':
            z = np.fft.rfft(a)
            freq = np.fft.rfftfreq(p)
            amplitude = 1 + rng.normal(0,.03,len(z))*np.exp(-freq*8)
            x = np.fft.irfft(z*amplitude,n=p)
        elif method == 'Residual Bootstrap':
            base = savgol_filter(a,min(21,p if p%2 else p-1),2)
            r = residual[rng.integers(len(residual))]
            x = base + rng.uniform(.65,1.15)*r
        elif method == 'PCA-Gaussian':
            # Classwise shrinkage covariance in a globally trained PCA space.
            x = None
        elif method == 'Gaussian Mixture':
            x = None
        else:
            raise ValueError('Unsupported statistical augmentation: '+method)
        if x is not None:
            samples.append(x)
    if method in ('PCA-Gaussian','Gaussian Mixture'):
        from sklearn.covariance import LedoitWolf
        from sklearn.mixture import GaussianMixture
        d = max(1,min(LATENT_DIM,n-2,p))
        pca = PCA(n_components=d,random_state=seed)
        latent = pca.fit_transform(real)
        sc = StandardScaler().fit(latent)
        z = sc.transform(latent)
        generated = np.empty((len(labels),d),float)
        for c in (0,1):
            xc=z[ytr==c]
            idx=np.flatnonzero(labels==c)
            if method=='PCA-Gaussian':
                cov=LedoitWolf().fit(xc).covariance_+1e-4*np.eye(d)
                generated[idx]=rng.multivariate_normal(xc.mean(0),cov,len(idx))
            else:
                components=min(2,max(1,len(xc)//3))
                try:
                    gm=GaussianMixture(n_components=components,reg_covar=.03,
                                       random_state=seed,max_iter=150).fit(xc)
                    # Explicit per-class sampling using configured reproducible RNG.
                    weights=gm.weights_/gm.weights_.sum()
                    comp=rng.choice(components,size=len(idx),p=weights)
                    for j,k in enumerate(comp):
                        generated[idx[j]]=rng.multivariate_normal(gm.means_[k],gm.covariances_[k])
                except (ValueError,np.linalg.LinAlgError):
                    cov=LedoitWolf().fit(xc).covariance_+1e-3*np.eye(d)
                    generated[idx]=rng.multivariate_normal(xc.mean(0),cov,len(idx))
        samples=pca.inverse_transform(sc.inverse_transform(generated))
    else:
        samples=np.asarray(samples)
    result=snv(samples)
    if result.shape!=(len(labels),p) or not np.isfinite(result).all():
        raise FloatingPointError('Invalid synthetic spectra for '+method)
    return result,labels


def synthetic_quality(xreal,yreal,xsyn,ysyn):
    """Training-only, descriptive classwise fidelity / diversity diagnostics."""
    vals=[]
    for c in (0,1):
        r=snv(xreal[yreal==c]);s=snv(xsyn[ysyn==c])
        if len(s)==0:
            continue
        mean_rmse=float(np.sqrt(np.mean((r.mean(0)-s.mean(0))**2)))
        std_rmse=float(np.sqrt(np.mean((r.std(0)-s.std(0))**2)))
        div=float(np.mean(s.std(0))/max(1e-8,np.mean(r.std(0))))
        # Squared Euclidean computations avoid broadcasting [synthetic,real,p].
        sq=np.maximum(0,(s*s).sum(1)[:,None]+(r*r).sum(1)[None,:]-2*s@r.T)
        near=float(np.sqrt(sq.min(1)/s.shape[1]).mean())
        # Original-versus-original nearest-neighbor scale (excluding self).
        rr=np.maximum(0,(r*r).sum(1)[:,None]+(r*r).sum(1)[None,:]-2*r@r.T)
        np.fill_diagonal(rr,np.inf)
        realnear=float(np.sqrt(rr.min(1)/r.shape[1]).mean())
        vals.append((mean_rmse,std_rmse,div,near/max(realnear,1e-8)))
    if not vals:
        return dict(mean_rmse=np.nan,std_rmse=np.nan,diversity_ratio=np.nan,near_ratio=np.nan)
    return dict(zip(('mean_rmse','std_rmse','diversity_ratio','near_ratio'),
                    map(float,np.mean(vals,axis=0))))


CLASSIFIER_ALL = ['PLS-DA','Elastic-net LR','Linear SVM','RBF SVM',
                  'Random Forest','Extra Trees','KNN']
NEURAL_AUGMENTATIONS = ['Latent Flow','Conditional Diffusion','Conditional GAN',
                        'Conditional VAE','Conditional WGAN-GP']
STATISTICAL_AUGMENTATIONS = ['Spectral Mixup','Spectral SMOTE','Noise Injection',
  'Spectral Shift','Baseline Perturbation','Smooth Warping',
  'Fourier Perturbation','Residual Bootstrap','PCA-Gaussian','Gaussian Mixture']
AUGMENT_METHODS_ALL = STATISTICAL_AUGMENTATIONS + NEURAL_AUGMENTATIONS

small_k, mid_k, large_k = FEATURE_K
SELECTORS_ALL = ['Full spectrum','PCA-4','PCA-8','PLS-2',
  'SG derivative + PCA-8',f'ANOVA-{small_k}',f'ANOVA-{mid_k}',
  f'ANOVA-{large_k}',f'mRMR-{small_k}',f'mRMR-{large_k}',
  f'Stability-{small_k}',f'Stability-{mid_k}',f'PCA peaks-{mid_k}',
  f'PCA peaks-{large_k}']
MODELS = list(P['classifier_subset'] or CLASSIFIER_ALL)
SELECTORS = list(P['feature_subset'] or SELECTORS_ALL)
AUGMENTATION_METHODS = list(P['augmentation_subset'] or AUGMENT_METHODS_ALL)
AUGMENTATIONS = ['Real only'] + [f'{method} {n}/class'
      for method in AUGMENTATION_METHODS for n in AUG_COUNTS]

class FeatureProjection:
    """Fold-safe feature extraction fit exclusively on ORIGINAL training patients."""
    def __init__(self, name, Xreal, yreal, seed):
        self.name = name
        self.derivative = name.startswith('SG derivative')
        raw = self._process(Xreal)
        k = min(int(name.split('-')[-1]), len(raw) - 2, raw.shape[1])
        if name.startswith('PLS'):
            self.mapper = PLSRegression(n_components=k, scale=True).fit(raw, yreal)
        else:
            self.mapper = PCA(n_components=k, random_state=seed).fit(raw)

    def _process(self, x):
        x = np.asarray(x, float)
        if not self.derivative:
            return x
        window = min(11, x.shape[1] if x.shape[1] % 2 else x.shape[1] - 1)
        return savgol_filter(x, window_length=max(5, window), polyorder=2,
                             deriv=1, axis=1)

    def transform(self, x):
        return np.asarray(self.mapper.transform(self._process(x)), dtype=float)


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
    if selector.startswith(('PCA-', 'PLS-', 'SG derivative + PCA-')):
        return FeatureProjection(selector, x_real, y_real, seed)
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
    """Fitted real-only projection or direct selected wavelength columns."""
    if indices is None:
        return x
    if isinstance(indices, FeatureProjection):
        return indices.transform(x)
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




def train_and_predict(model, xtr, ytr, xte, seed, encoder=None):
    seed_everything(seed)
    margin, fit_s, test_s = predict_model(model, xtr, ytr, xte, seed, encoder)
    return float(margin[0]), fit_s, test_s


def inner_indices(n, y_all, original_outer_tr, outer_fold):
    if INNER_PER_CLASS is None:
        return list(original_outer_tr)
    rng = np.random.default_rng(SEED + outer_fold*137)
    chosen=[]
    for cls in (0,1):
        possibilities = original_outer_tr[y_all[original_outer_tr]==cls]
        chosen.extend(rng.choice(possibilities,
            size=min(INNER_PER_CLASS,len(possibilities)),replace=False).tolist())
    return sorted(chosen)

def parse_augmentation(aug):
    if aug == 'Real only':
        return 'Real only', 0
    method, count = aug.rsplit(' ', 1)
    return method, int(count.split('/')[0])




def aug_dataset(record, aug):
    """Cache generator fits and synthetic samples within a training fold only.

    Standalone estimated cold-start times charge generator training for every
    candidate separately. The actual search shares fits, to reduce compute.
    """
    meta=dict(generator_fit_s=0.,sampling_s=0.,generator_params=0,
              mean_rmse=np.nan,std_rmse=np.nan,
              diversity_ratio=np.nan,near_ratio=np.nan)
    if aug == 'Real only':
        record['last_aug_meta']=meta
        return record['xtr'], record['ytr']
    method,count = parse_augmentation(aug)
    if method in NEURAL_AUGMENTATIONS:
        if method not in record['generator_models']:
            sync_device();start=time.perf_counter()
            gen=ClassConditionalLatentGenerator(method,record['seed']).fit(
                record['xtr'],record['ytr'])
            sync_device()
            record['generator_models'][method]=gen
            record['generator_fit_times'][method]=time.perf_counter()-start
        fit_time=record['generator_fit_times'][method]
        params=record['generator_models'][method].params
    else:
        fit_time=0.;params=0
    key=(method,count)
    if key not in record['synthetic_cache']:
        sync_device();start=time.perf_counter()
        if method in NEURAL_AUGMENTATIONS:
            record['synthetic_cache'][key]=record['generator_models'][method].sample(count)
        else:
            record['synthetic_cache'][key]=sample_statistical_augmentation(
                record['xtr'],record['ytr'],method,count,record['seed']+337)
        sync_device()
        record['sampling_times'][key]=time.perf_counter()-start
    xs,ys=record['synthetic_cache'][key]
    meta.update(generator_fit_s=fit_time,sampling_s=record['sampling_times'][key],
                generator_params=params)
    if RUN_QUALITY:
        if key not in record['quality_cache']:
            record['quality_cache'][key]=synthetic_quality(record['xtr'],record['ytr'],xs,ys)
        meta.update(record['quality_cache'][key])
    record['last_aug_meta']=meta
    return np.vstack([record['xtr'],xs]),np.r_[record['ytr'],ys]

def make_inner_records(x, y, outer_tr, outer_fold):
    result = []
    for j, inner_test in enumerate(inner_indices(len(x), y, outer_tr, outer_fold)):
        inner_tr = outer_tr[outer_tr != inner_test]
        result.append(dict(inner_fold=j, xtr=snv(x[inner_tr]),
            ytr=y[inner_tr], xte=snv(x[inner_test:inner_test+1]),
            y_true=int(y[inner_test]), synthetic_cache={},
            generator_models={}, generator_fit_times={}, sampling_times={},
            quality_cache={}, feature_fit_times={},
            seed=int(SEED + (outer_fold + 1) * 100003 + (inner_test + 1) * 101),
            selector_cache={}))
    return result


def evaluate_candidates(records, candidates, model, aug, selector, step, encoder, candidates_axis):
    rows = []
    for rec in records:
        if selector is not None and selector not in rec['selector_cache']:
            start = time.perf_counter()
            rec['selector_cache'][selector] = feature_indices(
                selector, rec['xtr'], rec['ytr'], rec['seed'])
            rec['feature_fit_times'][selector] = time.perf_counter() - start
        for name in candidates:
            mod = name if candidates_axis == 'model' else model
            ag = name if candidates_axis == 'augmentation' else aug
            sel = name if candidates_axis == 'selector' else selector
            if sel not in rec['selector_cache']:
                start = time.perf_counter()
                rec['selector_cache'][sel] = feature_indices(sel, rec['xtr'],
                                                              rec['ytr'], rec['seed'])
                rec['feature_fit_times'][sel] = time.perf_counter() - start


            idx = rec['selector_cache'][sel]
            aa, yy = aug_dataset(rec, ag)
            meta = rec['last_aug_meta'].copy()
            start = time.perf_counter()
            xt = representation(rec['xte'], idx, mod)
            xx = representation(aa, idx, mod)
            transform_s = time.perf_counter() - start
            start = time.perf_counter()
            score, clf_fit_s, test_s = train_and_predict(
                mod, xx, yy, xt, rec['seed'], encoder)
            classifier_s = time.perf_counter() - start
            feature_s = rec['feature_fit_times'].get(sel, 0.) + transform_s
            cold_s = (meta['generator_fit_s'] + meta['sampling_s'] +
                      feature_s + classifier_s)
            rows.append(dict(step=step, candidate=name, inner_fold=rec['inner_fold'],
                y_true=rec['y_true'], score=score, y_pred=int(score >= 0),
                classifier_s=classifier_s, classifier_fit_s=clf_fit_s,
                test_predict_s=test_s, feature_s=feature_s,
                generator_fit_s=meta['generator_fit_s'], sampling_s=meta['sampling_s'],
                generator_params=meta['generator_params'], feature_dim=xx.shape[1],
                 mean_rmse=meta['mean_rmse'],std_rmse=meta['std_rmse'],
                 diversity_ratio=meta['diversity_ratio'],near_ratio=meta['near_ratio'],
                 cold_total_s=cold_s, train_matrix_mb=xx.nbytes / (1024 ** 2)))
    return pd.DataFrame(rows)


def outer_predict(raw, y, train, test, model, aug, selector, seed, encoder,
                  outer_record):
    rec = outer_record
    if selector not in rec['selector_cache']:
        start = time.perf_counter()
        rec['selector_cache'][selector] = feature_indices(
            selector, rec['xtr'], rec['ytr'], seed)
        rec['feature_fit_times'][selector] = time.perf_counter() - start
    idx = rec['selector_cache'][selector]
    aa, yy = aug_dataset(rec, aug)
    meta = rec['last_aug_meta'].copy()
    start = time.perf_counter()
    xx = representation(aa, idx, model)
    xt = representation(rec['xte'], idx, model)
    feature_s = rec['feature_fit_times'].get(selector, 0.) + time.perf_counter() - start
    start = time.perf_counter()
    score, clf_fit_s, test_s = train_and_predict(
        model, xx, yy, xt, seed, encoder)
    classifier_s = time.perf_counter() - start
    resource = dict(generator_fit_s=meta['generator_fit_s'],
        sampling_s=meta['sampling_s'], feature_s=feature_s,
        classifier_s=classifier_s, classifier_fit_s=clf_fit_s,
        test_predict_s=test_s, generator_params=meta['generator_params'],
        mean_rmse=meta['mean_rmse'],std_rmse=meta['std_rmse'],
        diversity_ratio=meta['diversity_ratio'],near_ratio=meta['near_ratio'],
        feature_dim=xx.shape[1], train_matrix_mb=xx.nbytes/(1024**2),
        cold_total_s=meta['generator_fit_s'] + meta['sampling_s'] + feature_s + classifier_s)
    return score, resource


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
    stage_labels = {1:'classifier', 2:'augmentation', 3:'feature engineering'}
    table_outputs = []
    refs = {1:'fold classifier average', 2:'Real only', 3:'Full spectrum'}
    for step in (1,2,3):
        data = candidate_scores[candidate_scores.step == step].copy()
        grouped = data.groupby('candidate', sort=False).agg(
            ValBA=('BA','mean'), SD_BA=('BA','std'),
            ValAUC=('AUC','mean'), ValAcc=('ACC','mean'),
            MeanTime_s=('cold_total_s','mean'),
             GenFit_s=('generator_fit_s','mean'),
             Sample_s=('sampling_s','mean'),
             Feature_s=('feature_s','mean'),
             Classifier_s=('classifier_s','mean'),
             TestPredict_s=('test_predict_s','mean'),
             FeatureDim=('feature_dim','mean'),
             MatrixMB=('train_matrix_mb','mean'),
             GenParams=('generator_params','mean'),
             MeanRMSE=('mean_rmse','mean'),StdRMSE=('std_rmse','mean'),
              Diversity=('diversity_ratio','mean'),NearRatio=('near_ratio','mean'),
              NOuter=('outer_fold','nunique')).reset_index()
        if step == 1:
            baseline = data.groupby('outer_fold',as_index=False)['BA'].mean().rename(
                columns={'BA':'ReferenceBA'})
        else:
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
             ('ValAUC','Inner AUC'),('MeanTime_s','Cold time (s)'),
              ('Selected','Selections')],tex,
            caption=f'Step {step} ({stage_labels[step]}): nested inner-validation '
                    f'candidate comparison; mean paired differences versus {refs[step]}. '
                    'Selections count the outer folds choosing each candidate. '
                    'Inner validation means are descriptive, not outer-test estimates.',
            label=f'tab:step{step}_candidates',numeric={
                'ValBA':'.3f','SD_BA':'.3f','MeanDeltaBA_pp':'+.1f','ValAUC':'.3f','MeanTime_s':'.3f'})
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
             + ('Subset-only diagnostic; not whole-cohort estimate.' if SUBSET else
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
                           ExactMcNemarP=np.nan if SUBSET else exact))
    result=pd.DataFrame(result)
    if not SUBSET:
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



def tidy(ax):
    ax.spines['left'].set_color(PALETTE['light'])
    ax.spines['bottom'].set_color(PALETTE['light'])
    ax.tick_params(length=0, pad=5)
    ax.set_facecolor('white')




def plot_stage_results(scores, selections, metrics, pred):
    ref={1:'fold classifier average',2:'Real only',3:'Full spectrum'}
    labels={1:'Classifier',2:'Augmentation',3:'Feature engineering'}
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
        if stage == 1:
            delta=mat.sub(mat.mean(axis=0),axis=1)*100
        else:
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
        a.text(0,1.035,'A)  Candidate accuracy',transform=a.transAxes,
               fontsize=10,color=PALETTE['ink'])
        for i,row in summary.iterrows():
            b.plot([0,row.freq],[i,i],color=PALETTE['light'],lw=3,zorder=1)
            b.scatter(row.freq,i,s=54,color=colors[stage],zorder=2)
            #b.text(row.freq+.13,i,str(row.freq),fontsize=9,va='center')
        b.set_ylim(n-.5,-.5);b.set_xlim(0,max(1,summary.freq.max())+2.4)
        b.set_yticks(y);b.set_yticklabels([])
        b.set_xlabel('Outer folds selecting this candidate')
        b.text(0,1.035,'B)  Selection frequency',transform=b.transAxes,
               fontsize=10,color=PALETTE['ink'])
        lim=max(3.,float(np.nanmax(abs(delta.to_numpy()))))
        from matplotlib.colors import TwoSlopeNorm
        cmap=plt.get_cmap('RdBu')
        image=c.imshow(delta.to_numpy(),cmap=cmap,
                       norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),aspect='auto')
        c.set_yticks(y);c.set_yticklabels(selected,fontsize=8.3)
        c.set_xticks(np.arange(mat.shape[1]));c.set_xticklabels(
            [str(z) for z in mat.columns],fontsize=7.7)
        c.set_xlabel('Outer-fold index (each entry is an inner-CV result)')
        c.text(0,1.035,'C)  Paired gain relative to '+ref[stage],transform=c.transAxes,
               fontsize=10,color=PALETTE['ink'])
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
        d.text(0,1.035,'D)  Average paired change',transform=d.transAxes,
               fontsize=10,color=PALETTE['ink'])
        fig.suptitle(f'Step {stage}  |  {labels[stage]} ablation',
                     x=.195,ha='left',fontsize=14,weight='bold',color=PALETTE['ink'])
        #fig.text(.195,.026,'Descriptive inner cross-validation summaries; the selected candidate '
                 #'is refitted before the outer patient is evaluated.',
                 #fontsize=8,color=PALETTE['muted'])
        filename = f'0{stage}_' + {1:'classifier_ablation',
            2:'augmentation_ablation',3:'feature_engineering_ablation'}[stage] + '.pdf'
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
        ax2.text(.5,.5,'No discrete wavelength selections; full/projection representations chosen',
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




def plot_computation_effort(candidate_scores):
    """Three matched stage panels: held-in CV BA vs standalone cold-start time."""
    names = {1:'Classifiers',2:'Augmentation',3:'Feature extraction / selection'}
    colors = {1:PALETTE['navy'],2:PALETTE['teal'],3:PALETTE['blue']}
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.5))
    for stage, ax in enumerate(axes, 1):
        data = candidate_scores[candidate_scores.step == stage]
        g = data.groupby('candidate', sort=False).agg(
            BA=('BA', 'mean'), seconds=('cold_total_s', 'mean'),
            SD=('BA','std')).reset_index()
        for _, r in g.iterrows():
            x = max(float(r['seconds']), 1e-5)
            ax.errorbar(x, 100 * r.BA, yerr=100*(r.SD if np.isfinite(r.SD) else 0),
                        fmt='o', ms=5.5, capsize=2.5, color=colors[stage], alpha=.87)
            ax.annotate(r.candidate, (x, 100*r.BA), xytext=(4,5),
                        textcoords='offset points', fontsize=7.2, alpha=.9)
        ax.set_xscale('log')
        ax.set_ylim(0, 109)
        ax.set_xlabel('Cold-start time / inner fold (s, log scale)')
        ax.set_ylabel('Inner balanced accuracy (%)')
        # ax.set_title(f'Step {stage}  |  {names[stage]}',loc='left',weight='bold',fontsize=11)
        tidy(ax)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUTPUT_DIR/'11_accuracy_vs_compute.pdf', bbox_inches='tight')
    plt.close(fig)

    parts=[('generator_fit_s','Generator fit',PALETTE['navy']),
           ('sampling_s','Synthetic sampling',PALETTE['blue']),
           ('feature_s','Feature engineering',PALETTE['amber']),
           ('classifier_s','Classifier fit / predict',PALETTE['teal'])]
    fig,axes=plt.subplots(1,3,figsize=(15,7.0))
    for stage,ax in enumerate(axes,1):
        g=candidate_scores[candidate_scores.step==stage].groupby('candidate',sort=False).agg(
           **{key:(key,'mean') for key,_,_ in parts}).reset_index()
        g['total']=sum(g[key] for key,_,_ in parts)
        g=g.sort_values('total',ascending=True)
        y=np.arange(len(g)); left=np.zeros(len(g))
        for key,label,color in parts:
            v=g[key].to_numpy(float)
            ax.barh(y,v,left=left,height=.65,color=color,label=label)
            left+=v
        ax.set_yticks(y);ax.set_yticklabels(g.candidate,fontsize=8)
        ax.set_xlabel('Standalone cold-start time (s)')
        ax.set_title(f'Step {stage}', loc='left',weight='bold')
        tidy(ax)
    axes[0].legend(loc='upper left',frameon=False,fontsize=8,ncol=1)
    fig.tight_layout(w_pad=3.2)
    fig.savefig(OUTPUT_DIR/'12_compute_breakdown.pdf',bbox_inches='tight')
    plt.close(fig)


def save_efficiency_table(candidate_scores, outer_resources, run_wall, cpu_s, stage_search_s):
    g=candidate_scores.groupby(['step','candidate'],sort=False).agg(
        InnerBA=('BA','mean'),InnerAUC=('AUC','mean'),
        ColdSeconds=('cold_total_s','mean'),GenFitSeconds=('generator_fit_s','mean'),
        SamplingSeconds=('sampling_s','mean'),FeatureSeconds=('feature_s','mean'),
        ClassifierSeconds=('classifier_s','mean'),
        ClassifierFitSeconds=('classifier_fit_s','mean'),
        TestPredictSeconds=('test_predict_s','mean'),
        FeatureDim=('feature_dim','mean'),
        TrainMatrixMB=('train_matrix_mb','mean'),GenParams=('generator_params','mean'),
        MeanRMSE=('mean_rmse','mean'),StdRMSE=('std_rmse','mean'),
        Diversity=('diversity_ratio','mean'),NearRatio=('near_ratio','mean')
    ).reset_index()
    g=g.sort_values(['step','InnerBA','ColdSeconds'],ascending=[True,False,True])
    g.to_csv(OUTPUT_DIR/'computational_effort_comparison.csv',index=False)
    write_latex_table(g, [('step','Stage'),('candidate','Candidate'),
        ('InnerBA','Inner BA'),('ColdSeconds','Cold time (s)'),
        ('GenFitSeconds','Gen. fit (s)'),('SamplingSeconds','Sample (s)'),
        ('FeatureSeconds','Features (s)'),('ClassifierSeconds','Classifier (s)'),
        ('TestPredictSeconds','Test pred. (s)'),('FeatureDim','Dim.')],OUTPUT_DIR/'computational_effort_comparison.tex',
        caption='Mean standalone cold-start costs per inner fold. Generator training is '
                'cached across search candidates but charged to each candidate as if '
                'executed independently. The time to evaluate the full nested search '
                'is recorded separately. Timing is hardware-dependent.',
        label='tab:computational_effort',
        numeric={key:'.3f' for key in ('InnerBA','ColdSeconds','GenFitSeconds',
            'SamplingSeconds','FeatureSeconds','ClassifierSeconds',
            'TestPredictSeconds','FeatureDim')})
    outer_resources.to_csv(OUTPUT_DIR/'outer_stage_computational_effort.csv',index=False)
    actual = dict(run_wall_s=run_wall, run_cpu_s=cpu_s,
        total_inner_candidate_evaluations=int(candidate_scores.shape[0]),
        outer_pipeline_refits=int(len(outer_resources)),
        stage_search_wall_s={str(k):round(v,6) for k,v in stage_search_s.items()},
        peak_process_rss_mb=peak_process_rss_mb(),
        platform=sys.platform, cpu_threads=torch.get_num_threads(),
        device=str(DEVICE))
    (OUTPUT_DIR/'run_resources.json').write_text(json.dumps(actual,indent=2),encoding='utf-8')
    return g


def peak_process_rss_mb():
    # resource.ru_maxrss is kB on Linux, bytes on macOS, and process-wide.
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak/(1024**2) if sys.platform=='darwin' else peak/1024.


def save_augmentation_quality(val):
    """Aggregate TRAINING-FOLD synthetic fidelity separately from test metrics."""
    data=val[(val.step==2)&(val.candidate!='Real only')].copy()
    if len(data)==0:
        return
    data['Method']=data.candidate.str.rsplit(' ',n=1).str[0]
    data['PerClass']=data.candidate.str.rsplit(' ',n=1).str[-1].str.split('/').str[0].astype(int)
    group=data.groupby(['Method','PerClass'],as_index=False).agg(
        ValBA=('BA','mean'),ValAUC=('AUC','mean'),
        MeanRMSE=('mean_rmse','mean'),StdRMSE=('std_rmse','mean'),
        Diversity=('diversity_ratio','mean'),NearRatio=('near_ratio','mean'),
        ColdSeconds=('cold_total_s','mean'),GeneratorSeconds=('generator_fit_s','mean'),
        SamplingSeconds=('sampling_s','mean'),EvaluatedOuter=('outer_fold','nunique'))
    group.to_csv(OUTPUT_DIR/'augmentation_quality_comparison.csv',index=False)
    write_latex_table(group,[('Method','Augmentation'),('PerClass','Per class'),
        ('ValBA','Inner BA'),('MeanRMSE','Mean RMSE'),('StdRMSE','SD RMSE'),
        ('Diversity','Diversity'),('NearRatio','NN distance ratio'),
        ('ColdSeconds','Cold (s)')],OUTPUT_DIR/'augmentation_quality_comparison.tex',
        caption='Training-fold-only spectral fidelity and inner-CV classification results. '
                'Lower RMSE is closer in mean or standard deviation, diversity '
                'near unity matches real spectral dispersion, and nearest-neighbor '
                'distance ratio near unity matches real patient separation. '
                'None is a clinical validity guarantee.',
        label='tab:augmentation_quality', numeric={
            'ValBA':'.3f','MeanRMSE':'.3f','StdRMSE':'.3f','Diversity':'.3f',
            'NearRatio':'.3f','ColdSeconds':'.2f'})
    with (OUTPUT_DIR/'paper_tables.tex').open('a',encoding='utf-8') as f:
        f.write('\\input{augmentation_quality_comparison.tex}\n')
    if RUN_PLOTS:
        fig,axs=plt.subplots(1,3,figsize=(16.2,5.7))
        metrics=[('MeanRMSE','Spectral mean RMSE'),('Diversity','Diversity ratio'),
                 ('NearRatio','Nearest-real distance ratio')]
        for ax,(key,label) in zip(axs,metrics):
            for method,subset in group.groupby('Method',sort=False):
                ax.plot(subset.PerClass,subset[key],'o-',lw=1.2,ms=4,label=method)
            ax.set_xlabel('Synthetic samples per class')
            ax.set_ylabel(label)
            if key!='MeanRMSE':
                ax.axhline(1,color=PALETTE['muted'],ls='--',lw=.9)
            tidy(ax)
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False,fontsize=8)
        fig.tight_layout(rect=[0,.17,1,1])
        fig.savefig(OUTPUT_DIR/'13_augmentation_quality.pdf',bbox_inches='tight')
        plt.close(fig)


def save_synthetic_example_figure(wn,rec):
    """Fold-safe examples: uses only a training split from the FIRST outer fold."""
    if not RUN_PLOTS or not rec['synthetic_cache']:
        return
    methods=['Spectral Mixup','Spectral SMOTE','PCA-Gaussian','Gaussian Mixture',
             'Latent Flow','Conditional Diffusion','Conditional GAN',
             'Conditional VAE','Conditional WGAN-GP']
    selected=[]
    for method in methods:
        matching=[k for k in rec['synthetic_cache'] if k[0]==method]
        if matching:
            selected.append(max(matching,key=lambda x:x[1]))
    if not selected:
        return
    real=rec['xtr']; yreal=rec['ytr']
    fig,axes=plt.subplots(len(selected),2,figsize=(12,2.1*len(selected)),sharex=True)
    axes=np.asarray(axes).reshape(len(selected),2)
    for row,key in enumerate(selected):
        syn,ysyn=rec['synthetic_cache'][key]
        for c in (0,1):
            ax=axes[row,c]
            x=snv(real[yreal==c]);s=snv(syn[ysyn==c]);
            ax.plot(wn,x.mean(0),lw=1.3,color=PALETTE['navy'],label='Training real')
            ax.plot(wn,s.mean(0),lw=1.3,color=PALETTE['teal'],label='Training synthetic')
            ax.fill_between(wn,s.mean(0)-s.std(0),s.mean(0)+s.std(0),
                            color=PALETTE['teal'],alpha=.12,lw=0)
            ax.set_ylabel(key[0],fontsize=8)
            if row==0:
                ax.set_title(('Healthy' if c==0 else 'Diseased')+' (fold training only)',fontsize=10)
            tidy(ax)
    axes[-1,0].set_xlabel('Wavenumber (cm$^{-1}$)')
    axes[-1,1].set_xlabel('Wavenumber (cm$^{-1}$)')
    axes[0,0].legend(loc='upper left',frameon=False,fontsize=7)
    fig.tight_layout(h_pad=.9)
    fig.savefig(OUTPUT_DIR/'14_training_fold_synthetic_examples.pdf',bbox_inches='tight')
    plt.close(fig)


def independent_outer_record(xtr,ytr,xte,seed):
    return dict(xtr=xtr,ytr=ytr,xte=xte,seed=seed,synthetic_cache={},
                generator_models={},generator_fit_times={},sampling_times={},
                quality_cache={},selector_cache={},feature_fit_times={})


def save_final_seed_robustness(rows):
    if not rows:
        return
    d=pd.DataFrame(rows)
    d.to_csv(OUTPUT_DIR/'final_pipeline_seed_predictions.csv',index=False)
    summary=[]
    for rep,z in d.groupby('repeat'):
        summary.append(dict(Repeat=int(rep),**evaluate_predictions(
            z.y_true,z.y_pred,z.score)))
    z=pd.DataFrame(summary)
    z.to_csv(OUTPUT_DIR/'final_pipeline_seed_metrics.csv',index=False)
    studied=['BalancedAccuracy','Accuracy','Sensitivity','Specificity','AUC','AP','F1','MCC']
    stab=pd.DataFrame([dict(Metric=k,Mean=float(z[k].mean()),
        SD=float(z[k].std(ddof=1)) if len(z)>1 else np.nan,
        Minimum=float(z[k].min()),Maximum=float(z[k].max())) for k in studied])
    stab.to_csv(OUTPUT_DIR/'final_pipeline_seed_summary.csv',index=False)
    patients=d.groupby('patient',sort=False).agg(
        actual=('y_true','first'),correct_rate=('correct','mean'),
        prediction_rate=('y_pred','mean'),n_runs=('repeat','size')).reset_index()
    patients.to_csv(OUTPUT_DIR/'final_pipeline_patient_stability.csv',index=False)
    write_latex_table(z,[('Repeat','Seed repeat'),('BalancedAccuracy','BA'),
         ('Accuracy','Acc.'),('Sensitivity','Sens.'),('Specificity','Spec.'),
         ('AUC','AUC'),('Correct','Correct')],
         OUTPUT_DIR/'final_pipeline_seed_metrics.tex',
         caption='Repeat fits of the *same selected stage-3 pipeline* per outer '
          'patient using different random seeds. Repetitions share patients and '
          'are not independent test observations.',label='tab:final_seed_robustness',
         numeric={k:'.3f' for k in ('BalancedAccuracy','Accuracy','Sensitivity',
                                   'Specificity','AUC')})
    with (OUTPUT_DIR/'paper_tables.tex').open('a',encoding='utf-8') as f:
        f.write('\\input{final_pipeline_seed_metrics.tex}\n')


def append_factorial_fold(writer,rec,outer_fold):
    """Optional factorial audit, inner CV ONLY; never use test patients for tuning.

    One row per [outer fold, inner fold, classifier, augmentation, selector].
    Computationally VERY expensive for the 17 x 16 full LOOCV and every method.
    """
    for aug in AUGMENTATIONS:
        aa,yy=aug_dataset(rec,aug)
        meta=rec['last_aug_meta'].copy()
        for sel in SELECTORS:
            if sel not in rec['selector_cache']:
                rec['selector_cache'][sel]=feature_indices(sel,rec['xtr'],rec['ytr'],rec['seed'])
            idx=rec['selector_cache'][sel]
            xt=representation(rec['xte'],idx,None)
            xx=representation(aa,idx,None)
            for clf in MODELS:
                score,fit_s,predict_s=train_and_predict(clf,xx,yy,xt,rec['seed'])
                writer.writerow(dict(outer_fold=outer_fold,inner_fold=rec['inner_fold'],
                    classifier=clf,augmentation=aug,feature=sel,
                    y_true=rec['y_true'],score=score,y_pred=int(score>=0),
                    classifier_fit_s=fit_s,test_predict_s=predict_s,
                    generator_fit_s=meta['generator_fit_s'],sampling_s=meta['sampling_s']))


def summarize_factorial_audit(path):
    """Stream the optional full cross-product result file into ranked summaries.

    Inner folds are clustered by outer fold. Ranks are exploratory and should
    NOT be used to select a final pipeline from the same outer test cases.
    """
    accumulator={}
    keycols=['outer_fold','classifier','augmentation','feature','y_true']
    for chunk in pd.read_csv(path,chunksize=100000):
        chunk['correct']=(chunk.y_pred==chunk.y_true).astype(int)
        group=chunk.groupby(keycols,sort=False).agg(
            correct=('correct','sum'),n=('correct','size'),
            clf_s=('classifier_fit_s','sum'),gen_s=('generator_fit_s','sum'),
            sampling_s=('sampling_s','sum')).reset_index()
        for r in group.itertuples(index=False):
            key=(r.outer_fold,r.classifier,r.augmentation,r.feature,r.y_true)
            if key not in accumulator:
                accumulator[key]=[0.,0.,0.,0.,0.]
            a=accumulator[key]
            for i,v in enumerate((r.correct,r.n,r.clf_s,r.gen_s,r.sampling_s)):
                a[i]+=v
    rows=[]
    combinations=sorted(set(key[:4] for key in accumulator))
    for key in combinations:
        a=accumulator.get(key+(0,));b=accumulator.get(key+(1,))
        if a is None or b is None:
            raise AssertionError('Missing class in factorial-audit inner validation')
        ba=.5*(a[0]/a[1]+b[0]/b[1])
        rows.append(dict(outer_fold=key[0],classifier=key[1],augmentation=key[2],
            feature=key[3],BA=ba,MeanClassifierFit_s=(a[2]+b[2])/(a[1]+b[1]),
            MeanGeneratorFit_s=(a[3]+b[3])/(a[1]+b[1]),
            MeanSampling_s=(a[4]+b[4])/(a[1]+b[1])))
    if not rows:
        return
    folds=pd.DataFrame(rows)
    folds.to_csv(OUTPUT_DIR/'full_factorial_outer_fold_scores.csv',index=False)
    summary=folds.groupby(['classifier','augmentation','feature'],as_index=False).agg(
        BA=('BA','mean'),BA_SD=('BA','std'),
        ClassifierFit_s=('MeanClassifierFit_s','mean'),
        GeneratorFit_s=('MeanGeneratorFit_s','mean'),
        Sampling_s=('MeanSampling_s','mean'),OuterFolds=('outer_fold','nunique'))
    summary['ApproxTime_s']=summary[['ClassifierFit_s','GeneratorFit_s','Sampling_s']].sum(1)
    summary=summary.sort_values(['BA','ApproxTime_s'],ascending=[False,True])
    summary.to_csv(OUTPUT_DIR/'full_factorial_summary.csv',index=False)
    write_latex_table(summary.head(30),[('classifier','Classifier'),
         ('augmentation','Augmentation'),('feature','Features'),
         ('BA','Inner BA'),('BA_SD','SD'),('ApproxTime_s','Approx s')],
         OUTPUT_DIR/'full_factorial_top30.tex',
         caption='Top 30 full-factorial configurations according to mean nested '
                 'inner-fold balanced accuracy, descriptive only. Outer test '
                 'patients were never used in fitting these candidates.',
         label='tab:full_factorial_top', numeric={'BA':'.3f','BA_SD':'.3f',
                                                'ApproxTime_s':'.2f'})
    with (OUTPUT_DIR/'paper_tables.tex').open('a',encoding='utf-8') as f:
        f.write('\\input{full_factorial_top30.tex}\n')


def run_experiment():
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    seed_everything(SEED)
    torch.set_num_threads(min(3, torch.get_num_threads()))
    wn, raw, y, names = load_data()
    if len(AUG_COUNTS) < 1 or min(AUG_COUNTS) <= 0 or tuple(sorted(set(AUG_COUNTS))) != AUG_COUNTS:
        raise ValueError('AUG_COUNTS must contain increasing positive integers')
    if len(FEATURE_K) != 3 or tuple(sorted(set(FEATURE_K))) != FEATURE_K:
        raise ValueError('FEATURE_K must be three increasing feature counts')
    if OUTER_PER_CLASS is not None:
        rng = np.random.default_rng(SEED)
        outer_test = np.concatenate([rng.choice(np.flatnonzero(y==k),
            size=min(OUTER_PER_CLASS,int((y==k).sum())),replace=False)
            for k in (0,1)])
    else:
        outer_test = np.arange(len(y))
    candidate_scores=[]; selected_rows=[]; predictions=[]; features=[]; resource_rows=[]
    robust_rows=[]
    import csv
    audit_f=None; audit_writer=None
    if RUN_FACTORIAL_AUDIT:
        audit_f=(OUTPUT_DIR/'full_factorial_inner_fold_results.csv').open('w',newline='',encoding='utf-8')
        audit_writer=csv.DictWriter(audit_f,fieldnames=[
            'outer_fold','inner_fold','classifier','augmentation','feature',
            'y_true','score','y_pred','classifier_fit_s','test_predict_s',
            'generator_fit_s','sampling_s'])
        audit_writer.writeheader()
    stage_search_s={1:0.,2:0.,3:0.}
    selected_set = set(outer_test.tolist())
    for outer_fold, (train, held) in enumerate(LeaveOneOut().split(raw), 1):
        test=int(held[0])
        if test not in selected_set:
            continue
        inner = make_inner_records(raw, y, train, outer_fold)
        held_labels = [r['y_true'] for r in inner]
        if min(np.bincount(held_labels, minlength=2)) < 1:
            raise ValueError('Inner validation requires held-out patients from both classes')
        outer_seed=SEED + outer_fold * 100003
        outer_record = dict(xtr=snv(raw[train]), ytr=y[train],
            xte=snv(raw[test:test+1]), seed=outer_seed, synthetic_cache={},
            generator_models={}, generator_fit_times={}, sampling_times={},
            quality_cache={}, selector_cache={}, feature_fit_times={})

        # Sequential nested selection. No previously specified pipeline is a benchmark.
        step_clock=time.perf_counter()
        a=evaluate_candidates(inner, MODELS, None, 'Real only', 'Full spectrum',
                              1, None, 'model')
        stage_search_s[1]+=time.perf_counter()-step_clock
        model, by=select_winner(a, MODELS)
        step_clock=time.perf_counter()
        b=evaluate_candidates(inner, AUGMENTATIONS, model, None, 'Full spectrum',
                              2, None, 'augmentation')
        stage_search_s[2]+=time.perf_counter()-step_clock
        aug, by2=select_winner(b, AUGMENTATIONS)
        step_clock=time.perf_counter()
        c=evaluate_candidates(inner, SELECTORS, model, aug, None,
                              3, None, 'selector')
        stage_search_s[3]+=time.perf_counter()-step_clock
        selector,by3=select_winner(c,SELECTORS)
        if RUN_PLOTS and len(selected_rows)==0:
            save_synthetic_example_figure(wn,inner[0])
        if audit_writer is not None:
            for rec in inner:
                append_factorial_fold(audit_writer,rec,outer_fold)
            audit_f.flush()
        for step, chosen, scores, byscore in [(1,model,a,by), (2,aug,b,by2),
                                               (3,selector,c,by3)]:
            selected_rows.append(dict(outer_fold=outer_fold,held_out=names[test],
                step=step,candidate=chosen,inner_BA=byscore[chosen]['BA'],
                inner_AUC=byscore[chosen]['AUC']))
            for cand, vals in byscore.items():
                rr = scores.loc[scores.candidate == cand]
                cost_cols = ['classifier_s','feature_s','generator_fit_s',
                    'sampling_s','cold_total_s','classifier_fit_s','test_predict_s',
                    'feature_dim','train_matrix_mb','generator_params',
                    'mean_rmse','std_rmse','diversity_ratio','near_ratio']
                costs = {col: float(rr[col].mean()) for col in cost_cols}
                candidate_scores.append(dict(outer_fold=outer_fold,step=step,
                    candidate=cand,**vals,**costs))
        for step, mod, ag, sel in [(1,model,'Real only','Full spectrum'),
                                  (2,model,aug,'Full spectrum'),
                                  (3,model,aug,selector)]:
            score, cost=outer_predict(raw,y,train,test,mod,ag,sel,outer_seed,
                                      None,outer_record)
            predictions.append(dict(outer_fold=outer_fold,patient=names[test],
                step=step,model=mod,augmentation=ag,selector=sel,
                y_true=int(y[test]),y_pred=int(score>=0),score=score))
            resource_rows.append(dict(outer_fold=outer_fold,step=step,
                patient=names[test],model=mod,augmentation=ag,selector=sel,**cost))
            if step==3:
                robust_rows.append(dict(outer_fold=outer_fold,patient=names[test],
                    repeat=0,seed=outer_seed,model=mod,augmentation=ag,selector=sel,
                    y_true=int(y[test]),y_pred=int(score>=0),score=score,
                    correct=int((score>=0)==bool(y[test]))))
                for rep in range(1,FINAL_SEED_REPEATS):
                    repeat_seed=outer_seed+rep*10000019
                    new_record=independent_outer_record(outer_record['xtr'],
                        outer_record['ytr'],outer_record['xte'],repeat_seed)
                    newscore,newcost=outer_predict(raw,y,train,test,mod,ag,sel,
                                      repeat_seed,None,new_record)
                    robust_rows.append(dict(outer_fold=outer_fold,patient=names[test],
                        repeat=rep,seed=repeat_seed,model=mod,augmentation=ag,
                        selector=sel,y_true=int(y[test]),y_pred=int(newscore>=0),
                        score=newscore,correct=int((newscore>=0)==bool(y[test]))))
        if selector != 'Full spectrum':
            ids = outer_record['selector_cache'][selector]
            if isinstance(ids,np.ndarray):
                for j in ids:
                    features.append(dict(outer_fold=outer_fold,selector=selector,
                        channel=int(j),wavenumber=float(wn[j])))
        print(f'[{RUN_MODE}] {len(set(r["outer_fold"] for r in selected_rows)):02d}/'
              f'{len(outer_test):02d}  {names[test]} => '
              f'{model} | {aug} | {selector}', flush=True)

    if audit_f is not None:
        audit_f.close()
    sel=pd.DataFrame(selected_rows)
    val=pd.DataFrame(candidate_scores)
    pred=pd.DataFrame(predictions)
    outer_resource=pd.DataFrame(resource_rows)
    sel.to_csv(OUTPUT_DIR/'selected_configurations_by_outer_fold.csv',index=False)
    val.to_csv(OUTPUT_DIR/'inner_cv_candidate_scores.csv',index=False)
    pred.to_csv(OUTPUT_DIR/'nested_patient_predictions.csv',index=False)
    pd.DataFrame(features,columns=['outer_fold','selector','channel','wavenumber']).to_csv(
        OUTPUT_DIR/'selected_features_by_outer_fold.csv',index=False)
    metrics=save_stage_tables(val,sel,pred)
    paired=paired_comparisons(pred)
    if RUN_PLOTS:
        plot_stage_results(val,sel,metrics,pred)
        plot_selection_paths(sel)
        plot_feature_frequency(wn,raw,y,features,len(outer_test))
        plot_paired_stage_forest(paired)
        plot_stage_confusion(pred)
        plot_computation_effort(val)
    effort=save_efficiency_table(val,outer_resource,
        time.perf_counter()-start_wall,time.process_time()-start_cpu,stage_search_s)
    with (OUTPUT_DIR/'paper_tables.tex').open('a',encoding='utf-8') as f:
        f.write('\\input{computational_effort_comparison.tex}\n')
    save_augmentation_quality(val)
    save_final_seed_robustness(robust_rows)
    if RUN_FACTORIAL_AUDIT:
        summarize_factorial_audit(OUTPUT_DIR/'full_factorial_inner_fold_results.csv')
    config=dict(mode=RUN_MODE,n_total_patients=len(y),
        n_tested_patients=len(outer_test),n_healthy=int((y==0).sum()),
        n_sick=int((y==1).sum()), channels=raw.shape[1],
        root=str(ROOT), healthy_directory=str(HEALTHY_DIR),
        sick_directory=str(SICK_DIR), classifiers=MODELS,
        augmentations=AUGMENTATIONS,selectors=SELECTORS,
        feature_k=FEATURE_K,
        selection_criterion='inner patient LOOCV balanced_accuracy > AUC > accuracy > predefined candidate order',
        outer_protocol='patient-level LOOCV; within each fold, nested sequential selection',
        results_status='QUICK_SMOKE_TEST' if FAST else ('NORMAL_SUBSET_DIAGNOSTIC' if SUBSET
                          else 'EXPLORATORY_FULL_NESTED_LOOCV'),
        auc_note='Pooled outer AUC uses training-std-normalized decision margins, not calibrated probabilities.',
        augmentation_note='All generative methods are class-conditional in a training-fold PCA latent space.',
        compute_note='Candidate cold-start costs charge cached generator fits to every candidate; total actual execution in run_resources.json.',
        flow_steps=FLOW_STEPS,flow_ode_steps=FLOW_ODE_STEPS,
        diffusion_steps=DIFFUSION_STEPS,diffusion_timesteps=DIFFUSION_T,
        gan_steps=GAN_STEPS,vae_steps=VAE_STEPS,wgan_steps=WGAN_STEPS,
        latent_dim=LATENT_DIM,full_sequential_coverage=not SUBSET,
        full_cartesian_grid=RUN_FACTORIAL_AUDIT,plots=RUN_PLOTS,
        quality_diagnostics=RUN_QUALITY,final_seed_repeats=FINAL_SEED_REPEATS,
        augmentation_families=STATISTICAL_AUGMENTATIONS+NEURAL_AUGMENTATIONS,
        factorial_warning='Off by default: Cartesian factorial audit is expensive', 
        augmentation_counts=AUG_COUNTS,seed=SEED,python=sys.version,
        sklearn_version=__import__('sklearn').__version__,
        torch_version=torch.__version__)
    (OUTPUT_DIR/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Finished {RUN_MODE}: {len(outer_test)} outer patients; outputs: {OUTPUT_DIR}')
    return pred,metrics,sel,val,paired,effort


if __name__ == '__main__':
    run_experiment()
