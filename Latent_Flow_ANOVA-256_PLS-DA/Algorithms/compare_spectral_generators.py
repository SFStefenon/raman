"""Compare five binary-class spectral augmentation methods on doentes.zip / saudaveis.zip.

Requirements:
    pip install numpy pandas scipy scikit-learn matplotlib torch

Each CSV in the original ZIPs is ONE patient sample (not 2048 patients).
Generates class-conditional synthetic spectra and comparisons:
    spectral_mixup, pca_gaussian, cwgan_gp, cddpm, latent_flow

Exports under OUTPUT_DIR:
  generated_METHOD.zip    -- individual 2-column CSV spectra in original units
  generated_METHOD.npz    -- X_snv, X_original, y, wavenumber, class labels
  quality_metrics.csv     -- descriptive in-sample fidelity checks, NOT test metrics
  augmentation_comparison.pdf
  classification_loocv.csv and classification_predictions.csv when RUN_LOOCV

WARNING: 17 real patients are insufficient for credible clinical validation.
No synthetic example may be used in a held-out patient test fold.
"""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO, StringIO
import re
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(globals().get("ROOT", Path(__file__).resolve().parent))
OUTPUT_DIR = Path(globals().get("OUTPUT_DIR", ROOT / "synthetic_comparison"))
SEED = int(globals().get("SEED", 42))
LATENT_DIM = int(globals().get("LATENT_DIM", 4))
N_SYNTHETIC_PER_CLASS = int(globals().get("N_SYNTHETIC_PER_CLASS", 100))
TRAIN_STEPS = int(globals().get("TRAIN_STEPS", 500))
BATCH_SIZE = int(globals().get("BATCH_SIZE", 64))
RUN_LOOCV = bool(globals().get("RUN_LOOCV", True))
LOOCV_SYNTHETIC_PER_CLASS = int(globals().get("LOOCV_SYNTHETIC_PER_CLASS", 30))
LOOCV_TRAIN_STEPS = int(globals().get("LOOCV_TRAIN_STEPS", TRAIN_STEPS))
SAVE_INDIVIDUAL_CSVS = bool(globals().get("SAVE_INDIVIDUAL_CSVS", True))
METHODS = ["spectral_mixup", "pca_gaussian", "cwgan_gp", "cddpm", "latent_flow"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(min(4, torch.get_num_threads()))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_archives(root):
    records = []
    for zipname, label in (("saudaveis.zip", 0), ("doentes.zip", 1)):
        with ZipFile(root / zipname) as arc:
            for path in sorted(arc.namelist()):
                if not path.lower().endswith(".csv"):
                    continue
                if any(k in path.lower() for k in ("checkpoint", "__macosx")):
                    continue
                payload = arc.read(path)
                header = payload.splitlines()[0]
                sep = ";" if header.count(b";") > header.count(b",") else ","
                frame = pd.read_csv(BytesIO(payload), sep=sep, encoding="utf-8-sig")
                if frame.shape[1] != 2:
                    raise ValueError(f"Expected two columns: {path}")
                wn = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(float)
                signal = pd.to_numeric(frame.iloc[:, 1], errors="raise").to_numpy(float)
                if not np.isfinite(wn).all() or not np.isfinite(signal).all():
                    raise ValueError(f"Non-finite spectrum: {path}")
                stem = Path(path).stem
                disease = ("healthy" if label == 0 else
                           "dcr" if "DCR" in stem.upper() else
                           "lra" if "LRA" in stem.upper() else
                           "glomerulopathy" if "Glomerulopatia" in stem else
                           "lupus_nephritis" if "Nefrite" in stem else "other_sick")
                records.append((stem, label, disease, wn, signal))
    if not records:
        raise ValueError("No CSVs found in the ZIPs")
    ref = records[0][3]
    for stem, _, _, wn, signal in records:
        if len(wn) != len(ref) or not np.allclose(wn, ref, rtol=0, atol=1e-6):
            raise ValueError(f"Wavenumber grid differs: {stem}")
    return (ref, np.stack([r[4] for r in records]),
            np.asarray([r[1] for r in records], dtype=int),
            np.asarray([r[2] for r in records]),
            np.asarray([r[0] for r in records]))


def snv(X):
    """Standard normal variate, computed independently per patient."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return (X - X.mean(axis=1, keepdims=True)) / np.maximum(
        X.std(axis=1, keepdims=True), 1e-10)


class Latent:
    """Fit PCA and scaling ONLY on supplied training patient spectra."""
    def __init__(self, X):
        X = snv(X)
        d = min(LATENT_DIM, len(X) - 2, X.shape[1])
        if d < 1:
            raise ValueError("Need more independent patient samples")
        self.pca = PCA(n_components=d, random_state=SEED)
        self.scale = StandardScaler()
        self.z_train = self.scale.fit_transform(self.pca.fit_transform(X))
        self.dim = d

    def inverse(self, z):
        x = self.pca.inverse_transform(self.scale.inverse_transform(z))
        return snv(x)

    def project(self, x):
        return self.scale.transform(self.pca.transform(snv(x)))


def mixup_perturb(X, y, disease, labels, rng):
    """Same-subtype mixup plus small smooth baseline, shift and jitter."""
    x = snv(X)
    t = np.arange(x.shape[1], dtype=float)
    output = []
    for label in labels:
        possible = np.flatnonzero(y == label)
        first = int(rng.choice(possible))
        partners = possible[disease[possible] == disease[first]]
        second = int(rng.choice(partners))
        lam = float(rng.beta(0.3, 0.3)) if first != second else 1.0
        curve = lam * x[first] + (1.0 - lam) * x[second]
        # Perturbation amplitudes refer to SNV units. These are weak priors,
        # not a substitute for learning the true measurement noise process.
        shift = rng.uniform(-0.65, 0.65)  # sub-grid shift
        curve = np.interp(t + shift, t, curve, left=curve[0], right=curve[-1])
        u = np.linspace(-1.0, 1.0, x.shape[1])
        baseline = rng.normal(0, 0.008) + rng.normal(0, 0.006) * u
        noise = rng.normal(0, 0.007, size=len(t))
        output.append(snv((curve + baseline + noise)[None])[0])
    return np.asarray(output)


def gaussian_latents(z, y, labels, rng):
    """One shrinkage-covariance Gaussian per class, robust to small n."""
    out = np.empty((len(labels), z.shape[1]), dtype=float)
    for c in (0, 1):
        data = z[y == c]
        if len(data) < 2:
            raise ValueError(f"Class {c} needs at least 2 real patients")
        cov = LedoitWolf().fit(data).covariance_
        cov += np.eye(z.shape[1]) * 1e-5
        out[labels == c] = rng.multivariate_normal(
            data.mean(axis=0), cov, size=int((labels == c).sum()))
    return out


def cond(z, c):
    return torch.cat((z, F.one_hot(c.long(), num_classes=2).float()), dim=-1)


def time_features(t):
    a = 2 * np.pi * t
    return torch.cat([t, torch.sin(a), torch.cos(a), torch.sin(2*a),
                      torch.cos(2*a)], dim=1)


def mlp(nin, nout, hidden=64):
    return nn.Sequential(nn.Linear(nin, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, nout))


def train_gan(z_numpy, y_numpy, steps, rng):
    """Class-conditional Wasserstein GAN with gradient penalty (WGAN-GP)."""
    z = torch.tensor(z_numpy, dtype=torch.float32, device=DEVICE)
    labels = torch.tensor(y_numpy, dtype=torch.long, device=DEVICE)
    d = z.shape[1]
    G = mlp(8 + 2, d).to(DEVICE)
    D = mlp(d + 2, 1).to(DEVICE)
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    inds = [torch.where(labels == c)[0] for c in (0, 1)]
    for _ in range(steps):
        for _ in range(3):
            c = torch.randint(0, 2, (BATCH_SIZE,), device=DEVICE)
            real_idx = torch.empty(BATCH_SIZE, dtype=torch.long, device=DEVICE)
            for cls in (0, 1):
                mask = c == cls
                real_idx[mask] = inds[cls][torch.randint(
                    0, len(inds[cls]), (int(mask.sum()),), device=DEVICE)]
            real = z[real_idx]
            with torch.no_grad():
                fake = G(cond(torch.randn(BATCH_SIZE, 8, device=DEVICE), c))
            a = torch.rand(BATCH_SIZE, 1, device=DEVICE)
            inter = (a * real + (1 - a) * fake).detach().requires_grad_(True)
            grad = torch.autograd.grad(D(cond(inter, c)).sum(), inter,
                                       create_graph=True)[0]
            penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
            dloss = D(cond(fake, c)).mean() - D(cond(real, c)).mean() + 5.0 * penalty
            opt_d.zero_grad(set_to_none=True)
            dloss.backward()
            opt_d.step()
        c = torch.randint(0, 2, (BATCH_SIZE,), device=DEVICE)
        fake = G(cond(torch.randn(BATCH_SIZE, 8, device=DEVICE), c))
        gloss = -D(cond(fake, c)).mean()
        opt_g.zero_grad(set_to_none=True)
        gloss.backward()
        opt_g.step()
    return G.eval()


def cosine_betas(T):
    steps = np.arange(T + 1) / T
    alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
    alpha_bar /= alpha_bar[0]
    return np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 1e-4, 0.999)


def train_ddpm(z_numpy, y_numpy, steps, rng, T=64):
    """Conditional latent DDPM with cosine noise schedule."""
    x = torch.tensor(z_numpy, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_numpy, dtype=torch.long, device=DEVICE)
    d = x.shape[1]
    model = mlp(d + 2 + 5, d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    beta = torch.tensor(cosine_betas(T), dtype=torch.float32, device=DEVICE)
    alpha = 1 - beta
    abar = torch.cumprod(alpha, dim=0)
    class_indices = [torch.where(y == k)[0] for k in (0, 1)]
    for _ in range(steps):
        c = torch.randint(0, 2, (BATCH_SIZE,), device=DEVICE)
        idx = torch.empty(BATCH_SIZE, dtype=torch.long, device=DEVICE)
        for k in (0, 1):
            mask = c == k
            idx[mask] = class_indices[k][torch.randint(
                0, len(class_indices[k]), (int(mask.sum()),), device=DEVICE)]
        real = x[idx]
        t = torch.randint(0, T, (BATCH_SIZE,), device=DEVICE)
        eps = torch.randn_like(real)
        xt = abar[t, None].sqrt() * real + (1 - abar[t, None]).sqrt() * eps
        tt = (t.float() / max(T-1, 1)).unsqueeze(1)
        pred = model(torch.cat([cond(xt, c), time_features(tt)], dim=1))
        loss = F.mse_loss(pred, eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval(), beta, alpha, abar


def train_flow(z_numpy, y_numpy, steps, rng):
    """Conditional rectified flow: v(x_t,t,c) predicts data - noise."""
    x = torch.tensor(z_numpy, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_numpy, dtype=torch.long, device=DEVICE)
    d = x.shape[1]
    model = mlp(d + 2 + 5, d).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_indices = [torch.where(y == k)[0] for k in (0, 1)]
    for _ in range(steps):
        c = torch.randint(0, 2, (BATCH_SIZE,), device=DEVICE)
        idx = torch.empty(BATCH_SIZE, dtype=torch.long, device=DEVICE)
        for k in (0, 1):
            mask = c == k
            idx[mask] = class_indices[k][torch.randint(
                0, len(class_indices[k]), (int(mask.sum()),), device=DEVICE)]
        data = x[idx]
        noise = torch.randn_like(data)
        t = torch.rand(BATCH_SIZE, 1, device=DEVICE)
        xt = (1 - t) * noise + t * data
        velocity = model(torch.cat([cond(xt, c), time_features(t)], dim=1))
        loss = F.mse_loss(velocity, data - noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


def neural_sample(method, z, y, labels, steps, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    d, n = z.shape[1], len(labels)
    c = torch.tensor(labels, dtype=torch.long, device=DEVICE)
    if method == "cwgan_gp":
        G = train_gan(z, y, steps, rng)
        with torch.no_grad():
            out = G(cond(torch.randn(n, 8, device=DEVICE), c))
    elif method == "cddpm":
        T = 64
        model, beta, alpha, abar = train_ddpm(z, y, steps, rng, T)
        with torch.no_grad():
            out = torch.randn(n, d, device=DEVICE)
            for ti in range(T-1, -1, -1):
                t = torch.full((n, 1), ti / (T-1), device=DEVICE)
                eps = model(torch.cat([cond(out, c), time_features(t)], dim=1))
                mean = (out - beta[ti] * eps / torch.sqrt(1 - abar[ti])) / torch.sqrt(alpha[ti])
                if ti > 0:
                    var = beta[ti] * (1 - abar[ti-1]) / (1 - abar[ti])
                    out = mean + torch.sqrt(var) * torch.randn_like(out)
                else:
                    out = mean
    elif method == "latent_flow":
        model = train_flow(z, y, steps, rng)
        n_ode = 32
        dt = 1.0 / n_ode
        with torch.no_grad():
            out = torch.randn(n, d, device=DEVICE)
            for j in range(n_ode):
                t0 = torch.full((n, 1), j / n_ode, device=DEVICE)
                t1 = torch.full((n, 1), (j + 1) / n_ode, device=DEVICE)
                v0 = model(torch.cat([cond(out, c), time_features(t0)], dim=1))
                proposed = out + dt * v0
                v1 = model(torch.cat([cond(proposed, c), time_features(t1)], dim=1))
                out = out + 0.5 * dt * (v0 + v1)  # Heun ODE solver
    else:
        raise ValueError(method)
    array = out.detach().cpu().numpy().astype(float)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"Invalid generated latent values for {method}")
    return array


def generate(method, X_train, y_train, subtype_train, count_per_class, steps, seed):
    """Generate SNV-normalized synthetic spectra, training from this fold ONLY."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], count_per_class).astype(int)
    if method == "spectral_mixup":
        generated = mixup_perturb(X_train, y_train, subtype_train, labels, rng)
    else:
        latent = Latent(X_train)
        if method == "pca_gaussian":
            z_fake = gaussian_latents(latent.z_train, y_train, labels, rng)
        else:
            z_fake = neural_sample(method, latent.z_train, y_train, labels, steps, seed)
        generated = latent.inverse(z_fake)
    if not np.isfinite(generated).all():
        raise FloatingPointError(f"Non-finite synthetic spectra: {method}")
    return generated, labels


def restore_original_units(X_snv, labels, raw_train, y_train, seed):
    """Sample donor intensity mean/std from same-class real training patients."""
    rng = np.random.default_rng(seed)
    out = np.empty_like(X_snv)
    for j, cls in enumerate(labels):
        donor = raw_train[int(rng.choice(np.where(y_train == cls)[0]))]
        out[j] = X_snv[j] * max(donor.std(), 1e-10) + donor.mean()
    return out


def save_generated(method, wn, Xsnv, Xoriginal, labels):
    np.savez_compressed(OUTPUT_DIR / f"generated_{method}.npz",
                        wavenumber=wn, X_snv=Xsnv.astype(np.float32),
                        X_original=Xoriginal.astype(np.float32), y=labels,
                        description="Synthetic, not independent patient data")
    if not SAVE_INDIVIDUAL_CSVS:
        return
    # Matches the original two-column format and keeps model identity in filename.
    with ZipFile(OUTPUT_DIR / f"generated_{method}.zip", "w", ZIP_DEFLATED,
                 compresslevel=6) as archive:
        for j, (curve, cls) in enumerate(zip(Xoriginal, labels)):
            dest = "saudaveis" if cls == 0 else "doentes"
            fname = f"{method}/{dest}/SYN_{j:04d}_{'healthy' if cls==0 else 'sick'}.csv"
            df = pd.DataFrame({"Wavenumber (cm-1)": wn,
                               "Intensity (a.u.)": curve})
            archive.writestr(fname, df.to_csv(index=False, sep=";", float_format="%.9g"))


def descriptive_quality(method, X_real, y_real, X_gen, y_gen, seed):
    """In-sample distribution checks. Do NOT interpret as clinical validation."""
    rng = np.random.default_rng(seed)
    latent = Latent(X_real)
    rows = []
    for cls in (0, 1):
        a = snv(X_real[y_real == cls])
        b = snv(X_gen[y_gen == cls])
        d_rr = cdist(a, a)
        np.fill_diagonal(d_rr, np.inf)
        real_nn = np.median(d_rr.min(axis=1))
        synthetic_nn = np.median(cdist(b, a).min(axis=1))
        z_r, z_g = latent.project(a), latent.project(b)
        swd = []
        for _ in range(64):
            direction = rng.normal(size=z_r.shape[1])
            direction /= np.linalg.norm(direction)
            swd.append(wasserstein_distance(z_r @ direction, z_g @ direction))
        rows.append({
            "model": method, "class": cls, "n_original": len(a), "n_synthetic": len(b),
            "mean_spectrum_rmse_snv": np.sqrt(np.mean((a.mean(0)-b.mean(0))**2)),
            "std_spectrum_rmse_snv": np.sqrt(np.mean((a.std(0)-b.std(0))**2)),
            "diversity_ratio": float(pdist(b).mean() / max(pdist(a).mean(), 1e-12)),
            "nearest_real_distance_ratio": float(synthetic_nn / max(real_nn, 1e-12)),
            "sliced_wasserstein_pca": float(np.mean(swd)),
        })
    return rows


def plot_examples(wn, X_real, y_real, synthetic):
    fig, axes = plt.subplots(len(METHODS), 2, figsize=(12, 13), sharex=True)
    for i, method in enumerate(METHODS):
        X_fake, y_fake = synthetic[method]
        for cls in (0, 1):
            ax = axes[i, cls]
            real = snv(X_real[y_real == cls])
            fake = X_fake[y_fake == cls]
            for curve in real:
                ax.plot(wn, curve, color="0.65", alpha=0.26, linewidth=0.55)
            for curve in fake[:min(10, len(fake))]:
                ax.plot(wn, curve, color="C0" if cls == 0 else "C1",
                        alpha=0.22, linewidth=0.6)
            ax.plot(wn, real.mean(axis=0), color="black", lw=1.4, label="Real mean")
            ax.plot(wn, fake.mean(axis=0), color="C0" if cls == 0 else "C1",
                    lw=1.3, label="Synthetic mean")
            if i == 0:
                ax.set_title("Healthy" if cls == 0 else "Sick")
            if cls == 0:
                ax.set_ylabel(f"{method}\nSNV intensity")
            ax.grid(False)
            ax.legend(frameon=False, loc="upper left", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "augmentation_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def evaluate_loocv(X_real, y_real, subtype):
    """Per-patient outer LOOCV; fit every generator inside each training fold."""
    predictions = []
    splitter = LeaveOneOut()
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_real), start=1):
        Xtr, ytr = X_real[train_idx], y_real[train_idx]
        xt, yt = X_real[test_idx], int(y_real[test_idx[0]])
        for mi, method in enumerate(["real_only"] + METHODS):
            if method == "real_only":
                X_aug, y_aug = Xtr, ytr
            else:
                Xnew, ynew = generate(method, Xtr, ytr, subtype[train_idx],
                                      LOOCV_SYNTHETIC_PER_CLASS,
                                      LOOCV_TRAIN_STEPS, SEED + 10000*fold + mi)
                X_aug = np.concatenate([Xtr, Xnew])
                y_aug = np.concatenate([ytr, ynew])
            svm = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0,
                                                     class_weight="balanced"))
            svm.fit(snv(X_aug), y_aug)
            pred = int(svm.predict(snv(xt))[0])
            score = float(svm.decision_function(snv(xt))[0])
            predictions.append(dict(model=method, patient_index=int(test_idx[0]),
                                    actual=yt, predicted=pred, score=score))
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUT_DIR / "classification_predictions.csv", index=False)
    rows = []
    for name, group in pred_df.groupby("model", sort=False):
        actual, pred, score = (group[k].to_numpy() for k in
                               ("actual", "predicted", "score"))
        tn, fp, fn, tp = confusion_matrix(actual, pred, labels=[0, 1]).ravel()
        rows.append(dict(model=name, accuracy=accuracy_score(actual, pred),
                         balanced_accuracy=balanced_accuracy_score(actual, pred),
                         sensitivity_sick=tp/max(tp+fn,1),
                         specificity_healthy=tn/max(tn+fp,1),
                         roc_auc=roc_auc_score(actual, score),
                         correct=int((actual==pred).sum()), total=len(actual),
                         TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn)))
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "classification_loocv.csv", index=False)
    return out


wn, X, y, subtype, ids = load_archives(ROOT)
if np.bincount(y, minlength=2).min() < 3:
    raise ValueError("Each class needs at least 3 original patients")
print(f"Original dataset: {len(X)} patients, {X.shape[1]} wavenumbers, "
      f"healthy={(y==0).sum()}, sick={(y==1).sum()}, device={DEVICE}", flush=True)
all_synthetic, quality_rows = {}, []
for m, method in enumerate(METHODS):
    synthetic, labels = generate(method, X, y, subtype, N_SYNTHETIC_PER_CLASS,
                                 TRAIN_STEPS, SEED + 100*m)
    restored = restore_original_units(synthetic, labels, X, y, SEED + 100*m + 1)
    save_generated(method, wn, synthetic, restored, labels)
    all_synthetic[method] = (synthetic, labels)
    quality_rows.extend(descriptive_quality(method, X, y, synthetic, labels, SEED+m))
    print(f"{method}: {len(synthetic)} synthetic spectra exported", flush=True)
pd.DataFrame(quality_rows).to_csv(OUTPUT_DIR / "quality_metrics.csv", index=False)
plot_examples(wn, X, y, all_synthetic)
if RUN_LOOCV:
    metrics = evaluate_loocv(X, y, subtype)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"Results: {OUTPUT_DIR.resolve()}", flush=True)
