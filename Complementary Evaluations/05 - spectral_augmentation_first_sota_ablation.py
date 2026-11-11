"""Augmentation-first, nested patient-level Raman spectral benchmark.

Input folders beside this script (or override ROOT):
    saudaveis/*.csv  -- healthy label 0
    doentes/*.csv    -- sick label 1
Each original-patient CSV: two columns (wavenumber, intensity); shared grid.

Stage 1: With one FIXED prespecified anchor (Residual MLP + multi-resolution
         spectral pooling), select augmentation inside outer-training patients.
Stage 2: Holding that selected augmentation fixed, jointly ablate modern
         neural classifiers and train-fold-safe spectral representations.
Stage 3: Refit all chosen / comparator pipelines on outer-training patients
         and predict each untouched original held-out patient exactly once.

All feature learning and generation take place strictly inside the appropriate
training fold. Nothing is optimized from outer-test performance. Only the
real-only anchor is used for comparing augmentation methods on outer patients.

Quick mode is a smoke test, not a paper result. Even complete nested validation
with 17 patients is exploratory and requires external confirmation.

Dependencies: numpy pandas scipy scikit-learn matplotlib torch (Python 3.9+).
Optional: tabpfn with separately available pretrained model weights.
"""
from __future__ import annotations
import json
import random
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

# ------------------------------ USER SETTINGS ------------------------------
RUN_MODE = globals().get('RUN_MODE', 'complete')     # 'quick' or 'complete'
ROOT = Path(globals().get('ROOT', Path(__file__).resolve().parent))
HEALTHY_DIR = Path(globals().get('HEALTHY_DIR', ROOT / 'saudaveis'))
SICK_DIR = Path(globals().get('SICK_DIR', ROOT / 'doentes'))
OUTPUT_DIR = Path(globals().get('OUTPUT_DIR', ROOT / ('augmentation_first_' + RUN_MODE)))
SEED = int(globals().get('SEED', 1))
DEVICE = torch.device(globals().get('DEVICE', 'cpu'))
INCLUDE_TABPFN = bool(globals().get('INCLUDE_TABPFN', False))
OUTER_QUICK_PER_CLASS = int(globals().get('OUTER_QUICK_PER_CLASS', 2))
INNER_SPLITS = int(globals().get('INNER_SPLITS', 2 if RUN_MODE == 'quick' else 3))
CLASSIFIER_EPOCHS = int(globals().get('CLASSIFIER_EPOCHS', 12 if RUN_MODE == 'quick' else 45))
FEATURE_EPOCHS = int(globals().get('FEATURE_EPOCHS', 12 if RUN_MODE == 'quick' else 60))
GENERATOR_STEPS = int(globals().get('GENERATOR_STEPS', 24 if RUN_MODE == 'quick' else 240))
DIFFUSION_STEPS = int(globals().get('DIFFUSION_STEPS', 12 if RUN_MODE == 'quick' else 48))
ODE_STEPS = int(globals().get('ODE_STEPS', 8 if RUN_MODE == 'quick' else 24))
AUG_COUNTS = tuple(globals().get('AUG_COUNTS', (4,) if RUN_MODE == 'quick' else (10, 20)))
BOOTSTRAPS = int(globals().get('BOOTSTRAPS', 300 if RUN_MODE == 'quick' else 2500))
if RUN_MODE not in ('quick', 'complete'):
    raise ValueError('RUN_MODE must be quick or complete')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(min(3, torch.get_num_threads()))

# Recent method families, with compact *task-trained* implementations, not
# copies of published pretrained spectral checkpoints. No SVM, RF, PLS-DA,
# PCA/ANOVA feature selection, or GAN baseline is included in comparisons.
CLASSIFIERS = ['Residual MLP', 'ConvNeXt1D', 'Patch Transformer', 'Spectral Mixer']
FEATURES = ['Raw SNV', 'Multi-resolution patches', 'Masked spectral AE',
            'Supervised contrastive']
GENERATORS = ['Latent Flow Matching', 'Latent DDPM', 'Latent EDM-style']
AUGMENTATIONS = ['Real only'] + [f'{m} {n}/class'
    for m in GENERATORS for n in AUG_COUNTS]
ANCHOR_MODEL = 'Residual MLP'
ANCHOR_FEATURE = 'Multi-resolution patches'
if INCLUDE_TABPFN:
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise ImportError('To enable optional pretrained TabPFN: pip install tabpfn; '
                          'its model checkpoint must also be accessible.') from exc
    CLASSIFIERS.append('TabPFN v2')

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'axes.grid': False, 'font.size': 10,
    'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
COLOR = {'main': '#254D77', 'secondary': '#0A7B83', 'positive': '#31887C',
         'negative': '#C46561', 'muted': '#75859A', 'light': '#E6EDF2'}


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snv(values):
    z = np.atleast_2d(np.asarray(values, dtype=np.float64))
    return (z-z.mean(axis=1, keepdims=True)) / np.maximum(z.std(axis=1,keepdims=True),1e-10)


def load_data():
    rows = []
    for folder, cls in [(HEALTHY_DIR, 0), (SICK_DIR, 1)]:
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing {folder}. Place one original-patient '
                                    'CSV in saudaveis/ or doentes/, or override ROOT.')
        files = sorted(p for p in folder.rglob('*.csv') if not any(
            q.startswith('.') or q.lower() == '__macosx'
            for q in p.relative_to(folder).parts))
        if not files:
            raise FileNotFoundError(f'No patient CSVs in {folder}')
        for path in files:
            first = path.read_text(encoding='utf-8-sig').splitlines()[0]
            sep = ';' if first.count(';') > first.count(',') else ','
            def numeric(s):
                try: float(s.strip().strip('"')); return True
                except ValueError: return False
            no_header = len(first.split(sep)) == 2 and all(
                numeric(s) for s in first.split(sep))
            data = pd.read_csv(path, sep=sep, encoding='utf-8-sig',
                               header=None if no_header else 0)
            if data.shape[1] != 2:
                raise ValueError(f'Expected two columns, wavenumber and intensity: {path}')
            wn = pd.to_numeric(data.iloc[:, 0], errors='raise').to_numpy(float)
            intensity = pd.to_numeric(data.iloc[:, 1], errors='raise').to_numpy(float)
            if len(wn) < 16 or not np.isfinite(wn).all() or not np.isfinite(intensity).all():
                raise ValueError(f'Invalid spectrum: {path}')
            patient_id = ('healthy/' if cls == 0 else 'sick/') + str(
                path.relative_to(folder).with_suffix('')).replace('/', '_')
            rows.append((patient_id, cls, wn, intensity))
    ref = rows[0][2]
    for name, _, wn, _ in rows:
        if len(wn) != len(ref) or not np.allclose(wn, ref, rtol=0, atol=1e-6):
            raise ValueError(f'Spectra need the same spectral grid: {name}')
    names = np.array([r[0] for r in rows])
    if len(names) != len(set(names)):
        raise ValueError('Patient identifiers must be unique')
    X = snv(np.stack([r[3] for r in rows]))
    y = np.array([r[1] for r in rows], dtype=int)
    if np.bincount(y, minlength=2).min() < 4:
        raise ValueError('Need >=4 original patients/class for nested validation')
    return ref, X, y, names


# ------------------------ MODERN LATENT AUGMENTATION ------------------------
class ConditionalDynamics(nn.Module):
    """Small class/time conditioned dynamics; training set is tiny by design."""
    def __init__(self, dim, width=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+7, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, dim))

    def forward(self, z, labels, t):
        t = t.reshape(-1, 1)
        phi = [t, torch.sin(2*np.pi*t), torch.cos(2*np.pi*t),
               torch.sin(4*np.pi*t), torch.cos(4*np.pi*t)]
        return self.net(torch.cat([z, F.one_hot(labels.long(), 2).float()]+phi, dim=1))


class LatentGenerator:
    """Latent generative method, PCA used only for the common latent interface.

    PCA is NOT a candidate feature-selection method. Its fit is contained in the
    training fold; all generators share this identical generative input space.
    """
    def __init__(self, method, seed):
        self.method = method
        self.seed = int(seed)
        self.fit_seconds = 0.0

    def fit(self, x_real, y_real):
        set_seed(self.seed)
        t0 = time.perf_counter()
        self.dim = min(5, len(x_real)-2, x_real.shape[1])
        self.pca = PCA(n_components=self.dim, random_state=self.seed)
        self.scale = StandardScaler()
        latent = self.scale.fit_transform(self.pca.fit_transform(snv(x_real)))
        Z = torch.tensor(latent, dtype=torch.float32, device=DEVICE)
        Y = torch.tensor(y_real, dtype=torch.long, device=DEVICE)
        pools = [torch.where(Y == c)[0] for c in [0, 1]]
        if any(len(a) < 2 for a in pools):
            raise ValueError('At least two original training patients per class required')
        self.net = ConditionalDynamics(self.dim).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(GENERATOR_STEPS):
            labels = torch.randint(0,2,(32,),device=DEVICE)
            index = torch.empty_like(labels)
            for cls in (0, 1):
                mask = labels == cls
                index[mask] = pools[cls][torch.randint(
                    len(pools[cls]),(int(mask.sum()),),device=DEVICE)]
            z = Z[index]
            if self.method == 'Latent Flow Matching':
                initial = torch.randn_like(z)
                time_j = torch.rand(len(z),1,device=DEVICE)
                noisy = (1-time_j)*initial + time_j*z
                loss = F.mse_loss(self.net(noisy,labels,time_j),z-initial)
            elif self.method == 'Latent DDPM':
                if not hasattr(self,'beta'):
                    self.beta = torch.linspace(1e-4, 0.2, DIFFUSION_STEPS, device=DEVICE)
                    self.alpha = 1-self.beta
                    self.alpha_bar = torch.cumprod(self.alpha, 0)
                k = torch.randint(DIFFUSION_STEPS,(len(z),),device=DEVICE)
                noise = torch.randn_like(z)
                noisy = self.alpha_bar[k,None].sqrt()*z + (1-self.alpha_bar[k,None]).sqrt()*noise
                loss = F.mse_loss(self.net(noisy,labels,
                    k.float().reshape(-1,1)/max(1,DIFFUSION_STEPS-1)),noise)
            elif self.method == 'Latent EDM-style':
                # Simplified variance-exploding denoising score objective;
                # NOT a full reproduction of Karras et al. EDM preconditioning.
                sigma = (torch.randn(len(z),1,device=DEVICE)*1.0-0.5).exp()
                noisy = z + sigma*torch.randn_like(z)
                prediction = self.net(noisy,labels,torch.log(sigma)/5.)
                loss = (((prediction-z)**2)/(sigma**2+0.5**2)).mean()
            else:
                raise ValueError(self.method)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0)
            opt.step()
        self.net.eval()
        self.params = sum(p.numel() for p in self.net.parameters())
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        self.fit_seconds = time.perf_counter()-t0
        return self

    def sample(self, count):
        set_seed(self.seed+991+int(count))
        y = np.repeat([0,1],int(count))
        labels = torch.tensor(y,dtype=torch.long,device=DEVICE)
        with torch.no_grad():
            z = torch.randn(len(y),self.dim,device=DEVICE)
            if self.method == 'Latent Flow Matching':
                dt = 1.0/ODE_STEPS
                for j in range(ODE_STEPS):
                    a=torch.full((len(y),1),j*dt,device=DEVICE)
                    b=torch.full((len(y),1),(j+1)*dt,device=DEVICE)
                    v=self.net(z,labels,a)
                    z=z+dt*.5*(v+self.net(z+dt*v,labels,b))
            elif self.method == 'Latent DDPM':
                for k in range(DIFFUSION_STEPS-1,-1,-1):
                    t=torch.full((len(y),1),k/max(1,DIFFUSION_STEPS-1),device=DEVICE)
                    eps=self.net(z,labels,t)
                    mean=(z-self.beta[k]*eps/(1-self.alpha_bar[k]).sqrt())/self.alpha[k].sqrt()
                    if k > 0:
                        var=self.beta[k]*(1-self.alpha_bar[k-1])/(1-self.alpha_bar[k])
                        z=mean+var.clamp_min(0).sqrt()*torch.randn_like(z)
                    else: z=mean
            elif self.method == 'Latent EDM-style':
                sigma=torch.linspace(3.0,.015,ODE_STEPS,device=DEVICE)
                z=z*sigma[0]
                for i in range(ODE_STEPS):
                    s=sigma[i]
                    nex=sigma[i+1] if i+1 < ODE_STEPS else torch.tensor(0.,device=DEVICE)
                    log_t=torch.full((len(y),1),float(torch.log(s))/5.,device=DEVICE)
                    den=self.net(z,labels,log_t)
                    direction=(z-den)/s
                    euler=z+(nex-s)*direction
                    if i+1 < ODE_STEPS:
                        next_t=torch.full((len(y),1),float(torch.log(nex))/5.,device=DEVICE)
                        den2=self.net(euler,labels,next_t)
                        direction2=(euler-den2)/nex
                        z=z+(nex-s)*.5*(direction+direction2)
                    else: z=euler
            lat=z.cpu().numpy().clip(-8,8)
            x=snv(self.pca.inverse_transform(self.scale.inverse_transform(lat)))
        if not np.isfinite(x).all():
            raise FloatingPointError(f'{self.method} generated nonfinite values')
        return x,y


def split_augmentation(name):
    if name == 'Real only': return 'Real only',0
    method, n = name.rsplit(' ',1)
    return method, int(n.split('/')[0])


def new_record(x_train,y_train,x_test,seed):
    return dict(xtr=x_train,ytr=y_train,xte=x_test,seed=int(seed),
        generators={}, generated={}, transforms={}, models={})


def get_augmented(record,name):
    if name == 'Real only':
        return record['xtr'],record['ytr'],dict(generator_s=0.,sample_s=0.,params=0)
    method,count=split_augmentation(name)
    if method not in record['generators']:
        record['generators'][method]=LatentGenerator(method,record['seed']).fit(
            record['xtr'],record['ytr'])
    gen=record['generators'][method]
    key=(method,count)
    if key not in record['generated']:
        t=time.perf_counter()
        xx,yy=gen.sample(count)
        record['generated'][key]=(xx,yy,time.perf_counter()-t)
    x,y,sample_s=record['generated'][key]
    return (np.vstack((record['xtr'],x)),np.r_[record['ytr'],y],
            dict(generator_s=gen.fit_seconds,sample_s=sample_s,params=gen.params))


# ---------------------------- SPECTRAL FEATURES ----------------------------
def multires(x):
    z=torch.as_tensor(np.asarray(x,dtype=np.float32))[:,None,:]
    with torch.no_grad():
        return torch.cat([F.adaptive_avg_pool1d(z,k).flatten(1)
                          for k in (32,64,128)],dim=1).cpu().numpy().astype(float)


class SpectralEncoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(n_features,128),nn.GELU(),
            nn.LayerNorm(128),nn.Linear(128,64))
        self.decoder=nn.Sequential(nn.Linear(64,128),nn.GELU(),
            nn.Linear(128,n_features))

    def forward(self, x): return self.encoder(x)


def weak_spectral_augment(x):
    n,p=x.shape
    coords=torch.linspace(0,2*np.pi,p,device=x.device)[None,:]
    waviness=.025*torch.randn(n,1,device=x.device)*torch.sin(coords*3)
    return x*(1+.04*torch.randn(n,1,device=x.device)) + \
        waviness + .04*torch.randn_like(x)


class SpectralFeatures:
    """Each trainable map is fitted on current *inner training* data only."""
    def __init__(self,name,seed):
        self.name=name
        self.seed=int(seed)
        self.fit_seconds=0.

    def fit(self,X,y):
        t=time.perf_counter()
        if self.name in ('Raw SNV','Multi-resolution patches'):
            self.fit_seconds=0.
            return self
        set_seed(self.seed)
        Z=torch.tensor(snv(X),device=DEVICE,dtype=torch.float32)
        Y=torch.tensor(y,device=DEVICE,dtype=torch.long)
        net=SpectralEncoder(Z.shape[1]).to(DEVICE)
        opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=5e-4)
        net.train()
        for _ in range(FEATURE_EPOCHS):
            if self.name == 'Masked spectral AE':
                # Random mask and masked reconstruction on training spectra.
                mask=(torch.rand_like(Z)<.3).float()
                pred=net.decoder(net.encoder(Z*(1-mask)))
                loss=((pred-Z)**2*mask).sum()/mask.sum().clamp_min(1)
                loss=loss+.05*F.mse_loss(pred,Z)
            elif self.name == 'Supervised contrastive':
                # Class labels include TRAIN-fold-generated spectra only.
                views=torch.cat([weak_spectral_augment(Z),weak_spectral_augment(Z)])
                targets=torch.cat([Y,Y]); emb=F.normalize(net(views),dim=1)
                similarity=emb@emb.T/.15
                eye=torch.eye(len(targets),dtype=torch.bool,device=DEVICE)
                positive=(targets[:,None]==targets[None,:]) & (~eye)
                similarity=similarity.masked_fill(eye,-1e9)
                log_prob=similarity-torch.logsumexp(similarity,dim=1,keepdim=True)
                loss=-(log_prob*positive).sum(dim=1)/positive.sum(dim=1).clamp_min(1)
                loss=loss.mean()
            else: raise KeyError(self.name)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
        net.eval()
        self.net=net
        if DEVICE.type=='cuda': torch.cuda.synchronize()
        self.fit_seconds=time.perf_counter()-t
        return self

    def transform(self,X):
        if self.name == 'Raw SNV': return snv(X).astype(np.float32)
        if self.name == 'Multi-resolution patches': return multires(X).astype(np.float32)
        with torch.no_grad():
            z=torch.as_tensor(snv(X),device=DEVICE,dtype=torch.float32)
            return self.net(z).cpu().numpy()


def get_features(record,augment,feature):
    key=(augment,feature)
    if key not in record['transforms']:
        x,y,_=get_augmented(record,augment)
        record['transforms'][key]=SpectralFeatures(feature,record['seed']+43).fit(x,y)
    return record['transforms'][key]


# ------------------------------ CLASSIFIERS --------------------------------
class ResidualMLP(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.input=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,64),nn.GELU())
        self.residual=nn.Sequential(nn.Linear(64,64),nn.GELU(),
                                   nn.Dropout(.20),nn.Linear(64,64))
        self.head=nn.Linear(64,1)

    def forward(self,x):
        z=self.input(x)
        return self.head(F.gelu(z+self.residual(z))).flatten()


class ConvNeXt1D(nn.Module):
    """Compact ConvNeXt-style depthwise separable spectral CNN."""
    def __init__(self,d):
        super().__init__()
        self.stem=nn.Conv1d(1,24,kernel_size=7,stride=2,padding=3)
        self.dw=nn.Conv1d(24,24,kernel_size=7,padding=3,groups=24)
        self.norm=nn.GroupNorm(4,24)
        self.pw=nn.Sequential(nn.Conv1d(24,48,1),nn.GELU(),
                              nn.Conv1d(48,24,1))
        self.out=nn.Sequential(nn.AdaptiveAvgPool1d(4),nn.Flatten(),
                               nn.Linear(96,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,x):
        x=F.adaptive_avg_pool1d(x[:,None,:],128)
        z=self.stem(x)
        return self.out(z+self.pw(self.norm(self.dw(z)))).flatten()


class PatchTransformer(nn.Module):
    """Compact position-encoded patch attention, not pretrained PatchTST."""
    def __init__(self,d):
        super().__init__()
        self.patch=nn.Conv1d(1,32,kernel_size=4,stride=4)
        self.pos=nn.Parameter(torch.randn(1,16,32)*.015)
        encoder=nn.TransformerEncoderLayer(d_model=32,nhead=4,
            dim_feedforward=64,dropout=.1,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(encoder,num_layers=1,
                                          enable_nested_tensor=False)
        self.head=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,1))

    def forward(self,x):
        x=F.adaptive_avg_pool1d(x[:,None,:],64)
        tokens=self.patch(x).transpose(1,2)+self.pos
        return self.head(self.encoder(tokens).mean(dim=1)).flatten()


class MixerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.tnorm=nn.LayerNorm(32)
        self.tokens=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16))
        self.cnorm=nn.LayerNorm(32)
        self.channels=nn.Sequential(nn.Linear(32,64),nn.GELU(),
                                     nn.Linear(64,32))

    def forward(self,x):
        x=x+self.tokens(self.tnorm(x).transpose(1,2)).transpose(1,2)
        return x+self.channels(self.cnorm(x))


class SpectralMixer(nn.Module):
    """Compact MLP-Mixer spectral patch model."""
    def __init__(self,d):
        super().__init__()
        self.patch=nn.Conv1d(1,32,kernel_size=4,stride=4)
        self.blocks=nn.Sequential(MixerBlock(),MixerBlock())
        self.head=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,1))

    def forward(self,x):
        x=F.adaptive_avg_pool1d(x[:,None,:],64)
        return self.head(self.blocks(self.patch(x).transpose(1,2)).mean(dim=1)).flatten()


def make_classifier(name,d):
    return {'Residual MLP':ResidualMLP, 'ConvNeXt1D':ConvNeXt1D,
            'Patch Transformer':PatchTransformer,'Spectral Mixer':SpectralMixer}[name](d)


def classifier_predict(name,xtr,ytr,xte,seed):
    """Return test margins and cold training/prediction timing. No test labels."""
    set_seed(seed)
    begin=time.perf_counter()
    if name == 'TabPFN v2':
        # TabPFN v2 has a documented 500-feature limit; reduce raw spectra by
        # unsupervised fixed pooling if needed, and record this in metadata.
        if xtr.shape[1] > 500:
            xtr=F.adaptive_avg_pool1d(torch.tensor(xtr,dtype=torch.float32)[:,None,:],
                 256).flatten(1).numpy()
            xte=F.adaptive_avg_pool1d(torch.tensor(xte,dtype=torch.float32)[:,None,:],
                 256).flatten(1).numpy()
        clf=TabPFNClassifier(device=str(DEVICE))
        clf.fit(xtr,ytr)
        fit_s=time.perf_counter()-begin
        z=clf.predict_proba(xte)[:,1]
        score=np.log(np.clip(z,1e-6,1-1e-6)/np.clip(1-z,1e-6,1))
    else:
        model=make_classifier(name,xtr.shape[1]).to(DEVICE)
        x=torch.tensor(xtr,device=DEVICE,dtype=torch.float32)
        y=torch.tensor(ytr,device=DEVICE,dtype=torch.float32)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=.02)
        counts=np.bincount(np.asarray(ytr,int),minlength=2)
        pos_weight=torch.tensor(float(counts[0]/max(1,counts[1])),device=DEVICE)
        model.train()
        for _ in range(CLASSIFIER_EPOCHS):
            logits=model(x)
            loss=F.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        model.eval()
        if DEVICE.type=='cuda':torch.cuda.synchronize()
        fit_s=time.perf_counter()-begin
        with torch.no_grad():
            z=torch.tensor(xte,device=DEVICE,dtype=torch.float32)
            score=model(z).cpu().numpy()
    if DEVICE.type=='cuda':torch.cuda.synchronize()
    return np.asarray(score,float).reshape(-1),fit_s,time.perf_counter()-begin-fit_s


def evaluate(record,model,aug,feature):
    key=(model,aug,feature)
    if key in record['models']: return record['models'][key]
    x,y,aug_meta=get_augmented(record,aug)
    projector=get_features(record,aug,feature)
    x_train=projector.transform(x)
    x_test=projector.transform(record['xte'])
    score,fit_s,pred_s=classifier_predict(model,x_train,y,x_test,record['seed']+101)
    effort=dict(train_s=float(fit_s+projector.fit_seconds+aug_meta['generator_s']+
         aug_meta['sample_s']),classifier_s=float(fit_s),
         feature_s=float(projector.fit_seconds),generator_s=float(aug_meta['generator_s']),
         sample_s=float(aug_meta['sample_s']),predict_s=float(pred_s),
         features=int(x_train.shape[1]),train_samples=int(len(y)),
         classifier_params=(0 if model=='TabPFN v2' else int(sum(
             p.numel() for p in make_classifier(model,x_train.shape[1]).parameters()))))
    record['models'][key]=(score,effort)
    return score,effort

# --------------------------- NESTED EVALUATION -----------------------------
def joint_name(model,feature):
    return model+' || '+feature


def split_joint(name):
    return name.split(' || ',1)


def prediction_metrics(y_true,scores):
    y=np.asarray(y_true,int)
    z=np.asarray(scores,float)
    pred=(z>=0).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(N=int(len(y)),Correct=int((y==pred).sum()),
        Accuracy=float(accuracy_score(y,pred)),
        BA=float(balanced_accuracy_score(y,pred)),
        Sensitivity=float(tp/max(1,tp+fn)),
        Specificity=float(tn/max(1,tn+fp)),
        AUC=float(roc_auc_score(y,z)) if len(np.unique(y))==2 else np.nan,
        AP=float(average_precision_score(y,z)) if len(np.unique(y))==2 else np.nan,
        F1=float(f1_score(y,pred,zero_division=0)),
        MCC=float(matthews_corrcoef(y,pred)),
        TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn))


def choose_candidate(rows,names):
    """Rank only original-patient inner validation; no outer-test peeking."""
    frame=pd.DataFrame(rows)
    scores={}
    for candidate in names:
        sub=frame[frame.candidate==candidate]
        if len(sub)==0: raise RuntimeError('Missing inner candidate '+candidate)
        scores[candidate]=prediction_metrics(sub.y_true,sub.score)
    best=max(names,key=lambda c:(scores[c]['BA'],scores[c]['AUC'],
                                   scores[c]['Accuracy'],-names.index(c)))
    return best,scores


def quality_scores(record,augmentation):
    """Train-patient similarity checks; NOT a clinical validity criterion."""
    if augmentation=='Real only': return None
    method,count=split_augmentation(augmentation)
    x,y,_=record['generated'][(method,count)]
    vals=[]
    for cl in (0,1):
        real=record['xtr'][record['ytr']==cl]
        syn=x[y==cl]
        if len(real)<2 or len(syn)<2:continue
        rr=cdist(real,real)/np.sqrt(real.shape[1])
        np.fill_diagonal(rr,np.inf)
        typical=np.median(rr.min(axis=1))
        proximity=np.median(cdist(syn,real).min(axis=1)/np.sqrt(real.shape[1]))
        real_spread=np.median(pdist(real))/np.sqrt(real.shape[1])
        syn_spread=np.median(pdist(syn))/np.sqrt(real.shape[1])
        vals.append((proximity/max(typical,1e-9),
                     syn_spread/max(real_spread,1e-9)))
    return tuple(np.mean(v) for v in zip(*vals)) if vals else (np.nan,np.nan)


def make_inner_records(X,y,outer_train,outer_fold):
    k=min(INNER_SPLITS,int(np.bincount(y[outer_train]).min()))
    if k<2:raise ValueError('Need at least two inner folds')
    cv=StratifiedKFold(n_splits=k,shuffle=True,random_state=SEED+outer_fold)
    records=[]
    for j,(tr,va) in enumerate(cv.split(X[outer_train],y[outer_train])):
        real_tr=outer_train[tr]; real_va=outer_train[va]
        record=new_record(X[real_tr],y[real_tr],X[real_va],
            seed=SEED+outer_fold*100003+(j+1)*101)
        record['fold']=j
        record['validation_ids']=real_va
        records.append(record)
    return records


def search_stage(records,candidates,stage,selected_aug=None):
    rows=[]
    for rec in records:
        truth=rec['validation_ids']
        for name in candidates:
            if stage=='augmentation':
                model,aug,feature=ANCHOR_MODEL,name,ANCHOR_FEATURE
            else:
                model,feature=split_joint(name)
                aug=selected_aug
            prediction,effort=evaluate(rec,model,aug,feature)
            for patient_index,score in zip(truth,prediction):
                rows.append(dict(stage=stage,candidate=name,
                    inner_fold=rec['fold'],original_index=int(patient_index),
                    score=float(score),**effort))
    return pd.DataFrame(rows)


def set_validation_labels(data,y):
    data['y_true']=y[data.original_index.to_numpy(int)]
    data['y_pred']=(data.score>=0).astype(int)
    return data


def summary_validation(selection_data,selections,stage,order):
    sub=selection_data[selection_data.stage==stage]
    out=sub.groupby(['outer_fold','candidate'],sort=False).agg(
        BA=('BA','first'),AUC=('AUC','first'),
        Accuracy=('Accuracy','first'),TrainSeconds=('train_s','first'),
        FeatureSeconds=('feature_s','first'),ClassifierSeconds=('classifier_s','first'),
        GeneratorSeconds=('generator_s','first'),FeatureDim=('features','first')).reset_index()
    grouped=out.groupby('candidate',sort=False).agg(
        InnerBA=('BA','mean'),InnerBA_SD=('BA','std'),
        InnerAUC=('AUC','mean'),InnerACC=('Accuracy','mean'),
        MeanTrainSeconds=('TrainSeconds','mean'),
        MeanFeatureSeconds=('FeatureSeconds','mean'),
        MeanClassifierSeconds=('ClassifierSeconds','mean'),
        MeanGeneratorSeconds=('GeneratorSeconds','mean'),
        MeanDim=('FeatureDim','mean'),NOuter=('outer_fold','nunique')).reset_index()
    counts=selections.loc[selections.stage==stage,'candidate'].value_counts()
    grouped['Selections']=grouped.candidate.map(counts).fillna(0).astype(int)
    grouped['order']=grouped.candidate.map({name:j for j,name in enumerate(order)})
    return grouped.sort_values(['Selections','InnerBA','InnerAUC','order'],
        ascending=[False,False,False,True]).drop(columns='order').reset_index(drop=True)


def selected_pipeline_scores(pred):
    out=[]
    for arm,group in pred.groupby('arm',sort=False):
        group=group.sort_values('outer_fold')
        out.append(dict(arm=arm,**prediction_metrics(group.y_true,group.score)))
    return pd.DataFrame(out).sort_values('arm').reset_index(drop=True)


def paired_table(pred,reference,comparators,seed=SEED):
    """Patient-matched BA deltas, exploratory stratified bootstrap intervals."""
    ref=pred[pred.arm==reference].sort_values('outer_fold').reset_index(drop=True)
    y=ref.y_true.to_numpy(int)
    a=(ref.score.to_numpy(float)>=0).astype(int)
    negative=np.flatnonzero(y==0);positive=np.flatnonzero(y==1)
    rows=[]
    for j,name in enumerate(comparators):
        other=pred[pred.arm==name].sort_values('outer_fold').reset_index(drop=True)
        if len(other)!=len(ref) or other.patient.to_list()!=ref.patient.to_list():
            raise RuntimeError('Patient matching failed for '+name)
        b=(other.score.to_numpy(float)>=0).astype(int)
        rng=np.random.default_rng(seed+j*199+91)
        boot_n=rng.choice(negative,(BOOTSTRAPS,len(negative)),replace=True)
        boot_p=rng.choice(positive,(BOOTSTRAPS,len(positive)),replace=True)
        delta=50*((a[boot_n]==y[boot_n]).mean(axis=1)-
                  (b[boot_n]==y[boot_n]).mean(axis=1)+
                  (a[boot_p]==y[boot_p]).mean(axis=1)-
                  (b[boot_p]==y[boot_p]).mean(axis=1))
        recovered=int(((a==y)&(b!=y)).sum())
        lost=int(((a!=y)&(b==y)).sum())
        pval=binomtest(recovered,recovered+lost,.5).pvalue if recovered+lost else 1.
        rows.append(dict(Reference=reference,Comparator=name,
            DeltaBA_pp=100*(balanced_accuracy_score(y,a)-balanced_accuracy_score(y,b)),
            CI_low_pp=float(np.quantile(delta,.025)),
            CI_high_pp=float(np.quantile(delta,.975)),
            Recovered=recovered,Lost=lost,ExactMcNemarP=float(pval)))
    frame=pd.DataFrame(rows)
    if len(frame):
        p=frame.ExactMcNemarP.to_numpy()
        ix=np.argsort(p)
        adjusted=np.empty(len(p))
        adjusted[ix]=np.minimum(1.,np.maximum.accumulate(p[ix]*(len(p)-np.arange(len(p)))))
        frame['HolmP']=adjusted if RUN_MODE=='complete' else np.nan
    return frame


def add_outer_arm(record,patient,fold,truth,arm,model,augment,feature,
                  predictions,efforts,used):
    key=(model,augment,feature)
    if key not in used:
        score,cost=evaluate(record,model,augment,feature)
        used[key]=(float(score[0]),cost)
    score,cost=used[key]
    predictions.append(dict(outer_fold=fold,patient=patient,arm=arm,
        model=model,augmentation=augment,feature=feature,
        y_true=int(truth),score=score,y_pred=int(score>=0)))
    efforts.append(dict(outer_fold=fold,arm=arm,model=model,
        augmentation=augment,feature=feature,**cost))


def run_experiment():
    start=time.perf_counter()
    set_seed(SEED)
    wn,X,y,names=load_data()
    if len(AUG_COUNTS)<1 or tuple(sorted(set(AUG_COUNTS)))!=AUG_COUNTS:
        raise ValueError('AUG_COUNTS must be sorted unique positive integers')
    if RUN_MODE=='quick':
        rng=np.random.default_rng(SEED)
        indices = np.concatenate([
            rng.choice(
                np.flatnonzero(y == cl),
                size=min(OUTER_QUICK_PER_CLASS, int((y == cl).sum())),
                replace=False,
            )
            for cl in (0, 1)
        ])
    else:
        indices=np.arange(len(y))
    permitted=set(int(v) for v in indices)
    joint_grid=[joint_name(m,f) for m in CLASSIFIERS for f in FEATURES]
    val_predictions=[];validation_summaries=[];select_rows=[]
    outer_predictions=[];outer_efforts=[];quality_rows=[]
    step_times={'augmentation':0.,'joint':0.,'outer':0.}

    for outer_fold,(outer_tr,outer_te) in enumerate(LeaveOneOut().split(X),1):
        held=int(outer_te[0])
        if held not in permitted: continue
        train_records=make_inner_records(X,y,outer_tr,outer_fold)
        outer_record=new_record(X[outer_tr],y[outer_tr],X[held:held+1],
                                SEED+outer_fold*100003+772)
        patient=names[held]

        # STAGE 1 -- only the fixed, preregistered modern anchor evaluates
        # augmentation options. No classifier/feature search occurs yet.
        t=time.perf_counter()
        aug_inner=set_validation_labels(search_stage(train_records,AUGMENTATIONS,
                                        'augmentation'),y)
        selected_aug,aug_scores=choose_candidate(aug_inner.to_dict('records'),AUGMENTATIONS)
        synthetic=[a for a in AUGMENTATIONS if a!='Real only']
        forced_synthetic,_=choose_candidate(aug_inner.to_dict('records'),synthetic)
        step_times['augmentation']+=time.perf_counter()-t
        select_rows.extend([
            dict(outer_fold=outer_fold,patient=patient,stage='augmentation',
                 candidate=selected_aug,InnerBA=aug_scores[selected_aug]['BA']),
            dict(outer_fold=outer_fold,patient=patient,stage='best_synthetic',
                 candidate=forced_synthetic,InnerBA=aug_scores[forced_synthetic]['BA'])])
        for candidate,result in aug_scores.items():
            z=aug_inner[aug_inner.candidate==candidate]
            validation_summaries.append(dict(outer_fold=outer_fold,stage='augmentation',
                candidate=candidate,**result,
                **{k:float(z[k].mean()) for k in
                   ['train_s','classifier_s','feature_s','generator_s','features']}))
        aug_inner['outer_fold']=outer_fold
        val_predictions.append(aug_inner)

        # STAGE 2 -- jointly search neural classifier and spectral feature
        # engineering using ONLY the stage-1-selected augmentation.
        t=time.perf_counter()
        joint_inner=set_validation_labels(search_stage(train_records,joint_grid,
            'joint',selected_aug=selected_aug),y)
        winning_pair,pair_scores=choose_candidate(joint_inner.to_dict('records'),joint_grid)
        model,feature=split_joint(winning_pair)
        step_times['joint']+=time.perf_counter()-t
        select_rows.append(dict(outer_fold=outer_fold,patient=patient,
            stage='joint',candidate=winning_pair,InnerBA=pair_scores[winning_pair]['BA']))
        for candidate,result in pair_scores.items():
            z=joint_inner[joint_inner.candidate==candidate]
            validation_summaries.append(dict(outer_fold=outer_fold,stage='joint',
                candidate=candidate,**result,
                **{k:float(z[k].mean()) for k in
                   ['train_s','classifier_s','feature_s','generator_s','features']}))
        joint_inner['outer_fold']=outer_fold
        val_predictions.append(joint_inner)

        # STAGE 3 -- held-out original patient, without outer-score-based tuning.
        t=time.perf_counter()
        used={}
        add_outer_arm(outer_record,patient,outer_fold,y[held],'selected_pipeline',
            model,selected_aug,feature,outer_predictions,outer_efforts,used)
        add_outer_arm(outer_record,patient,outer_fold,y[held],'matched_real_only',
            model,'Real only',feature,outer_predictions,outer_efforts,used)
        add_outer_arm(outer_record,patient,outer_fold,y[held],'forced_synthetic',
            model,forced_synthetic,feature,outer_predictions,outer_efforts,used)

        # All outer augmentation methods use the SAME *fixed* anchor.
        # Otherwise, changes could be caused by retuned architectures.
        for aug in AUGMENTATIONS:
            add_outer_arm(outer_record,patient,outer_fold,y[held],
                'aug_anchor::'+aug,ANCHOR_MODEL,aug,ANCHOR_FEATURE,
                outer_predictions,outer_efforts,used)
            if aug!='Real only':
                proximity,diversity=quality_scores(outer_record,aug)
                quality_rows.append(dict(outer_fold=outer_fold,patient=patient,
                    augmentation=aug,NNProximityRatio=proximity,
                    DiversityRatio=diversity))

        # Component ablation: ONLY ONE component changes at a time, while
        # augmentation and other selected component remain fixed.
        for alternative in CLASSIFIERS:
            add_outer_arm(outer_record,patient,outer_fold,y[held],
                'replace_classifier::'+alternative,alternative,selected_aug,feature,
                outer_predictions,outer_efforts,used)
        for alternative in FEATURES:
            add_outer_arm(outer_record,patient,outer_fold,y[held],
                'replace_feature::'+alternative,model,selected_aug,alternative,
                outer_predictions,outer_efforts,used)
        # Full-pipeline no-augmentation/no-feature-engineering controls.
        # Each contemporary classifier gets the same raw SNV data.
        for alternative in CLASSIFIERS:
            add_outer_arm(outer_record,patient,outer_fold,y[held],
                'raw_real_only::'+alternative,alternative,'Real only','Raw SNV',
                outer_predictions,outer_efforts,used)
        step_times['outer']+=time.perf_counter()-t
        print(f'{RUN_MODE} {len(set(r["outer_fold"] for r in select_rows))}/'
            f'{len(indices)} | {patient} | augment: {selected_aug} | '
            f'{model} + {feature}',flush=True)

    val_pred=pd.concat(val_predictions,ignore_index=True)
    val_sum=pd.DataFrame(validation_summaries)
    selected=pd.DataFrame(select_rows)
    pred=pd.DataFrame(outer_predictions)
    effort=pd.DataFrame(outer_efforts)
    quality=pd.DataFrame(quality_rows)
    aug_summary=summary_validation(val_sum,selected,'augmentation',AUGMENTATIONS)
    joint_summary=summary_validation(val_sum,selected,'joint',joint_grid)
    metrics=selected_pipeline_scores(pred)
    selected_comparators=['matched_real_only','forced_synthetic'] + [
        'replace_classifier::'+m for m in CLASSIFIERS] + [
        'replace_feature::'+f for f in FEATURES] + [
        'raw_real_only::'+m for m in CLASSIFIERS]
    selected_pairs=paired_table(pred,'selected_pipeline',selected_comparators)
    anchor_pairs=paired_table(pred,'aug_anchor::Real only',[
        'aug_anchor::'+a for a in AUGMENTATIONS if a!='Real only'],seed=SEED+999)
    # This table's sign is "real-only minus augmentation"; reverse for plotting.
    anchor_pairs['AugmentationMinusReal_pp']=-anchor_pairs.DeltaBA_pp
    anchor_pairs['AugmentationCI_low_pp']=-anchor_pairs.CI_high_pp
    anchor_pairs['AugmentationCI_high_pp']=-anchor_pairs.CI_low_pp

    val_pred.to_csv(OUTPUT_DIR/'inner_patient_predictions.csv',index=False)
    val_sum.to_csv(OUTPUT_DIR/'inner_validation_by_outer_fold.csv',index=False)
    selected.to_csv(OUTPUT_DIR/'selected_configurations_by_outer_fold.csv',index=False)
    pred.to_csv(OUTPUT_DIR/'outer_patient_predictions.csv',index=False)
    effort.to_csv(OUTPUT_DIR/'outer_pipeline_effort.csv',index=False)
    quality.to_csv(OUTPUT_DIR/'synthetic_quality_train_only.csv',index=False)
    aug_summary.to_csv(OUTPUT_DIR/'augmentation_validation_summary.csv',index=False)
    joint_summary.to_csv(OUTPUT_DIR/'joint_ablation_validation_summary.csv',index=False)
    metrics.to_csv(OUTPUT_DIR/'outer_benchmark_metrics.csv',index=False)
    selected_pairs.to_csv(OUTPUT_DIR/'paired_selected_pipeline.csv',index=False)
    anchor_pairs.to_csv(OUTPUT_DIR/'paired_augmentation_fixed_anchor.csv',index=False)

    plot_augmentation_selection(aug_summary,selected)
    plot_augmentation_effects(anchor_pairs)
    plot_joint_heatmap(joint_summary)
    plot_component_effects(selected_pairs)
    plot_patient_transitions(pred)
    plot_roc_pr(pred)
    plot_accuracy_time(metrics,effort)
    plot_synthetic_quality(quality)
    plot_modern_raw_controls(selected_pairs)
    write_reports(aug_summary,joint_summary,metrics,selected_pairs,anchor_pairs,
                  quality,selected,X,y,wn,time.perf_counter()-start,step_times)
    print(f'Finished: {len(indices)} held-out ORIGINAL patients. Output: {OUTPUT_DIR}',
          flush=True)
    return pred,metrics,selected,val_sum,selected_pairs


# ----------------------------- PAPER FIGURES --------------------------------
def tidy(ax):
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR['light'])
    ax.spines['bottom'].set_color(COLOR['light'])
    ax.tick_params(length=0,pad=5)


def save_fig(fig,filename):
    fig.savefig(OUTPUT_DIR/filename,bbox_inches='tight',pad_inches=.09)
    plt.close(fig)


def plot_augmentation_selection(summ,sel):
    """Shows inner score variability and how often inner tuning picked an arm."""
    rows=summ.set_index('candidate').loc[AUGMENTATIONS].reset_index()
    yy=np.arange(len(rows))
    fig,ax=plt.subplots(figsize=(8.8,max(3.5,.49*len(rows)+1)))
    score=rows.InnerBA.to_numpy()*100
    deviation=rows.InnerBA_SD.fillna(0).to_numpy()*100
    ax.errorbar(score,yy,xerr=deviation,fmt='o',capsize=3,lw=1.5,
                color=COLOR['main'],ecolor=COLOR['muted'],label='Mean ± SD across outer folds')
    for j,(value,count) in enumerate(zip(score,rows.Selections)):
        ax.text(min(value+deviation[j]+1.2,100),j,
                f'{value:.1f}% · {count} selections',fontsize=8,va='center')
    ax.set_xlim(max(0,float(np.nanmin(score-deviation))-12),116)
    ax.set_ylim(len(rows)-.5,-.5)
    ax.set_yticks(yy);ax.set_yticklabels(rows.candidate,fontsize=9)
    ax.set_xlabel('Inner-validation balanced accuracy (%)')
    ax.legend(frameon=False,loc='lower right',fontsize=8)
    tidy(ax);fig.tight_layout()
    save_fig(fig,'01_augmentation_inner_validation.pdf')


def plot_augmentation_effects(paired):
    """Original held-out patient effect with the fixed anchor and 95% CIs."""
    d=paired.copy().sort_values('AugmentationMinusReal_pp')
    y=np.arange(len(d))
    avg=d.AugmentationMinusReal_pp.to_numpy()
    lo=d.AugmentationCI_low_pp.to_numpy()
    hi=d.AugmentationCI_high_pp.to_numpy()
    fig,ax=plt.subplots(figsize=(9.3,max(3.5,.53*len(d)+1)))
    ax.axvline(0,color=COLOR['muted'],ls='--',lw=1)
    for j in range(len(d)):
        col=COLOR['positive'] if avg[j]>=0 else COLOR['negative']
        ax.plot([lo[j],hi[j]],[j,j],color=col,lw=2)
        ax.plot(avg[j],j,'o',color=col,markersize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(d.Comparator.str.replace('aug_anchor::','',regex=False),fontsize=9)
    ax.set_xlabel('Augmentation − real only: Δ balanced accuracy (pp), 95% bootstrap CI')
    ax.set_ylim(-.5,len(d)-.5)
    tidy(ax);fig.tight_layout()
    save_fig(fig,'02_augmentation_paired_heldout.pdf')


def plot_joint_heatmap(summary):
    vals=summary.set_index('candidate')
    ba=np.array([[100*vals.loc[joint_name(m,f),'InnerBA'] for f in FEATURES]
                  for m in CLASSIFIERS])
    counts=np.array([[int(vals.loc[joint_name(m,f),'Selections']) for f in FEATURES]
                     for m in CLASSIFIERS])
    fig,ax=plt.subplots(figsize=(10,max(3.5,.68*len(CLASSIFIERS)+1.8)))
    im=ax.imshow(ba,aspect='auto',cmap='Blues',vmin=max(0,ba.min()-7),vmax=100)
    for i in range(len(CLASSIFIERS)):
        for j in range(len(FEATURES)):
            ax.text(j,i,f'{ba[i,j]:.1f}%\n({counts[i,j]} selected)',
                ha='center',va='center',fontsize=9,
                color='white' if ba[i,j]>=(ba.min()+100)/2 else '#1E334B')
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_xticklabels(FEATURES,rotation=18,ha='right',fontsize=9)
    ax.set_yticks(np.arange(len(CLASSIFIERS)))
    ax.set_yticklabels(CLASSIFIERS,fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    cb=fig.colorbar(im,ax=ax,shrink=.76,pad=.02)
    cb.set_label('Inner-validation balanced accuracy (%)')
    fig.tight_layout()
    save_fig(fig,'03_joint_classifier_feature_heatmap.pdf')


def plot_component_effects(pairs):
    d=pairs[pairs.Comparator.str.startswith((
        'replace_classifier::','replace_feature::'))].copy()
    d['Group']=np.where(d.Comparator.str.startswith('replace_classifier::'),
                        'Classifier replacement','Feature replacement')
    d['Label']=d.Comparator.str.replace('replace_classifier::','',regex=False).str.replace(
        'replace_feature::','',regex=False)
    d=d.sort_values(['Group','DeltaBA_pp']).reset_index(drop=True)
    yy=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(9.4,max(4.,.53*len(d)+1.2)))
    ax.axvline(0,color=COLOR['muted'],ls='--',lw=1)
    for i,r in d.iterrows():
        col=COLOR['main'] if r.Group=='Classifier replacement' else COLOR['secondary']
        ax.plot([r.CI_low_pp,r.CI_high_pp],[i,i],color=col,lw=2)
        ax.plot(r.DeltaBA_pp,i,'o',color=col,ms=6)
    ax.set_yticks(yy)
    ax.set_yticklabels([f'{r.Label} ({"model" if r.Group.startswith("Classifier") else "feature"})'
                        for _,r in d.iterrows()],fontsize=9)
    ax.set_xlabel('Selected pipeline − component replacement: ΔBA (pp), 95% CI')
    ax.set_ylim(-.5,len(d)-.5)
    tidy(ax);fig.tight_layout()
    save_fig(fig,'04_component_ablation_paired_effects.pdf')


def plot_patient_transitions(pred):
    selected=pred[pred.arm=='selected_pipeline'].sort_values('outer_fold').reset_index(drop=True)
    real=pred[pred.arm=='matched_real_only'].sort_values('outer_fold').reset_index(drop=True)
    assert selected.patient.to_list()==real.patient.to_list()
    truth=selected.y_true.to_numpy(int)
    a=selected.y_pred.to_numpy(int)==truth
    b=real.y_pred.to_numpy(int)==truth
    group=np.where(a&b,0,np.where(a&~b,1,np.where(~a&b,2,3)))
    mat=group[None,:]
    palette=ListedColormap(['#D6E5E9','#248C7D','#C66B69','#E3C4BA'])
    fig,ax=plt.subplots(figsize=(max(9.,len(group)*.55+2),2.6))
    ax.imshow(mat,aspect='auto',vmin=0,vmax=3,cmap=palette)
    ax.set_xticks(np.arange(len(group)))
    ax.set_xticklabels([p.split('/')[-1] for p in selected.patient],
                       rotation=60,ha='right',fontsize=8)
    ax.set_yticks([0]);ax.set_yticklabels(['Held-out patient'])
    for j,(case,cl) in enumerate(zip(group,truth)):
        ax.text(j,0,'H' if cl==0 else 'S',va='center',ha='center',
                color='white' if case in (1,2) else '#27384A',weight='bold',fontsize=9)
    for side in ax.spines.values():side.set_visible(False)
    from matplotlib.patches import Patch
    labels=['Both correct','Augmentation recovered','Augmentation lost','Both wrong']
    handles=[Patch(facecolor=palette(i/3),label=label) for i,label in enumerate(labels)]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.72),
              ncol=2,frameon=False,fontsize=8)
    fig.tight_layout()
    save_fig(fig,'05_patient_gain_loss_map.pdf')


def plot_roc_pr(pred):
    arms=[('selected_pipeline','Selected pipeline',COLOR['main']),
          ('matched_real_only','Same model/features, real only',COLOR['negative']),
          ('aug_anchor::Real only','Fixed anchor, real only',COLOR['muted'])]
    fig,axs=plt.subplots(1,2,figsize=(10,4.3))
    for arm,label,color in arms:
        d=pred[pred.arm==arm].sort_values('outer_fold')
        fpr,tpr,_=roc_curve(d.y_true,d.score)
        prec,rec,_=precision_recall_curve(d.y_true,d.score)
        axs[0].step(fpr,tpr,where='post',color=color,lw=1.8,
                    label=f'{label}: {roc_auc_score(d.y_true,d.score):.3f}')
        axs[1].step(rec,prec,where='post',color=color,lw=1.8,
                    label=f'{label}: {average_precision_score(d.y_true,d.score):.3f}')
    axs[0].plot([0,1],[0,1],ls='--',color=COLOR['light'],lw=1)
    axs[0].set(xlim=(0,1),ylim=(0,1.02),xlabel='False-positive rate',
               ylabel='True-positive rate')
    axs[1].set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision')
    for ax in axs:
        ax.legend(frameon=False,fontsize=7.6,loc='lower right')
        tidy(ax)
    fig.tight_layout(w_pad=2)
    save_fig(fig,'06_outer_roc_precision_recall.pdf')


def plot_accuracy_time(metrics,efforts):
    groups=['selected_pipeline','matched_real_only']+[
        'replace_classifier::'+m for m in CLASSIFIERS]+[
        'replace_feature::'+f for f in FEATURES]
    m=metrics.set_index('arm')
    e=efforts.groupby('arm',sort=False).train_s.mean()
    fig,ax=plt.subplots(figsize=(8.4,5.))
    groups += ['raw_real_only::'+m for m in CLASSIFIERS]
    for arm in groups:
        val=100*m.loc[arm,'BA'];seconds=max(float(e.loc[arm]),.001)
        if arm=='selected_pipeline':color=COLOR['main'];size=110
        elif arm=='matched_real_only':color=COLOR['negative'];size=80
        else:color=COLOR['muted'];size=38
        ax.scatter(seconds,val,color=color,s=size,alpha=.85,zorder=3)
        if arm in ['selected_pipeline','matched_real_only']:
            ax.annotate('Selected' if arm=='selected_pipeline' else 'Real only',
                        (seconds,val),xytext=(6,6),textcoords='offset points',fontsize=9)
    ax.set_xscale('log');ax.set_ylim(0,105)
    ax.set_xlabel('Mean standalone cold train time per outer fold (s, log)')
    ax.set_ylabel('Held-out original-patient balanced accuracy (%)')
    tidy(ax);fig.tight_layout()
    save_fig(fig,'07_accuracy_computation_tradeoff.pdf')


def plot_synthetic_quality(quality):
    if quality.empty:return
    summary=quality.groupby('augmentation',sort=False).agg(
        Proximity=('NNProximityRatio','median'),
        Diversity=('DiversityRatio','median')).reset_index()
    fig,ax=plt.subplots(figsize=(7.8,5.))
    for _,r in summary.iterrows():
        col=COLOR['secondary'] if 'Flow' in r.augmentation else (
            COLOR['main'] if 'DDPM' in r.augmentation else COLOR['negative'])
        ax.scatter(r.Proximity,r.Diversity,s=65,color=col)
        ax.annotate(r.augmentation,(r.Proximity,r.Diversity),
                    xytext=(5,4),textcoords='offset points',fontsize=8)
    ax.axhline(1,ls='--',color=COLOR['light'],lw=1)
    ax.axvline(1,ls='--',color=COLOR['light'],lw=1)
    ax.set_xlabel('Synthetic-to-real NN distance / real-to-real NN distance')
    ax.set_ylabel('Synthetic within-class diversity / real within-class diversity')
    tidy(ax);fig.tight_layout()
    save_fig(fig,'08_synthetic_proximity_diversity.pdf')


def plot_modern_raw_controls(pairs):
    """Full selected pipeline versus each no-augmentation, raw-spectrum model."""
    d=pairs[pairs.Comparator.str.startswith('raw_real_only::')].copy()
    d=d.sort_values('DeltaBA_pp').reset_index(drop=True)
    yy=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(9.2,max(3.3,.58*len(d)+1.3)))
    ax.axvline(0,ls='--',color=COLOR['muted'],lw=1)
    for i,r in d.iterrows():
        col=COLOR['positive'] if r.DeltaBA_pp>=0 else COLOR['negative']
        ax.plot([r.CI_low_pp,r.CI_high_pp],[i,i],lw=2,color=col)
        ax.scatter(r.DeltaBA_pp,i,color=col,s=38,zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels(d.Comparator.str.replace('raw_real_only::','',regex=False),
                       fontsize=9)
    ax.set_xlabel('Selected full pipeline − raw real-only model: ΔBA (pp), 95% CI')
    ax.set_ylim(-.5,len(d)-.5)
    tidy(ax);fig.tight_layout()
    save_fig(fig,'09_vs_modern_raw_real_only.pdf')


# ------------------------------ PAPER OUTPUTS ------------------------------
def latex_escape(v):
    return str(v).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace(
        '&',r'\&').replace('%',r'\%').replace('#',r'\#')


def write_latex(frame,fields,path,caption,label):
    n=len(fields)
    lines=[r'\begin{table*}[t]' if n>=6 else r'\begin{table}[t]',
        r'\centering','\\caption{'+caption+'}','\\label{'+label+'}',
        r'\footnotesize',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{'+('l'+'r'*(n-1))+'}',r'\toprule',
        ' & '.join(latex_escape(header) for _,header in fields)+r' \\',r'\midrule']
    for _,row in frame.iterrows():
        vals=[]
        for col,_ in fields:
            v=row[col]
            if pd.isna(v):s='--'
            elif isinstance(v,(float,np.floating)):s=f'{v:.3f}'
            else:s=str(v)
            vals.append(latex_escape(s))
        lines.append(' & '.join(vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
              r'\end{table*}' if n>=6 else r'\end{table}']
    (OUTPUT_DIR/path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def write_reports(aug,joint,metrics,paired,anchor,quality,sel,X,y,wn,seconds,times):
    write_latex(aug,[('candidate','Augmentation'),('InnerBA','BA'),
         ('InnerBA_SD','SD'),('Selections','Chosen'),
         ('MeanTrainSeconds','Train (s)')], '01_augmentation_validation.tex',
         'First-stage augmentation selection with a fixed neural anchor on '
         'original-patient inner validation. These scores are not outer-test results.',
         'tab:augmentation_first')
    write_latex(joint,[('candidate','Classifier / representation'),
         ('InnerBA','BA'),('InnerBA_SD','SD'),('Selections','Chosen'),
         ('MeanTrainSeconds','Train (s)')], '02_joint_ablation.tex',
         'Second-stage joint ablation of neural classifiers and spectral '
         'representations, conditioned on training-only selected augmentation.',
         'tab:joint_second')
    main=metrics[metrics.arm.isin(['selected_pipeline','matched_real_only',
                       'forced_synthetic','aug_anchor::Real only'])].copy()
    write_latex(main,[('arm','Pipeline'),('Accuracy','Accuracy'),
         ('BA','BA'),('Sensitivity','Sensitivity'),('Specificity','Specificity'),
         ('AUC','AUC'),('Correct','Correct')], '03_outer_metrics.tex',
         'Original-patient nested outer-test performance. All pipeline '
         'choices are made without using outer-test labels. Pooled AUC '
         'is descriptive because score calibration differs between folds.',
         'tab:outer_metrics')
    write_latex(paired,[('Comparator','Component replacement'),
         ('DeltaBA_pp','Delta BA (pp)'),('CI_low_pp','CI low'),
         ('CI_high_pp','CI high'),('Recovered','Recovered'),
         ('Lost','Lost'),('HolmP','Holm p')], '04_paired_ablation.tex',
         'Paired original-patient comparison versus the selected pipeline. '
         'Stratified patient bootstrap intervals and exploratory exact '
         'McNemar tests with Holm correction; limited sample size.',
         'tab:paired_components')
    write_latex(anchor,[('Comparator','Fixed-anchor augmentation'),
         ('AugmentationMinusReal_pp','Delta BA (pp)'),
         ('AugmentationCI_low_pp','CI low'),
         ('AugmentationCI_high_pp','CI high'),
         ('Recovered','Real-only wins'),('Lost','Augmentation wins')],
         '05_paired_augmentation.tex',
         'Held-out augmentation-only ablation with identical prespecified '
         'classifier and representation for every method.',
         'tab:paired_augmentation')
    order=['01_augmentation_validation.tex','02_joint_ablation.tex',
           '03_outer_metrics.tex','04_paired_ablation.tex',
           '05_paired_augmentation.tex']
    (OUTPUT_DIR/'paper_tables.tex').write_text(''.join(
        '\\input{'+f+'}\n' for f in order),encoding='utf-8')
    metadata=dict(run_mode=RUN_MODE,exploratory=True,
         tested_original_patients=int(sel.outer_fold.nunique()),
         total_original_patients=int(len(y)),healthy=int((y==0).sum()),
         diseased=int((y==1).sum()),wavenumbers=int(len(wn)),
         outer_cv='patient-level leave-one-out' if RUN_MODE=='complete' else
             'stratified 2-per-class outer LOOCV subset (smoke test)',
         inner_cv=f'patient-level stratified {INNER_SPLITS}-fold',
         stage_1='augmentation first; fixed residual MLP + multiresolution features',
         stage_2='joint neural classifier x representation conditioned on chosen augmentation',
         stage_3='unseen original-patient test; matched component/augmentation controls',
         selection_priority='inner balanced accuracy, ROC AUC, accuracy, fixed candidate order',
         anchor_model=ANCHOR_MODEL,anchor_features=ANCHOR_FEATURE,
         augmentation_candidates=AUGMENTATIONS,classifier_candidates=CLASSIFIERS,
         feature_candidates=FEATURES,seed=SEED,device=str(DEVICE),
         epochs=dict(classifier=CLASSIFIER_EPOCHS,feature=FEATURE_EPOCHS,
                     generator=GENERATOR_STEPS),
         time_seconds=times,total_wall_seconds=float(seconds),
         caveat='Only 17 patients: CIs, p values, and AUC are exploratory. '
                'Modern neural methods fitted from scratch are not pretrained SOTA models. '
                'External patient cohorts are required to substantiate clinical claims.',
         data_paths=dict(healthy=str(HEALTHY_DIR),sick=str(SICK_DIR)))
    (OUTPUT_DIR/'run_config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    readme='''# Augmentation-first, joint classifier-feature ablation

## Protocol

1. **Augmentation:** fixed, prespecified Residual MLP + multi-resolution
   spectral pooling ranks real only, latent flow matching, latent DDPM, and an
   EDM-style score model, with one or more synthetic sample counts. Only inner
   original-patient validation is used for selection.
2. **Joint ablation:** selected augmentation is held fixed while Residual MLP,
   ConvNeXt1D, Patch Transformer, and Spectral Mixer are crossed with Raw SNV,
   multi-resolution patches, a masked spectral autoencoder, and a supervised
   contrastive embedding. TabPFN is optional, requiring independent weights.
3. **Outer evaluation:** each originally held-out patient is predicted once
   per arm, matching patients across comparisons. Changes to either the
   classifier or the representation are evaluated in isolation.

## Figures and interpretation

- `01_augmentation_inner_validation.pdf`: inner validation BA/SD and number
  of folds choosing each augmentation; **selection evidence only**.
- `02_augmentation_paired_heldout.pdf`: matched out-of-fold augmentation
  delta BA versus real-only using the identical fixed anchor; uncertainty.
- `03_joint_classifier_feature_heatmap.pdf`: conditional inner joint search.
- `04_component_ablation_paired_effects.pdf`: paired held-out degradation or
  improvement from replacing exactly one of the selected components.
- `05_patient_gain_loss_map.pdf`: which original patients were recovered/lost
  by the selected pipeline versus same model/features trained on real only.
- `06_outer_roc_precision_recall.pdf`: descriptive, uncalibrated cross-fold
  ROC/PR discrimination; curves are unreliable at this cohort size.
- `07_accuracy_computation_tradeoff.pdf`: resource vs held-out BA tradeoff.
- `08_synthetic_proximity_diversity.pdf`: train-only similarity/diversity check
  **not** proof of biochemical validity.
- `09_vs_modern_raw_real_only.pdf`: selected full pipeline versus contemporary
  classifier baselines using no augmentation and raw SNV only, with paired CIs.

CSV files contain per-patient outer predictions, stage-wise validation,
selection counts, uncertainty intervals, exact exploratory McNemar tests and
resource utilization. Five LaTeX tables are combined in `paper_tables.tex`.

### Limitations

The synthetic generator learns from 6–10 ORIGINAL examples per class, often
fewer in inner folds. It may memorize or distort biological signals. Every
augmented observation is generated only from original TRAIN patients. Do not
report synthetic samples as independent patients, pick final designs from
outer-test rankings, describe these compact networks as published pretrained
methods, or claim clinical significance without independent patient data.
For complete runs (all 17 held-out patients), the effective resolution of raw
accuracy is one patient. With only 17 total originals, inferential statistics
and effect-size intervals are extremely unstable.

## Install and run

`pip install numpy pandas scipy scikit-learn matplotlib torch`

Place `saudaveis` and `doentes` beside this file, set `RUN_MODE` at the top,
and run `python spectral_augmentation_first_sota_ablation.py`.
Set `RUN_MODE = 'complete'` for all 17 outer patients and the larger grid.
No ZIP archives or GPU are required. CPU training can be lengthy.
'''
    (OUTPUT_DIR/'README.md').write_text(readme,encoding='utf-8')


if __name__ == '__main__':
    run_experiment()
