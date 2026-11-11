"""Augmentation-first, modern spectral classification benchmark (original patients only).

REWRITTEN DOWNLOAD EDITION (2026-11-11). This file is stand-alone: edit the
USER SETTINGS below, keep the patient CSVs in the two specified folders,
and run it as a Python script. All output is written beside the dataset.


Order: (1) generative augmentation search using a LOCKED Patch Transformer probe,
       (2) joint modern classifier x representation ablation with selected augmentation,
       (3) matched real-only and modern real-only outer-fold benchmarks.

Complete: nested patient-level LOOCV outer + stratified inner CV, never test-sample leakage.
Quick: a reduced engineering smoke test; NOT valid for publication.

Required: numpy, pandas, scipy, scikit-learn, matplotlib, torch.
Python >=3.9. One CSV per patient in saudaveis/ and doentes/, columns wavenumber,intensity.
Networks are COMPACT, from-scratch, paper-inspired architecture families; they are
not pretrained foundation models or official reproductions of original papers.
With 17 patients results are exploratory. Do not infer statistical superiority from
selection or multiple benchmarking comparisons without external validation.
"""
from __future__ import annotations
from pathlib import Path
import json, time, math, warnings, random, sys, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter1d
from scipy.fft import dct
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve, average_precision_score,
                             precision_recall_curve, f1_score, matthews_corrcoef)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import torch
from torch import nn
from torch.nn import functional as F

# ------------------------------ USER SETTINGS -------------------------------
RUN_MODE = 'complete'                 # 'quick' (smoke test) or 'complete' (study)
OVERRIDES = {                   # Edit this ROOT to your folder, if needed.
    # 'ROOT': '/path/to/dataset',   # must contain saudaveis/ and doentes/
    # 'DEVICE': 'cuda',             # optional, if CUDA is available
}
ROOT = Path(OVERRIDES.get('ROOT', Path(__file__).resolve().parent))
HEALTHY_DIR = Path(OVERRIDES.get('HEALTHY_DIR', ROOT/'saudaveis'))
SICK_DIR = Path(OVERRIDES.get('SICK_DIR', ROOT/'doentes'))
OUT_DIR = Path(OVERRIDES.get('OUTPUT_DIR', ROOT/('modern_augmentation_first_'+RUN_MODE)))
SEED = int(OVERRIDES.get('SEED', 7))
if RUN_MODE not in ('quick','complete'):
    raise ValueError('RUN_MODE must be quick or complete')
FAST = RUN_MODE == 'quick'
DEVICE = torch.device(OVERRIDES.get('DEVICE','cpu'))
INPUT_POINTS = int(OVERRIDES.get('INPUT_POINTS', 128 if FAST else 256))
EPOCHS = int(OVERRIDES.get('EPOCHS', 3 if FAST else 32))
GEN_STEPS = int(OVERRIDES.get('GEN_STEPS', 18 if FAST else 280))
GEN_LATENT_DIM = int(OVERRIDES.get('GEN_LATENT_DIM', 4))
GEN_ODE_STEPS = int(OVERRIDES.get('GEN_ODE_STEPS', 6 if FAST else 24))
INNER_SPLITS = int(OVERRIDES.get('INNER_SPLITS', 2 if FAST else 3))
OUTER_QUICK_PER_CLASS = int(OVERRIDES.get('OUTER_QUICK_PER_CLASS', 1))
GEN_COUNTS = tuple(OVERRIDES.get('GEN_COUNTS', (4,) if FAST else (12,30)))
AUG_METHODS = ['Rectified Flow', 'Stochastic Interpolant', 'VP Flow Matching', 'Latent EDM']
CLASSIFIERS = ['Patch Transformer', 'ModernTCN-lite', 'ConvNeXt-1D', 'Spectral Mixer']
# Optional independently published, strong sequence-classification benchmarks.
# aeon 1.6 currently needs Python >=3.11 and is NOT required for the base script.
INCLUDE_AEON = bool(OVERRIDES.get('INCLUDE_AEON', False))
if INCLUDE_AEON:
    try:
        from aeon.classification.convolution_based import (MultiRocketHydraClassifier,
                                                            HydraClassifier)
    except ImportError as exc:
        raise ImportError('INCLUDE_AEON requires the aeon package and a supported Python version; '
                          'consider Python 3.11 and pip install aeon') from exc
    CLASSIFIERS += ['MultiRocket-Hydra (aeon)', 'HYDRA (aeon)']
REPRESENTATIONS = ['SNV raw', 'Multiscale patches', 'Gradient + shape',
                   'Scale-space bands', 'Frequency hybrid']
ACTIVE_AUG = (['Real only','Rectified Flow 4/class','Stochastic Interpolant 4/class']
              if FAST else ['Real only']+[f'{m} {n}/class' for m in AUG_METHODS for n in GEN_COUNTS])
ACTIVE_MODELS = CLASSIFIERS[:2] if FAST else CLASSIFIERS
ACTIVE_REPS = REPRESENTATIONS[:2] if FAST else REPRESENTATIONS
PROBE_MODEL = 'Patch Transformer'   # specified before looking at augmentation results
N_BOOT = 250 if FAST else int(OVERRIDES.get('N_BOOTSTRAP',2000))
BATCH_SIZE = int(OVERRIDES.get('BATCH_SIZE', 32))
OUT_DIR.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white',
                     'savefig.facecolor':'white','axes.grid':False,'font.size':10,
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
C = dict(ink='#283646', blue='#2F5C92', teal='#218B87', orange='#C68144',
         red='#B45B62', gray='#8695A5', pale='#DCE4ED')


def seeded(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def snv(x):
    x=np.atleast_2d(np.asarray(x,dtype=float))
    return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-10)


def load_data():
    rows=[]
    for directory,cl,name in [(HEALTHY_DIR,0,'healthy'),(SICK_DIR,1,'sick')]:
        files=sorted(directory.rglob('*.csv')) if directory.exists() else []
        if not files: raise FileNotFoundError('Missing patient CSVs in '+str(directory))
        for p in files:
            with p.open(encoding='utf-8-sig') as f: first=f.readline().strip()
            sep=';' if first.count(';')>first.count(',') else ','
            fields=[v.strip().strip('"') for v in first.split(sep)]
            def number(v):
                try: float(v); return True
                except ValueError: return False
            has_header=not(len(fields)==2 and all(number(v) for v in fields))
            a=pd.read_csv(p,sep=sep,header=0 if has_header else None,encoding='utf-8-sig')
            if a.shape[1]!=2: raise ValueError('Expected 2 columns: '+str(p))
            wav=pd.to_numeric(a.iloc[:,0],errors='raise').to_numpy(float)
            sig=pd.to_numeric(a.iloc[:,1],errors='raise').to_numpy(float)
            if not np.isfinite(wav).all() or not np.isfinite(sig).all():
                raise ValueError('Non-finite values in '+str(p))
            rows.append((name+'/'+p.relative_to(directory).with_suffix('').as_posix(),cl,wav,sig))
    wn=rows[0][2]
    if any(len(r[2])!=len(wn) or not np.allclose(r[2],wn,atol=1e-6,rtol=0) for r in rows):
        raise ValueError('All patients require the same wavenumber axis')
    ids=np.array([r[0] for r in rows]); y=np.array([r[1] for r in rows],int)
    if len(set(ids))!=len(ids) or min(np.bincount(y,minlength=2))<4:
        raise ValueError('Require unique patient IDs and at least 4 original patients per class')
    X=np.stack([r[3] for r in rows]).astype(float)
    return wn,snv(X),y,ids


def resample(x,n=INPUT_POINTS):
    x=np.atleast_2d(np.asarray(x,float)); old=np.linspace(0,1,x.shape[1]); new=np.linspace(0,1,n)
    return np.asarray([np.interp(new,old,s) for s in x],dtype=np.float32)


def representation(x,mode):
    """Deterministic transforms; no population-level fit and no access to labels."""
    x=snv(x)
    if mode=='SNV raw': out=resample(x)
    elif mode=='Multiscale patches':
        pieces=[resample(x,k) for k in (INPUT_POINTS//4, INPUT_POINTS//4, INPUT_POINTS//2)]
        out=np.concatenate(pieces,axis=1)
    elif mode=='Gradient + shape':
        out=np.concatenate([resample(x,INPUT_POINTS//2),
              resample(np.gradient(x,axis=1),INPUT_POINTS//2)],axis=1)
    elif mode=='Scale-space bands':
        parts=[]; levels=[1.5,6,18]
        smooth=[gaussian_filter1d(x,s,axis=1) for s in levels]
        for part in (x-smooth[0],smooth[0]-smooth[1],smooth[1]-smooth[2],smooth[2]):
            parts.append(resample(part,INPUT_POINTS//4))
        out=np.concatenate(parts,axis=1)
    elif mode=='Frequency hybrid':
        freq=np.abs(dct(x,type=2,axis=1,norm='ortho'))
        out=np.concatenate([resample(x,INPUT_POINTS//2),
                            resample(freq,INPUT_POINTS//2)],axis=1)
    else: raise KeyError(mode)
    if out.shape[1]!=INPUT_POINTS: raise RuntimeError('representation dimensions disagree')
    # normalize each representation per sample; no leakage or synthetic-driven fit
    return snv(out).astype(np.float32)


# ------------------------ GENERATORS: MODERN FAMILIES -----------------------
class VelocityNet(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d+7,64),nn.SiLU(),nn.Linear(64,64),
                               nn.SiLU(),nn.Linear(64,d))
    def forward(self,z,y,t):
        emb=torch.cat([t,torch.sin(2*math.pi*t),torch.cos(2*math.pi*t),
                      torch.sin(4*math.pi*t),torch.cos(4*math.pi*t)],1)
        return self.net(torch.cat([z,F.one_hot(y,2).float(),emb],1))


class DenoiseNet(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d+3,96),nn.SiLU(),nn.Linear(96,96),
                               nn.SiLU(),nn.Linear(96,d))
    def forward(self,x,y,sigma):
        return self.net(torch.cat([x,F.one_hot(y,2).float(),sigma.log()],1))


class Generator:
    """Lightweight latent-space implementations, NOT official SOTA checkpoints."""
    def __init__(self,method,seed): self.method=method;self.seed=int(seed)
    def fit(self,real,y):
        seeded(self.seed)
        self.train_real=snv(real)
        dim=min(GEN_LATENT_DIM,len(real)-2,real.shape[1])
        self.pca=PCA(n_components=dim,random_state=self.seed)
        self.scale=StandardScaler()
        z=self.scale.fit_transform(self.pca.fit_transform(self.train_real)).astype(np.float32)
        self.d=dim
        zz=torch.as_tensor(z,device=DEVICE); yy=torch.as_tensor(y,dtype=torch.long,device=DEVICE)
        pool=[torch.where(yy==i)[0] for i in (0,1)]
        if any(len(p)<2 for p in pool): raise ValueError('At least 2 real training patients per class')
        self.net=(DenoiseNet(dim) if self.method=='Latent EDM' else VelocityNet(dim)).to(DEVICE)
        optim=torch.optim.AdamW(self.net.parameters(),lr=1e-3,weight_decay=1e-4)
        for step in range(GEN_STEPS):
            labels=torch.randint(0,2,(32,),device=DEVICE)
            indices=torch.zeros_like(labels)
            for c in (0,1):
                mask=labels==c
                indices[mask]=pool[c][torch.randint(len(pool[c]),(int(mask.sum()),),device=DEVICE)]
            z1=zz[indices]; z0=torch.randn_like(z1)
            if self.method=='Latent EDM':
                sigma=torch.exp(torch.rand((len(labels),1),device=DEVICE)*4-2)
                noise=torch.randn_like(z1)
                x=z1+sigma*noise
                denoised=self.net(x/(1+sigma**2).sqrt(),labels,sigma)
                loss=(F.mse_loss(denoised,z1,reduction='none')/(1+sigma**2)).mean()
            else:
                t=torch.rand((len(labels),1),device=DEVICE).clamp(.001,.999)
                if self.method=='Rectified Flow':
                    xt=(1-t)*z0+t*z1; velocity=z1-z0
                elif self.method=='Stochastic Interpolant':
                    eps=torch.randn_like(z1);amp=.2
                    xt=(1-t)*z0+t*z1+amp*torch.sin(math.pi*t)*eps
                    velocity=z1-z0+amp*math.pi*torch.cos(math.pi*t)*eps
                elif self.method=='VP Flow Matching':
                    xt=torch.cos(math.pi*t/2)*z0+torch.sin(math.pi*t/2)*z1
                    velocity=(math.pi/2)*(-torch.sin(math.pi*t/2)*z0+
                                          torch.cos(math.pi*t/2)*z1)
                else: raise ValueError(self.method)
                loss=F.mse_loss(self.net(xt,labels,t),velocity)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def sample(self,n):
        seeded(self.seed+int(n)*197+33)
        labels=np.repeat([0,1],int(n)).astype(int)
        y=torch.as_tensor(labels,device=DEVICE)
        if self.method=='Latent EDM':
            # Probability-flow ODE for a variance-exploding denoising model.
            sigmas=torch.exp(torch.linspace(math.log(5.),math.log(.015),GEN_ODE_STEPS+1,
                                            device=DEVICE))
            z=torch.randn((len(y),self.d),device=DEVICE)*sigmas[0]
            for i in range(GEN_ODE_STEPS):
                s=sigmas[i].expand(len(y),1); nxt=sigmas[i+1]
                den=self.net(z/(1+s*s).sqrt(),y,s)
                deriv=(z-den)/s
                znew=z+(nxt-s)*deriv
                if i<GEN_ODE_STEPS-1:
                    sn=nxt.expand(len(y),1)
                    den2=self.net(znew/(1+sn*sn).sqrt(),y,sn)
                    d2=(znew-den2)/sn
                    z=z+(nxt-s)*.5*(deriv+d2)
                else: z=znew
        else:
            z=torch.randn((len(y),self.d),device=DEVICE)
            dt=1/GEN_ODE_STEPS
            for i in range(GEN_ODE_STEPS):
                t=torch.full((len(y),1),i*dt,device=DEVICE)
                v=self.net(z,y,t)
                te=torch.full_like(t,(i+1)*dt)
                v2=self.net(z+dt*v,y,te)
                z=z+dt*.5*(v+v2)
        zz=z.cpu().numpy().clip(-8,8)
        synth=snv(self.pca.inverse_transform(self.scale.inverse_transform(zz)))
        if not np.isfinite(synth).all(): raise FloatingPointError('Invalid generated spectra')
        return synth,labels


# --------------------------- MODERN NEURAL MODELS ---------------------------
class PatchTransformer(nn.Module):
    def __init__(self,d):
        super().__init__()
        patch=8
        self.p=nn.Conv1d(1,48,patch,stride=patch)
        tokens=d//patch
        self.pos=nn.Parameter(torch.zeros(1,tokens,48))
        layer=nn.TransformerEncoderLayer(48,4,96,dropout=.1,batch_first=True,
                                         activation='gelu',norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(48)
        self.out=nn.Linear(48,2)
    def forward(self,x):
        z=self.p(x[:,None]).transpose(1,2)+self.pos
        return self.out(self.norm(self.encoder(z)).mean(1))


class ConvNeXtBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.dw=nn.Conv1d(c,c,7,padding=3,groups=c)
        self.norm=nn.LayerNorm(c)
        self.p1=nn.Linear(c,3*c);self.p2=nn.Linear(3*c,c)
    def forward(self,x):
        z=self.dw(x).transpose(1,2)
        z=self.p2(F.gelu(self.p1(self.norm(z)))).transpose(1,2)
        return x+z


class ConvNeXt1D(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.stem=nn.Conv1d(1,48,7,stride=4,padding=3)
        self.blocks=nn.Sequential(ConvNeXtBlock(48),ConvNeXtBlock(48))
        self.out=nn.Linear(48,2)
    def forward(self,x): return self.out(self.blocks(self.stem(x[:,None])).mean(-1))


class InceptionLayer(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.proj=nn.Conv1d(cin,24,1)
        self.convs=nn.ModuleList([nn.Conv1d(24,cout,k,padding=k//2)
                                  for k in [5,11,21]])
        self.pool=nn.Sequential(nn.MaxPool1d(3,stride=1,padding=1),nn.Conv1d(cin,cout,1))
        self.norm=nn.BatchNorm1d(cout*4)
    def forward(self,x):
        z=self.proj(x)
        return F.gelu(self.norm(torch.cat([m(z) for m in self.convs]+[self.pool(x)],dim=1)))


class InceptionTimeLite(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.a=InceptionLayer(1,12);self.b=InceptionLayer(48,12)
        self.short=nn.Conv1d(1,48,1)
        self.out=nn.Linear(48,2)
    def forward(self,x):
        inp=x[:,None]
        z=self.b(self.a(inp))+self.short(inp)
        return self.out(F.gelu(z).mean(-1))


class ModernTCNBlock(nn.Module):
    # Compact modern dilated large-kernel convolution family; not official ModernTCN.
    def __init__(self,channels,dilation):
        super().__init__()
        self.depth=nn.Conv1d(channels,channels,15,padding=7*dilation,
                             dilation=dilation,groups=channels)
        self.norm=nn.GroupNorm(8,channels)
        self.expand=nn.Conv1d(channels,channels*2,1)
        self.reduce=nn.Conv1d(channels*2,channels,1)
    def forward(self,x):
        v=self.depth(x)
        v=self.reduce(F.gelu(self.expand(self.norm(v))))
        return x+v


class ModernTCNLite(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.stem=nn.Conv1d(1,48,7,stride=4,padding=3)
        self.blocks=nn.Sequential(*[ModernTCNBlock(48,2**i) for i in range(3)])
        self.head=nn.Linear(48,2)
    def forward(self,x):
        z=self.blocks(self.stem(x[:,None]))
        return self.head(z.mean(-1))


class MixerBlock(nn.Module):
    def __init__(self,tokens,dim):
        super().__init__()
        self.n1=nn.LayerNorm(dim)
        self.token=nn.Sequential(nn.Linear(tokens,2*tokens),nn.GELU(),nn.Linear(2*tokens,tokens))
        self.n2=nn.LayerNorm(dim)
        self.channel=nn.Sequential(nn.Linear(dim,2*dim),nn.GELU(),nn.Linear(2*dim,dim))
    def forward(self,x):
        x=x+self.token(self.n1(x).transpose(1,2)).transpose(1,2)
        return x+self.channel(self.n2(x))


class SpectralMixer(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.patch=nn.Conv1d(1,48,8,stride=8)
        self.blocks=nn.Sequential(MixerBlock(d//8,48),MixerBlock(d//8,48))
        self.head=nn.Linear(48,2)
    def forward(self,x):
        z=self.patch(x[:,None]).transpose(1,2)
        return self.head(self.blocks(z).mean(1))


def create_model(name):
    return {'Patch Transformer':PatchTransformer,'ConvNeXt-1D':ConvNeXt1D,
            'ModernTCN-lite':ModernTCNLite,'Spectral Mixer':SpectralMixer}[name](INPUT_POINTS).to(DEVICE)


def fit_predict(name,X,y,Z,seed,return_time=False):
    seeded(seed)
    if name.endswith('(aeon)'):
        # Official reference implementation, distinct from our compact neural models.
        model=(MultiRocketHydraClassifier(n_kernels=256 if FAST else 2000)
               if name.startswith('MultiRocket') else
               HydraClassifier(n_kernels=4 if FAST else 8,n_groups=8 if FAST else 64))
        data=np.asarray(X,np.float32)[:,None,:]
        t0=time.perf_counter()
        model.fit(data,np.asarray(y,int))
        zz=np.asarray(Z,np.float32)[:,None,:]
        score=np.asarray(model.predict_proba(zz))[:,1].astype(float)
        elapsed=time.perf_counter()-t0
        return (score,elapsed) if return_time else score

    net=create_model(name)
    optim=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
    xx=torch.as_tensor(np.asarray(X,np.float32),device=DEVICE)
    yy=torch.as_tensor(np.asarray(y,int),device=DEVICE)
    zz=torch.as_tensor(np.asarray(Z,np.float32),device=DEVICE)
    t0=time.perf_counter()
    rng=np.random.default_rng(seed)
    counts=np.bincount(np.asarray(y,int),minlength=2).astype(float)
    class_weights=torch.as_tensor(len(y)/(2*np.maximum(counts,1)),dtype=torch.float32,device=DEVICE)
    net.train()
    for _ in range(EPOCHS):
        perm=rng.permutation(len(y))
        for batch in np.array_split(perm,max(1,math.ceil(len(y)/BATCH_SIZE))):
            pred=net(xx[batch]); loss=F.cross_entropy(pred,yy[batch],weight=class_weights)
            optim.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            optim.step()
    net.eval()
    with torch.no_grad():
        logits=net(zz)
        prob=logits.softmax(-1)[:,1].cpu().numpy()
    runtime=time.perf_counter()-t0
    del net
    return (np.asarray(prob,float),runtime) if return_time else np.asarray(prob,float)


# --------------------------- NESTED EXPERIMENT ------------------------------
def split_aug(name):
    if name=='Real only': return 'Real only',0
    method,count=name.rsplit(' ',1)
    return method,int(count.split('/')[0])


class FoldData:
    def __init__(self,X,y,seed):
        self.real=X;self.y=y;self.seed=seed
        self.generators={};self.samples={};self.fit_seconds={};self.sample_seconds={}
        self.rep_cache={}
    def dataset(self,name):
        if name=='Real only': return self.real,self.y,0.
        method,n=split_aug(name)
        if method not in self.generators:
            t=time.perf_counter()
            self.generators[method]=Generator(method,self.seed+111).fit(self.real,self.y)
            self.fit_seconds[method]=time.perf_counter()-t
        key=(method,n)
        if key not in self.samples:
            t=time.perf_counter();self.samples[key]=self.generators[method].sample(n)
            self.sample_seconds[key]=time.perf_counter()-t
        samples,labels=self.samples[key]
        return np.vstack((self.real,samples)),np.r_[self.y,labels],(
            self.fit_seconds[method]+self.sample_seconds[key])
    def features(self,aug,rep):
        key=(aug,rep)
        if key not in self.rep_cache:
            a,b,cost=self.dataset(aug)
            t=time.perf_counter()
            self.rep_cache[key]=(representation(a,rep),b,cost+time.perf_counter()-t)
        return self.rep_cache[key]


def score_set(y,score):
    pred=(np.asarray(score)>=.5).astype(int)
    return {'BA':float(balanced_accuracy_score(y,pred)),
            'AUC':float(roc_auc_score(y,score)),
            'Accuracy':float(accuracy_score(y,pred))}


def pick_winner(candidates,records):
    """Fixed priority metric: patient-level inner BA > AUC > accuracy > fixed order."""
    stats={}
    for c in candidates:
        subset=[r for r in records if r['candidate']==c]
        y=np.concatenate([r['truth'] for r in subset]);p=np.concatenate([r['score'] for r in subset])
        stats[c]=score_set(y,p)
    chosen=max(candidates,key=lambda c:(stats[c]['BA'],stats[c]['AUC'],
                                         stats[c]['Accuracy'],-candidates.index(c)))
    return chosen,stats


def get_inner_records(X,y,outer_train,fold_id):
    sk=StratifiedKFold(n_splits=INNER_SPLITS,shuffle=True,random_state=SEED+fold_id)
    groups=[]
    for j,(train,val) in enumerate(sk.split(X[outer_train],y[outer_train])):
        tr=outer_train[train];va=outer_train[val]
        seed=SEED+fold_id*1009+j*71
        groups.append((FoldData(X[tr],y[tr],seed),X[va],y[va],j))
    return groups


def evaluate_grid(groups,candidates,stage,augmentation=None,baseline=False):
    """Candidate choices do not use outer held-out test patient or test labels."""
    rows=[];total_times={c:[] for c in candidates}
    for data,Xval,yval,inner_fold in groups:
        for c in candidates:
            if stage=='augmentation':
                aug=c;model=PROBE_MODEL;rep='SNV raw'
            else:
                model,rep=c.split(' || ',1)
                aug='Real only' if baseline else augmentation
            xx,yy,gen_s=data.features(aug,rep)
            xval=representation(Xval,rep)
            scores,clf_s=fit_predict(model,xx,yy,xval,data.seed+19,True)
            rows.append({'candidate':c,'stage':stage,'inner_fold':inner_fold,
                         'truth':np.asarray(yval,int),'score':np.asarray(scores,float),
                         'seconds':clf_s+gen_s})
            total_times[c].append(clf_s+gen_s)
    return rows,total_times


def metrics(d):
    y=np.asarray(d.truth,int);pred=np.asarray(d.pred,int);score=np.asarray(d.score,float)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(N=len(y),Correct=int((y==pred).sum()),
                Accuracy=accuracy_score(y,pred),BA=balanced_accuracy_score(y,pred),
                Sensitivity=tp/max(1,tp+fn),Specificity=tn/max(1,tn+fp),
                AUC=roc_auc_score(y,score),AP=average_precision_score(y,score),
                F1=f1_score(y,pred,zero_division=0),MCC=matthews_corrcoef(y,pred),
                TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn))


def paired_ba_permutation(y,reference_prediction,comparator_prediction):
    # Exploratory exact paired sign-flip test of class-balanced correctness changes.
    # Assumes patient-level exchangeability; overlapping LOOCV fits weaken inference.
    y=np.asarray(y,int); a=np.asarray(reference_prediction,int); b=np.asarray(comparator_prediction,int)
    counts=np.bincount(y,minlength=2)
    delta=((a==y).astype(float)-(b==y).astype(float))/(2*counts[y])
    delta=delta[np.abs(delta)>1e-12]
    k=len(delta)
    if not k: return 1.
    # <=17 original patients is affordable; assert against accidental large studies.
    if k>20: raise ValueError('Exact sign-flip enumeration is limited to 20 discordant patients')
    states=np.arange(1<<k,dtype=np.uint32)[:,None]
    bits=((states>>np.arange(k,dtype=np.uint32)[None,:])&1).astype(np.int8)
    distribution=(2*bits-1).astype(float)@delta
    return float(np.mean(np.abs(distribution)>=abs(float(delta.sum()))-1e-12))


def paired(pred,main='best_synthetic'):
    groups={c:df.sort_values('fold').reset_index(drop=True)
            for c,df in pred.groupby('arm')}
    ref=groups[main];out=[]
    for j,(name,base) in enumerate(groups.items()):
        if name==main:continue
        assert np.array_equal(ref.patient,base.patient)
        truth=ref.truth.to_numpy(int); r=ref.pred.to_numpy(int);b=base.pred.to_numpy(int)
        err_to_right=int(((b!=truth)&(r==truth)).sum())
        right_to_err=int(((b==truth)&(r!=truth)).sum())
        rg=np.random.default_rng(SEED+977+j)
        zero=np.where(truth==0)[0];one=np.where(truth==1)[0]
        i0=rg.choice(zero,(N_BOOT,len(zero)),replace=True)
        i1=rg.choice(one,(N_BOOT,len(one)),replace=True)
        change=50*((r[i0]==truth[i0]).mean(1)-(b[i0]==truth[i0]).mean(1)+
                   (r[i1]==truth[i1]).mean(1)-(b[i1]==truth[i1]).mean(1))
        p=binomtest(err_to_right,err_to_right+right_to_err,.5).pvalue if err_to_right+right_to_err else 1.
        out.append(dict(Comparator=name,DeltaBA_pp=100*(balanced_accuracy_score(truth,r)-
                                                      balanced_accuracy_score(truth,b)),
            CI_low_pp=np.percentile(change,2.5),CI_high_pp=np.percentile(change,97.5),
            Recovered=err_to_right,New_errors=right_to_err,
            Exact_McNemar_p=np.nan if FAST else p,
            Exact_Paired_BA_Permutation_p=np.nan if FAST else
                paired_ba_permutation(truth,r,b)))
    out=pd.DataFrame(out)
    if FAST:
        out['Holm_McNemar_p']=np.nan
        out['Holm_Paired_BA_p']=np.nan
    else:
        for source,target in [('Exact_McNemar_p','Holm_McNemar_p'),
                              ('Exact_Paired_BA_Permutation_p','Holm_Paired_BA_p')]:
            p=out[source].to_numpy(float);order=np.argsort(p)
            adjust=np.empty(len(p));adjust[order]=np.minimum(1,np.maximum.accumulate(
                p[order]*(len(p)-np.arange(len(p)))))
            out[target]=adjust
    return out


def summarize_search(rows,select,stage,candidates):
    filtered=[r for r in rows if r['stage']==stage]
    out=[]
    for candidate in candidates:
        subs=[r for r in filtered if r['candidate']==candidate]
        y=np.concatenate([r['truth'] for r in subs]);s=np.concatenate([r['score'] for r in subs])
        val=score_set(y,s)
        folds=[k for k in set(r['outer_fold'] for r in subs)]
        # Each candidate has one aggregated BA per outer fold.
        bas=[]
        for f in sorted(folds):
            arr=[r for r in subs if r['outer_fold']==f]
            bas.append(score_set(np.concatenate([r['truth'] for r in arr]),
                                 np.concatenate([r['score'] for r in arr]))['BA'])
        out.append(dict(Candidate=candidate,InnerBA_mean=np.mean(bas),
                        InnerBA_sd=np.std(bas,ddof=1) if len(bas)>1 else 0.,
                        PooledInnerAUC=val['AUC'],
                        Selected=int(sum(k['stage']==stage and k['candidate']==candidate for k in select)),
                        MeanSeconds=np.mean([r['seconds'] for r in subs]),
                        Outer_folds=len(folds)))
    return pd.DataFrame(out)


def tidy(ax):
    for spine in ('top','right'): ax.spines[spine].set_visible(False)
    ax.tick_params(length=0,pad=5)


def title(fig,heading,explanation):
    fig.text(.035,.982,heading,ha='left',va='top',fontsize=13,color=C['ink'],weight='bold')
    fig.text(.035,.035,textwrap.fill(explanation,width=112),ha='left',va='bottom',
             fontsize=8.1,color=C['ink'],linespacing=1.25)


def fig1_augmentation(aug,select):
    # Plot INNER validation BA change versus the real-only control across outer folds.
    d=aug.set_index('Candidate').reindex(ACTIVE_AUG).reset_index()
    base=float(d.loc[d.Candidate=='Real only','InnerBA_mean'].iloc[0])*100
    gains=(d.InnerBA_mean.to_numpy()-base/100)*100
    y=np.arange(len(d))
    fig,ax=plt.subplots(figsize=(12,max(5.4,2.9+.48*len(d))))
    ax.axvline(0,color=C['ink'],lw=1)
    for k,g in enumerate(gains):
        col=C['gray'] if k==0 else C['teal'] if g>=0 else C['red']
        ax.barh(k,g,height=.58,color=col)
        ax.text(g+(.4 if g>=0 else -.4),k,f'{g:+.1f} pp',ha='left' if g>=0 else 'right',va='center',fontsize=9)
    lim=max(8,1.3*max(abs(gains)))
    ax.set_xlim(-lim,lim);ax.set_yticks(y);ax.set_yticklabels(d.Candidate)
    ax.invert_yaxis();ax.set_xlabel('Inner-CV balanced accuracy change vs no augmentation (pp)')
    ax.set_title(f'Fixed Patch Transformer probe; Real-only BA = {base:.1f}%',loc='left',fontsize=10,pad=14)
    tidy(ax)
    fig.subplots_adjust(left=.24,right=.94,top=.83,bottom=.17)
    title(fig,'Figure 1 | Which generator helps the fixed classification probe?',
        'Positive = better than the real-only probe. Selection uses training patients only. '
        'These are descriptive INNER-CV results, not outer-test improvements.')
    fig.savefig(OUT_DIR/'01_augmentation_probe_gain.pdf',bbox_inches='tight');plt.close(fig)


def fig2_joint(joint):
    z=joint.copy();z[['Classifier','Feature']]=z.Candidate.str.split(' || ',n=1,expand=True,regex=False)
    mat=z.pivot(index='Classifier',columns='Feature',values='InnerBA_mean').reindex(index=ACTIVE_MODELS,columns=ACTIVE_REPS)
    sel=z.pivot(index='Classifier',columns='Feature',values='Selected').reindex(index=ACTIVE_MODELS,columns=ACTIVE_REPS)
    arr=100*mat.to_numpy();fig,ax=plt.subplots(figsize=(12.7,max(4.9,2.9+.62*len(ACTIVE_MODELS))))
    im=ax.imshow(arr,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set_xticks(range(len(ACTIVE_REPS)));ax.set_xticklabels(ACTIVE_REPS,rotation=24,ha='right')
    ax.set_yticks(range(len(ACTIVE_MODELS)));ax.set_yticklabels(ACTIVE_MODELS)
    ax.set_xlabel('Spectral representation')
    for i in range(len(ACTIVE_MODELS)):
        for j in range(len(ACTIVE_REPS)):
            v=arr[i,j];n=int(sel.iloc[i,j]);ax.text(j,i,f'{v:.1f}%\nselected {n}×',
                ha='center',va='center',fontsize=8,color='white' if v>65 else C['ink'])
    fig.colorbar(im,ax=ax,fraction=.034,pad=.03,label='Mean inner balanced accuracy (%)')
    fig.subplots_adjust(left=.18,right=.92,top=.82,bottom=.28)
    title(fig,'Figure 2 | Joint classifier × representation ablation',
        'Each cell is inner-CV balanced accuracy with the generator chosen in Stage 1. '
        '“Selected” is the number of outer folds choosing that pair; it is NOT a test accuracy.')
    fig.savefig(OUT_DIR/'02_joint_model_feature_heatmap.pdf',bbox_inches='tight');plt.close(fig)


def fig3_benchmark(score):
    order=['matched_real_only','best_synthetic','final_cv_selected']+[f'real_only::{m}' for m in ACTIVE_MODELS]
    d=score.set_index('arm').reindex(order).reset_index()
    labels=['Same selected pair | REAL ONLY','Same selected pair | BEST SYNTHETIC',
            'CV-chosen: real or synthetic']+[f'{m} | REAL ONLY (inner-selected features)' for m in ACTIVE_MODELS]
    colors=[C['blue'],C['orange'],C['teal']]+[C['gray']]*len(ACTIVE_MODELS)
    fig,ax=plt.subplots(figsize=(12.5,max(5.9,2.5+.53*len(d))))
    for i,r in d.iterrows():
        v=100*r.BA;ax.barh(i,v,height=.6,color=colors[i])
        ax.text(min(101,v+1),i,f'{v:.1f}%   ({int(r.Correct)}/{int(r.N)})',va='center',fontsize=8.5)
    ax.set_yticks(range(len(d)));ax.set_yticklabels(labels)
    ax.invert_yaxis();ax.set_xlim(0,115)
    ax.set_xlabel('Balanced accuracy on ORIGINAL held-out patients (%)')
    ax.axhline(2.5,color=C['pale'],lw=1.7);tidy(ax)
    fig.subplots_adjust(left=.36,right=.95,top=.83,bottom=.16)
    title(fig,'Figure 3 | Patient-level benchmark: is augmentation beneficial?',
        'The FIRST TWO rows use identical fold-specific classifier and features. Best synthetic is selected among generators even when real-only wins the probe. '
        'Only training data change. All other modern classifiers are REAL-ONLY baselines.')
    fig.savefig(OUT_DIR/'03_outer_patient_balanced_accuracy.pdf',bbox_inches='tight');plt.close(fig)


def fig4_patient_changes(pred):
    r=pred[pred.arm=='matched_real_only'].sort_values('fold')
    a=pred[pred.arm=='best_synthetic'].sort_values('fold')
    assert np.array_equal(a.patient,r.patient)
    y=r.truth.to_numpy(int); b=r.pred.to_numpy(int)==y;g=a.pred.to_numpy(int)==y
    good=int((~b&g).sum());bad=int((b&~g).sum());both=int((b&g).sum());neither=int((~b&~g).sum())
    fig,axs=plt.subplots(1,2,figsize=(12,5.4),gridspec_kw={'width_ratios':[.75,1.25]})
    ax=axs[0];counts=[good,bad,both,neither]
    names=['Recovered\nerrors','New errors','Correct in\nboth','Wrong in\nboth']
    colors=[C['teal'],C['red'],C['blue'],C['gray']]
    ax.barh(range(4),counts,color=colors,height=.56)
    ax.set_yticks(range(4));ax.set_yticklabels(names);ax.invert_yaxis()
    ax.set_xlabel('Number of original patients')
    ax.set_xlim(0,max(2,max(counts)+1))
    for j,v in enumerate(counts): ax.text(v+.08,j,str(v),va='center')
    tidy(ax)
    ax=axs[1]
    state=np.where(~b&g,2,np.where(b&~g,-1,np.where(b&g,1,0)))
    from matplotlib.colors import ListedColormap,BoundaryNorm
    cmap=ListedColormap([C['red'],C['gray'],C['blue'],C['teal']])
    ax.imshow(state[None,:],aspect='auto',cmap=cmap,
              norm=BoundaryNorm([-1.5,-.5,.5,1.5,2.5],4))
    ax.set_yticks([]);ax.set_xticks(range(len(y)))
    ax.set_xticklabels([f'{"H" if val==0 else "S"}{i+1}' for i,val in enumerate(y)],rotation=60)
    ax.set_xlabel('Original held-out patient (H=healthy; S=sick)')
    for j,val in enumerate(state):
        ax.text(j,0,'+' if val==2 else '−' if val==-1 else '✓' if val==1 else '×',
                ha='center',va='center',fontsize=13,color='white')
    fig.subplots_adjust(left=.13,right=.98,wspace=.27,top=.82,bottom=.19)
    title(fig,'Figure 4 | Does augmentation correct errors or introduce new ones?',
        'The synthetic method was selected among generators using the fixed probe. A recovered error was wrong without augmentation but correct with it; '
        'a new error is the reverse. Each original patient appears exactly once.')
    fig.savefig(OUT_DIR/'04_paired_patient_error_transitions.pdf',bbox_inches='tight');plt.close(fig)


def fig5_effects(effects):
    d=effects.copy();d['order']=d.Comparator.map(lambda v:0 if v=='matched_real_only' else 1)
    d=d.sort_values(['order','Comparator']).reset_index(drop=True)
    y=np.arange(len(d));vals=d.DeltaBA_pp.to_numpy();lo=d.CI_low_pp.to_numpy();hi=d.CI_high_pp.to_numpy()
    fig,ax=plt.subplots(figsize=(12,max(5.5,2.4+.52*len(d))))
    ax.axvline(0,color=C['ink'],lw=1,linestyle='--')
    for i,(v,l,h) in enumerate(zip(vals,lo,hi)):
        ax.plot([l,h],[i,i],color=C['gray'],linewidth=2)
        ax.scatter([v],[i],color=C['teal'] if v>=0 else C['red'],s=58,zorder=4)
    labels=[('Matched same pair, REAL ONLY' if k=='matched_real_only' else
             k.replace('real_only::','REAL ONLY: ').replace('final_cv_selected','CV selected')) for k in d.Comparator]
    ax.set_yticks(y);ax.set_yticklabels(labels);ax.invert_yaxis()
    bound=max(8,1.2*np.max(np.abs(np.r_[lo,hi])))
    ax.set_xlim(-bound,bound)
    ax.set_xlabel('Paired Δ balanced accuracy (percentage points): augmented − comparator')
    tidy(ax)
    fig.subplots_adjust(left=.33,right=.96,top=.84,bottom=.16)
    title(fig,'Figure 5 | Size and uncertainty of the augmentation benefit',
        'Dots: paired BA differences. Lines: exploratory 95% stratified-patient bootstrap '
        'intervals (NOT proof of significance). Crossing zero = improvement is uncertain.')
    fig.savefig(OUT_DIR/'05_paired_effect_sizes_and_uncertainty.pdf',bbox_inches='tight');plt.close(fig)


def fig6_roc(pred):
    sets=[('matched_real_only','Matched real only',C['blue']),
          ('best_synthetic','Matched best synthetic',C['orange']),
          ('final_cv_selected','Inner-CV selected',C['teal'])]
    fig,axs=plt.subplots(1,2,figsize=(11.7,5.3))
    for arm,name,color in sets:
        d=pred[pred.arm==arm].sort_values('fold')
        f,t,_=roc_curve(d.truth,d.score);p,rec,_=precision_recall_curve(d.truth,d.score)
        axs[0].step(f,t,where='post',color=color,lw=1.9,label=f'{name} (AUC {roc_auc_score(d.truth,d.score):.2f})')
        axs[1].step(rec,p,where='post',color=color,lw=1.9,label=f'{name} (AP {average_precision_score(d.truth,d.score):.2f})')
    axs[0].plot([0,1],[0,1],'--',color=C['gray'],lw=1)
    sample=pred[pred.arm=='matched_real_only']
    axs[1].axhline(sample.truth.mean(),linestyle='--',color=C['gray'],lw=1)
    axs[0].set(xlim=(0,1),ylim=(0,1.02),xlabel='False-positive rate',ylabel='Sensitivity')
    axs[1].set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall (sick patients)',ylabel='Precision')
    for ax,name in zip(axs,['A  ROC curve','B  Precision–recall curve']):
        tidy(ax);ax.set_title(name,loc='left',fontsize=10,pad=12)
        ax.legend(loc='lower right',frameon=False,fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,wspace=.29,top=.82,bottom=.18)
    title(fig,'Figure 6 | Discrimination among held-out original patients',
        'Scores pool separately fitted outer-fold models; they are not cross-fold calibrated. '
        'AUC and average precision are descriptive secondary metrics.')
    fig.savefig(OUT_DIR/'06_patient_roc_and_precision_recall.pdf',bbox_inches='tight');plt.close(fig)


def fig7_spectra(example,wn):
    if not example: return
    x,y,synthetic,ys,method=example
    fig,axes=plt.subplots(1,2,figsize=(12,5.1),sharey=True)
    for cl,ax in enumerate(axes):
        real=x[y==cl];fake=synthetic[ys==cl]
        ax.plot(wn,np.median(real,axis=0),color=C['blue'],lw=1.7,label='Real train median')
        ax.plot(wn,np.median(fake,axis=0),color=C['orange'],lw=1.7,label='Generated median')
        ax.fill_between(wn,np.quantile(real,.1,axis=0),np.quantile(real,.9,axis=0),
                        color=C['blue'],alpha=.13)
        ax.fill_between(wn,np.quantile(fake,.1,axis=0),np.quantile(fake,.9,axis=0),
                        color=C['orange'],alpha=.12)
        ax.set_xlabel('Wavenumber');ax.set_title('Healthy' if cl==0 else 'Sick',loc='left',fontsize=10)
        tidy(ax)
    axes[0].set_ylabel('SNV-normalized intensity')
    axes[0].legend(loc='upper left',frameon=False,fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,wspace=.15,top=.82,bottom=.17)
    title(fig,'Figure 7 | Spectral plausibility of synthetic data ('+method+')',
          'Medians and 10–90% bands shown for one OUTER TRAINING fold only. '
          'Visual similarity does not establish biological validity or generator superiority.')
    fig.savefig(OUT_DIR/'07_synthetic_vs_train_spectra.pdf',bbox_inches='tight');plt.close(fig)


def fig8_resources(search):
    fig,axes=plt.subplots(1,2,figsize=(12.7,5.7))
    for ax,stage,subtitle in zip(axes,['augmentation','joint'],
        ['A  Generator search','B  Classifier + representation search']):
        d=search[search.stage==stage].copy()
        xx=np.maximum(d.MeanSeconds.to_numpy(),1e-5)
        yy=100*d.InnerBA_mean.to_numpy()
        ax.scatter(xx,yy,color=C['blue'] if stage=='joint' else C['teal'],
                   alpha=.66,s=48)
        best=d.sort_values(['Selected','InnerBA_mean'],ascending=False).iloc[0]
        x=max(float(best.MeanSeconds),1e-5)
        y=100*float(best.InnerBA_mean)
        ax.scatter(x,y,s=115,facecolors='none',edgecolors=C['orange'],lw=1.7,zorder=5)
        short=str(best.Candidate).replace(' || ',' + ')
        if len(short)>37: short=short[:34]+'...'
        ax.text(.03,.96,'Most selected: '+short,transform=ax.transAxes,
                va='top',ha='left',fontsize=8.1,color=C['ink'])
        ax.set_ylim(0,105)
        xmin,xmax=xx.min(),xx.max()
        if xmax/max(xmin,1e-8)>8:
            ax.set_xscale('log')
            ax.set_xlim(xmin*.7,xmax*1.5)
        else:
            width=max(xmax-xmin,xmax*.3,1e-5)
            ax.set_xlim(max(0,xmin-.25*width),xmax+.25*width)
        ax.set_xlabel('Mean inner-fold cold-start training time (seconds)')
        ax.set_ylabel('Inner-validation balanced accuracy (%)')
        ax.set_title(subtitle,loc='left',fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4)) if ax.get_xscale()=='linear' else None
        tidy(ax)
    fig.subplots_adjust(left=.09,right=.98,wspace=.24,top=.82,bottom=.19)
    title(fig,'Figure 8 | Is extra computation justified by better validation accuracy?',
          'Each point is an inner-search candidate. Orange ring = most frequently selected. '
          'Higher is better; lower training time is better. Times include generator fitting.')
    fig.savefig(OUT_DIR/'08_accuracy_vs_computation.pdf',bbox_inches='tight');plt.close(fig)


def latex_table(df,filename,caption):
    # Write a minimal booktabs table without hidden pandas dependency.
    cols=list(df.columns)
    code=['\\begin{table*}[t]','\\centering','\\caption{'+caption.replace('%','\\%')+'}',
          '\\footnotesize','\\begin{tabular}{'+'l'+'r'*(len(cols)-1)+'}',
          '\\toprule',' & '.join(c.replace('_','\\_') for c in cols)+' \\\\', '\\midrule']
    for tup in df.itertuples(index=False,name=None):
        items=[]
        for cell in tup:
            if pd.isna(cell): items.append('--')
            elif isinstance(cell,(float,np.floating)):items.append(f'{cell:.3f}')
            else:items.append(str(cell).replace('_','\\_').replace('%','\\%').replace('&','\\&'))
        code.append(' & '.join(items)+' \\\\')
    code+=['\\bottomrule','\\end{tabular}','\\end{table*}']
    (OUT_DIR/filename).write_text('\n'.join(code)+'\n',encoding='utf-8')


def run_experiment():
    started=time.perf_counter()
    seeded(SEED);torch.set_num_threads(min(3,torch.get_num_threads()))
    if INPUT_POINTS<64 or INPUT_POINTS%32!=0:
        raise ValueError('INPUT_POINTS must be divisible by 32 and >=64')
    wn,X,y,patient=load_data()
    tested=np.arange(len(X))
    if FAST:
        rg=np.random.default_rng(SEED)
        tested=np.concatenate([rg.choice(np.where(y==cl)[0],size=OUTER_QUICK_PER_CLASS,
                                         replace=False) for cl in (0,1)])
    if min(np.bincount(y,minlength=2))-1 < INNER_SPLITS:
        raise ValueError('Too few training patients for requested inner folds')
    augment_search=[];joint_search=[];real_search=[];choices=[];outputs=[];costs=[]
    example=None
    pairs=[m+' || '+r for m in ACTIVE_MODELS for r in ACTIVE_REPS]
    for fold,(train,test) in enumerate(LeaveOneOut().split(X),1):
        i=int(test[0])
        if i not in set(tested): continue
        groups=get_inner_records(X,y,train,fold)
        # Stage A: augmentation before any model/feature selection.
        ar,_=evaluate_grid(groups,ACTIVE_AUG,'augmentation')
        choice_a,astats=pick_winner(ACTIVE_AUG,ar)
        # Report the best synthetic method separately without forcing its selection.
        synthetic=[m for m in ACTIVE_AUG if m!='Real only']
        forced_a,_=pick_winner(synthetic,ar)
        for r in ar: augment_search.append({**r,'outer_fold':fold})
        choices.append(dict(fold=fold,patient=patient[i],stage='augmentation',
                            candidate=choice_a,InnerBA=astats[choice_a]['BA']))
        # Stage B: classifier * feature under the selected augmentation.
        jr,_=evaluate_grid(groups,pairs,'joint',augmentation=choice_a)
        choice_pair,jstats=pick_winner(pairs,jr)
        best_model,best_rep=choice_pair.split(' || ',1)
        for r in jr: joint_search.append({**r,'outer_fold':fold})
        choices.append(dict(fold=fold,patient=patient[i],stage='joint',
                            candidate=choice_pair,InnerBA=jstats[choice_pair]['BA']))
        # Fair modern REAL-ONLY alternatives: individually choose feature per model.
        rr=([{**r,'stage':'realonly'} for r in jr] if choice_a=='Real only'
            else evaluate_grid(groups,pairs,'realonly',baseline=True)[0])
        for r in rr: real_search.append({**r,'outer_fold':fold})
        baseline_pairs={}
        for m in ACTIVE_MODELS:
            subset=[p for p in pairs if p.startswith(m+' || ')]
            baseline_pairs[m]=pick_winner(subset,rr)[0].split(' || ',1)[1]
        # Stage C: independent original outer-patient evaluation, once per arm.
        outer=FoldData(X[train],y[train],SEED+fold*1009)
        # The forced-synthetic contrast is reported even when the safer
        # deployment choice is no augmentation. The real-only arm is matched.
        arms=[('best_synthetic',forced_a,best_model,best_rep),
              ('matched_real_only','Real only',best_model,best_rep),
              ('final_cv_selected',choice_a,best_model,best_rep)]
        arms +=[(f'real_only::{m}','Real only',m,baseline_pairs[m]) for m in ACTIVE_MODELS]
        cache={}
        for name,aug,mod,rep in arms:
            key=(aug,mod,rep)
            if key not in cache:
                xx,yy,gen_s=outer.features(aug,rep)
                scores,clf_s=fit_predict(mod,xx,yy,representation(X[test],rep),
                                         SEED+fold*1009+19,True)
                cache[key]=(float(scores[0]),clf_s+gen_s,clf_s,gen_s)
            score,seconds,clf_s,gen_s=cache[key]
            outputs.append(dict(fold=fold,patient=patient[i],truth=int(y[i]),
                arm=name,augmentation=aug,classifier=mod,feature=rep,
                score=score,pred=int(score>=.5)))
            costs.append(dict(fold=fold,arm=name,Seconds=seconds,
                              ModelSeconds=clf_s,GeneratorSeconds=gen_s))
        if example is None:
            method,count=split_aug(forced_a)
            synth,ys=outer.generators[method].sample(count)
            example=(outer.real,outer.y,synth,ys,method)
        print(f'[{RUN_MODE}] fold={fold} patient={patient[i]} generator={choice_a} '
              f'model={best_model} representation={best_rep}',flush=True)
    pred=pd.DataFrame(outputs)
    select=pd.DataFrame(choices)
    allsearch=augment_search+joint_search+real_search
    selected_aug=summarize_search(augment_search,choices,'augmentation',ACTIVE_AUG)
    selected_joint=summarize_search(joint_search,choices,'joint',pairs)
    real_only=summarize_search(real_search,choices,'realonly',pairs)
    summaries=pd.concat([selected_aug.assign(stage='augmentation'),
                         selected_joint.assign(stage='joint'),
                         real_only.assign(stage='realonly')],ignore_index=True)
    benchmark=pd.DataFrame([{'arm':arm,**metrics(d)} for arm,d in pred.groupby('arm')])
    effects=paired(pred)
    for fn,df in [('inner_augmentation_summary.csv',selected_aug),
                  ('inner_joint_modern_summary.csv',selected_joint),
                  ('inner_realonly_baselines.csv',real_only),
                  ('outer_patient_predictions.csv',pred),
                  ('outer_patient_metrics.csv',benchmark),
                  ('paired_original_patient_effects.csv',effects),
                  ('selected_configurations.csv',select),
                  ('outer_computational_cost.csv',pd.DataFrame(costs))]:
        df.to_csv(OUT_DIR/fn,index=False)
    # Flatten variable-sized patient arrays before writing detailed search results.
    long=[]
    for r in allsearch:
        for k,(truth,score) in enumerate(zip(r['truth'],r['score'])):
            long.append(dict(stage=r['stage'],outer_fold=r['outer_fold'],
                inner_fold=r['inner_fold'],candidate=r['candidate'],patient_in_inner_fold=k,
                truth=int(truth),score=float(score),seconds=float(r['seconds'])))
    pd.DataFrame(long).to_csv(OUT_DIR/'inner_patient_candidate_predictions.csv',index=False)
    latex_table(selected_aug.sort_values('InnerBA_mean',ascending=False),
                '01_augmentation_table.tex','Stage 1: fixed-probe generator selection on original-patient inner CV.')
    latex_table(selected_joint.sort_values('InnerBA_mean',ascending=False),
                '02_joint_ablation_table.tex','Stage 2: joint classifier--representation ablation after generator selection.')
    latex_table(benchmark,'03_outer_benchmark_table.tex',
                'Outer held-out original-patient performance; only training folds contain generated data.')
    latex_table(effects,'04_paired_effects_table.tex',
                'Exploratory paired original-patient changes in balanced accuracy relative to modern real-only baselines.')
    fig1_augmentation(selected_aug,select)
    fig2_joint(selected_joint)
    fig3_benchmark(benchmark)
    fig4_patient_changes(pred)
    fig5_effects(effects)
    fig6_roc(pred)
    fig7_spectra(example,wn)
    fig8_resources(summaries)
    config=dict(run_mode=RUN_MODE,patients=int(len(y)),tested=int(len(tested)),
                healthy=int((y==0).sum()),sick=int((y==1).sum()),
                channels=int(X.shape[1]),seed=SEED,device=str(DEVICE),
                input_points=INPUT_POINTS,epochs=EPOCHS,generator_steps=GEN_STEPS,
                inner_folds=INNER_SPLITS,
                generation_candidates=ACTIVE_AUG,models=ACTIVE_MODELS,
                representations=ACTIVE_REPS,wall_seconds=time.perf_counter()-started,
                protocol='nested LOOCV outer; StratifiedKFold inner using ONLY original patients',
                augmentation_probe='locked Patch Transformer + SNV raw, no classifier tuning at augmentation stage',
                selection='inner original-patient balanced accuracy, then AUC, accuracy, fixed candidate order',
                no_augmentation_control='same fold-specific classifier and representation',
                best_synthetic='best of synthetic candidates using probe; no guarantee better than real-only',
                caveat='17 independent patients. Exploratory; no external validation. Architecture families are compact approximations, not official paper reproductions.')
    (OUT_DIR/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (OUT_DIR/'FIGURE_GUIDE.md').write_text('''# Figure interpretation guide

1. **Augmentation probe gain** — Inner-CV balanced accuracy difference from no augmentation, using the SAME fixed Patch Transformer on original-patient validation folds. Not test performance.
2. **Joint ablation heatmap** — Once augmentation is chosen, mean inner-CV balanced accuracy for each neural classifier and spectral representation. Selection count is not accuracy.
3. **Outer held-out balanced accuracy** — Top two methods have IDENTICAL classifier and features: real only versus best synthetic (even if real only was selected by the probe). The third arm is the preferred real/synthetic option. All remaining modern methods are real-only.
4. **Paired patient error transitions** — Shows how many original patients change correctness as a result of augmentation (recovered vs newly introduced errors).
5. **Paired effect sizes** — Forced best-synthetic versus paired real-only alternatives (and the preferred CV choice). Point and exploratory 95% bootstrap interval. CSV also includes exploratory exact class-balanced paired sign-flip and exact McNemar tests, Holm corrected. These tests are low-power and fold fits overlap; not proof of superiority with 17 patients.
6. **ROC and precision–recall** — Secondary descriptive views of held-out discrimination; separate models across folds are not probability calibrated.
7. **Spectral plausibility** — Real versus best-synthetic class-conditional median and 10–90% range on ONE outer training fold; never held-out data. Always plotted for the best-synthetic candidate; it may lose to the real-only option.
8. **Accuracy vs compute** — Training-only inner-CV score versus standalone cold-start time, generator fit included.

Critical controls: no held-out original patient participates in generation, classifier fit, representation construction, augmentation selection, or model/feature selection. Synthetic examples never count as independent held-out patients. Quick mode is a smoke test. With 17 patients all inferential claims require independent external validation.
''',encoding='utf-8')
    print(f'Done. Outer patients={len(tested)}; outputs={OUT_DIR}',flush=True)
    return pred,benchmark,effects

if __name__=='__main__': run_experiment()
