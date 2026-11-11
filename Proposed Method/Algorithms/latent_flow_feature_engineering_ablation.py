"""Leakage-safe FEATURE-ENGINEERING ablation with conditional Latent Flow always ON.

Input beside this script: doentes.zip, saudaveis.zip.
Every outer leave-one-patient-out fold:
  1) fit a NEW PCA + class-conditional latent-flow generator on the remaining 16 originals;
  2) generate 30 healthy + 30 sick spectra (using the same 100/class pool across variants);
  3) train EVERY classifier on real training spectra PLUS those synthetic spectra;
  4) fit all feature transformations on training-fold data only;
  5) predict the single held-out ORIGINAL patient.
No full-data synthetic archive is used in the cross-validated evaluation.

Outputs: metrics, patient predictions, paired-bootstrap uncertainty, explanatory PDF figures,
full-fit model and its fold-independent synthetic training spectra.

pip install numpy pandas scipy scikit-learn torch matplotlib cloudpickle
python latent_flow_feature_engineering_ablation.py

NB: Only 17 independent patients; results are exploratory, and choosing a best pipeline
from LOOCV scores creates optimistic selection bias. Not for clinical diagnosis.
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
from scipy.signal import savgol_filter, detrend
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn
from torch.nn import functional as F
import cloudpickle

ROOT = Path(globals().get('ROOT', Path(__file__).resolve().parent))
OUTPUT_DIR = Path(globals().get('OUTPUT_DIR', ROOT / 'latent_flow_feature_ablation'))
HEALTHY_ZIP = Path(globals().get('HEALTHY_ZIP', ROOT / 'saudaveis.zip'))
SICK_ZIP = Path(globals().get('SICK_ZIP', ROOT / 'doentes.zip'))
SEED = int(globals().get('SEED', 42))
LATENT_DIM = int(globals().get('LATENT_DIM', 4))
FLOW_TRAIN_STEPS = int(globals().get('FLOW_TRAIN_STEPS', 350))
FLOW_BATCH_SIZE = int(globals().get('FLOW_BATCH_SIZE', 64))
SYNTHETIC_PER_CLASS = int(globals().get('SYNTHETIC_PER_CLASS', 30))
# Keep the generation procedure identical across feature configurations.
SYNTHETIC_POOL_PER_CLASS = int(globals().get('SYNTHETIC_POOL_PER_CLASS', 100))
OCCLUSION_BANDS = int(globals().get('OCCLUSION_BANDS', 32))
BOOTSTRAP_REPS = int(globals().get('BOOTSTRAP_REPS', 2500))
TRAIN_FINAL_MODEL = bool(globals().get('TRAIN_FINAL_MODEL', True))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(min(3, torch.get_num_threads()))
plt.rcParams.update({'figure.facecolor':'white', 'axes.facecolor':'white',
    'savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':False,'font.size':9,'pdf.fonttype':42})

# All feature pipelines include the same 30/class Latent Flow spectra.
FEATURES = [
    'SNV',
    'SG smoothing + SNV',
    'SG first derivative',
    'SG second derivative',
    'Detrend + SNV',
    'MSC',
    'PCA (4 components)',
    'PCA (8 components)',
    'ANOVA top 64',
    'ANOVA top 256',
    '32-band statistics',
    'SNV + derivative fusion',
    'FFT magnitude (64)',
]
MODELS = ['PLS-DA', 'RBF SVM', 'Linear SVM']
COLOR = {'PLS-DA':'#315b71','RBF SVM':'#8a5938','Linear SVM':'#63825b'}


def load_spectra():
    rows=[]
    for archive, label in [(HEALTHY_ZIP,0),(SICK_ZIP,1)]:
        if not archive.exists(): raise FileNotFoundError(archive)
        with ZipFile(archive) as z:
            for fname in sorted(z.namelist()):
                if not fname.lower().endswith('.csv') or '.ipynb_checkpoints' in fname.lower() or '__macosx' in fname.lower():
                    continue
                content=z.read(fname)
                head=content.splitlines()[0]
                sep=';' if head.count(b';')>head.count(b',') else ','
                a=pd.read_csv(BytesIO(content),sep=sep,encoding='utf-8-sig')
                if a.shape[1]!=2: raise ValueError(f'Not a two-column spectrum: {fname}')
                w=a.iloc[:,0].to_numpy(dtype=float)
                v=a.iloc[:,1].to_numpy(dtype=float)
                if not (np.isfinite(w).all() and np.isfinite(v).all()): raise ValueError(fname)
                rows.append((Path(fname).stem,int(label),w,v))
    if not rows: raise ValueError('No spectra in input archives')
    grid=rows[0][2]
    for name,_,w,v in rows:
        if len(w)!=len(grid) or not np.allclose(grid,w,atol=1e-6,rtol=0):
            raise ValueError(f'Wavenumber mismatch: {name}')
    names=np.asarray([r[0] for r in rows]);y=np.asarray([r[1] for r in rows],dtype=int)
    X=np.asarray([r[3] for r in rows],dtype=float)
    if len(set(names))!=len(names): raise ValueError('Duplicate patient names')
    return grid,X,y,names


def snv(x):
    x=np.atleast_2d(np.asarray(x,dtype=float))
    return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-10)


class ConditionalFlow(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d+7,64),nn.SiLU(),nn.Linear(64,64),
                               nn.SiLU(),nn.Linear(64,d))
    def forward(self,x,c,t):
        q=2*np.pi*t
        tt=[t,torch.sin(q),torch.cos(q),torch.sin(2*q),torch.cos(2*q)]
        return self.net(torch.cat([x,F.one_hot(c.long(),2).float()]+tt,dim=1))


def generate_flow(Xreal,yreal,seed,n_per_class,train_steps=FLOW_TRAIN_STEPS):
    """PCA and generator refit strictly on the given real training patients."""
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    z_real=snv(Xreal)
    dim=min(LATENT_DIM,len(Xreal)-2)
    pca=PCA(n_components=dim,random_state=seed)
    std=StandardScaler()
    z=std.fit_transform(pca.fit_transform(z_real)).astype(np.float32)
    targets=torch.as_tensor(z,dtype=torch.float32,device=DEVICE)
    labels=torch.as_tensor(yreal,dtype=torch.long,device=DEVICE)
    pools=[torch.where(labels==k)[0] for k in (0,1)]
    if min(map(len,pools))<2: raise ValueError('Need at least two real patients per class')
    net=ConditionalFlow(dim).to(DEVICE)
    optim=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    for step in range(train_steps):
        cc=torch.randint(0,2,(FLOW_BATCH_SIZE,),device=DEVICE)
        idx=torch.empty_like(cc)
        for k in (0,1):
            ok=cc==k
            idx[ok]=pools[k][torch.randint(len(pools[k]),(int(ok.sum()),),device=DEVICE)]
        x1=targets[idx]
        x0=torch.randn_like(x1)
        t=torch.rand((len(cc),1),device=DEVICE)
        xt=(1-t)*x0+t*x1
        loss=F.mse_loss(net(xt,cc,t),x1-x0)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    yy=np.repeat([0,1],n_per_class).astype(int)
    class_tensor=torch.as_tensor(yy,dtype=torch.long,device=DEVICE)
    net.eval()
    with torch.no_grad():
        xt=torch.randn((len(yy),dim),device=DEVICE)
        nsteps=32; dt=1/nsteps
        for tstep in range(nsteps):
            ta=torch.full((len(yy),1),tstep/nsteps,device=DEVICE)
            tb=torch.full((len(yy),1),(tstep+1)/nsteps,device=DEVICE)
            v0=net(xt,class_tensor,ta)
            v1=net(xt+v0*dt,class_tensor,tb)
            xt=xt+0.5*(v0+v1)*dt
        zg=xt.cpu().numpy()
    Xnew=snv(pca.inverse_transform(std.inverse_transform(zg)))
    if not np.isfinite(Xnew).all(): raise FloatingPointError('Invalid flow output')
    return Xnew,yy


class FeatureEngineer:
    """Transforms SNV spectra. Stateful transforms fit ONLY on training fold."""
    def __init__(self, variant, wavenumbers):
        self.variant=variant
        self.wn=np.asarray(wavenumbers,dtype=float)
        self.reducer=None
    def fit(self,Xtrain,ytrain,Xreal=None):
        mode=self.variant
        if mode.startswith('PCA ('):
            d=int(mode.split('(')[1].split(' ')[0]); d=min(d,len(Xtrain)-2,Xtrain.shape[1])
            self.reducer=PCA(n_components=d,random_state=SEED).fit(Xtrain)
        elif mode.startswith('ANOVA top'):
            d=int(mode.split()[-1]);d=min(d,Xtrain.shape[1])
            self.reducer=SelectKBest(score_func=f_classif,k=d)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.reducer.fit(Xtrain,ytrain)
        elif mode=='MSC':
            # Reference is computed exclusively from real patients in this fold.
            self.reference=np.mean(Xreal if Xreal is not None else Xtrain,axis=0)
            self.reference=self.reference.astype(float)
            self.ref_mean=float(self.reference.mean())
            self.ref_center=self.reference-self.ref_mean
            self.denom=float(np.sum(self.ref_center**2))+1e-12
        return self
    def transform(self,X):
        X=np.atleast_2d(np.asarray(X,dtype=float))
        mode=self.variant
        if mode=='SNV': return X.copy()
        if mode=='SG smoothing + SNV':return snv(savgol_filter(X,15,2,axis=1))
        if mode=='SG first derivative':
            delta=float(np.median(np.abs(np.diff(self.wn))))
            return snv(savgol_filter(X,21,3,deriv=1,delta=delta,axis=1))
        if mode=='SG second derivative':
            delta=float(np.median(np.abs(np.diff(self.wn))))
            return snv(savgol_filter(X,21,3,deriv=2,delta=delta,axis=1))
        if mode=='Detrend + SNV':return snv(detrend(X,axis=1,type='linear'))
        if mode=='MSC':
            means=X.mean(1,keepdims=True)
            slopes=((X-means)@self.ref_center/self.denom).reshape(-1,1)
            slopes=np.where(slopes>0.1,slopes,0.1)
            intercepts=means-slopes*self.ref_mean
            return (X-intercepts)/slopes
        if mode.startswith('PCA ('):return self.reducer.transform(X)
        if mode.startswith('ANOVA top'):return self.reducer.transform(X)
        if mode=='32-band statistics':
            # 32 contiguous wavenumber regions; local mean, SD, RMS gradient, peak span.
            features=[]
            for ix in np.array_split(np.arange(X.shape[1]),32):
                x=X[:,ix]
                dx=np.diff(x,axis=1)
                features += [x.mean(1),x.std(1),np.sqrt(np.mean(dx*dx,axis=1)),x.max(1)-x.min(1)]
            return np.column_stack(features)
        if mode=='SNV + derivative fusion':
            delta=float(np.median(np.abs(np.diff(self.wn))))
            first=snv(savgol_filter(X,21,3,deriv=1,delta=delta,axis=1))
            return np.concatenate([X,first],axis=1)
        if mode=='FFT magnitude (64)':
            return np.log1p(np.abs(np.fft.rfft(X,axis=1)[:,1:65]))
        raise ValueError(mode)


class PLSDA:
    def fit(self,X,y):
        n_components=min(4,len(X)-2,X.shape[1])
        self.model=PLSRegression(n_components=n_components,scale=True)
        self.model.fit(X,y)
        return self
    def decision_function(self,X):return np.asarray(self.model.predict(X)).ravel()-0.5


def new_classifier(name):
    if name=='PLS-DA': return PLSDA()
    if name=='RBF SVM':return make_pipeline(StandardScaler(),SVC(C=1,gamma='scale',kernel='rbf',class_weight='balanced'))
    if name=='Linear SVM':return make_pipeline(StandardScaler(),SVC(C=1,kernel='linear',class_weight='balanced'))
    raise ValueError(name)


def score_classifer(model,X):
    return np.asarray(model.decision_function(X),dtype=float).reshape(-1)


def score_metrics(df):
    yy=df['true'].to_numpy(dtype=int)
    pp=df['pred'].to_numpy(dtype=int)
    sc=df['score'].to_numpy(dtype=float)
    tn,fp,fn,tp=confusion_matrix(yy,pp,labels=[0,1]).ravel()
    return {'n_patients':len(yy),'accuracy':accuracy_score(yy,pp),
        'balanced_accuracy':balanced_accuracy_score(yy,pp),
        'sensitivity':tp/max(1,tp+fn), 'specificity':tn/max(1,tn+fp),
        'roc_auc':roc_auc_score(yy,sc),
        'TP':int(tp),'FN':int(fn),'TN':int(tn),'FP':int(fp)}


def train_variant(mode,clf,wn,Xtr,ys,Xte,Xreal):
    engineer=FeatureEngineer(mode,wn).fit(Xtr,ys,Xreal=Xreal)
    A=engineer.transform(Xtr)
    B=engineer.transform(Xte)
    mdl=new_classifier(clf)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mdl.fit(A,ys)
    return engineer,mdl,score_classifer(mdl,B)


wn, X, y, patients=load_spectra()
assert len(X)>=5 and len(np.unique(y))==2
assert SYNTHETIC_PER_CLASS>=1 and SYNTHETIC_POOL_PER_CLASS>=SYNTHETIC_PER_CLASS
print(f'Inputs: {len(y)} original patients, {np.sum(y==0)} healthy, '
      f'{np.sum(y==1)} sick, {len(wn)} wavenumbers',flush=True)
print(f'ALL {len(FEATURES)*len(MODELS)} pipelines use '
      f'{SYNTHETIC_PER_CLASS} Latent Flow samples/class per training fold',flush=True)

rows=[]
# Save data needed to re-fit the post-hoc selected model for the occlusion plot.
fold_cache=[]
for fold,(train,held) in enumerate(LeaveOneOut().split(X),start=1):
    foldseed=SEED+fold*1031
    sX,sy=generate_flow(X[train],y[train],foldseed,SYNTHETIC_POOL_PER_CLASS)
    keep=(sy==0)&(np.cumsum(sy==0)<=SYNTHETIC_PER_CLASS) | (sy==1)&(np.cumsum(sy==1)<=SYNTHETIC_PER_CLASS)
    sX,sy=sX[keep],sy[keep]
    original_train=snv(X[train]); held_orig=snv(X[held])
    train_all=np.vstack([original_train,sX]); train_labels=np.concatenate([y[train],sy])
    fold_cache.append((fold,train,held,original_train,held_orig,train_all,train_labels))
    for mode in FEATURES:
        eng=FeatureEngineer(mode,wn).fit(train_all,train_labels,Xreal=original_train)
        AA=eng.transform(train_all)
        BB=eng.transform(held_orig)
        for clf in MODELS:
            model=new_classifier(clf)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(AA,train_labels)
            score=float(score_classifer(model,BB)[0])
            rows.append({'fold':fold,'patient':patients[held[0]],'true':int(y[held[0]]),
                         'feature_method':mode,'classifier':clf,'pred':int(score>=0),
                         'score':score,'n_real_training':len(train),'n_synthetic_training':len(sy)})
    print(f'{fold:02d}/{len(X)} {patients[held[0]]}: completed',flush=True)

pred=pd.DataFrame(rows)
pred.to_csv(OUTPUT_DIR/'feature_ablation_patient_predictions.csv',index=False)
metrics=[]
for (mode,clf),data in pred.groupby(['feature_method','classifier'],sort=False):
    metrics.append({'feature_method':mode,'classifier':clf,
                    'training':'Original patients + fold-specific Latent Flow',
                    'synthetic_per_class':SYNTHETIC_PER_CLASS,**score_metrics(data)})
results=pd.DataFrame(metrics)
results['delta_bal_acc_vs_snv_pp']=0.
for clf in MODELS:
    base=float(results[(results.classifier==clf)&(results.feature_method=='SNV')].balanced_accuracy.iloc[0])
    results.loc[results.classifier==clf,'delta_bal_acc_vs_snv_pp']=(results.loc[results.classifier==clf,'balanced_accuracy']-base)*100
results.to_csv(OUTPUT_DIR/'feature_ablation_metrics.csv',index=False)

# Post-hoc selection for visualization; NOT an unbiased estimate of model selection.
# Primary comparison focuses on PLS-DA, the strongest augmented model in the prior study.
pls=results[results.classifier=='PLS-DA'].copy()
pls['rank']=pls.feature_method.map({v:i for i,v in enumerate(FEATURES)})
pls=pls.sort_values(['balanced_accuracy','roc_auc','rank'],ascending=[False,False,True])
best=pls.iloc[0]['feature_method']
ref='SNV'
print(f'Best PLS-DA feature pipeline (post-hoc): {best}',flush=True)

# Patient-stratified, paired bootstrap of changes vs reference using the same held-out patients.
rng=np.random.default_rng(SEED+926)
bootstrap=[]
for model_name in MODELS:
    dref=pred[(pred.classifier==model_name)&(pred.feature_method==ref)].sort_values('fold')
    ytrue=dref.true.to_numpy(int)
    neg=np.where(ytrue==0)[0];pos=np.where(ytrue==1)[0]
    draws=np.column_stack([np.concatenate([rng.choice(neg,size=len(neg),replace=True),
                                            rng.choice(pos,size=len(pos),replace=True)])
                           for _ in range(BOOTSTRAP_REPS)]).T
    def bs_ba(a):
        correct=(a==ytrue).astype(float)
        return 0.5*(correct[draws[:,:len(neg)]].mean(1)+correct[draws[:,len(neg):]].mean(1))
    base_ba=bs_ba(dref.pred.to_numpy(int))
    for v in FEATURES:
        dt=pred[(pred.classifier==model_name)&(pred.feature_method==v)].sort_values('fold')
        dv=bs_ba(dt.pred.to_numpy(int))-base_ba
        bootstrap.append({'classifier':model_name,'feature_method':v,
            'delta_ba_pp':float(dv.mean()*100),
            'delta_ba_ci95_low_pp':float(np.quantile(dv,0.025)*100),
            'delta_ba_ci95_high_pp':float(np.quantile(dv,0.975)*100)})
boot=pd.DataFrame(bootstrap)
boot.to_csv(OUTPUT_DIR/'paired_bootstrap_feature_differences.csv',index=False)

# Plot 1: 13x3 heatmap with exact percentages.
heat=results.pivot(index='feature_method',columns='classifier',values='balanced_accuracy').loc[FEATURES,MODELS]*100
fig,ax=plt.subplots(figsize=(8.8,7.3))
im=ax.imshow(heat.to_numpy(),cmap='Blues',vmin=45,vmax=100,aspect='auto')
ax.set_xticks(np.arange(len(MODELS)),labels=MODELS)
ax.set_yticks(np.arange(len(FEATURES)),labels=FEATURES)
for i in range(len(FEATURES)):
    for j in range(len(MODELS)):
        value=float(heat.iloc[i,j]);ax.text(j,i,f'{value:.1f}',ha='center',va='center',
                                        fontsize=8,color='white' if value>83 else 'black')
ax.set_xlabel('Classifier (always trained with synthetic spectra)')
cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.02);cb.set_label('Balanced accuracy (%)')
fig.tight_layout();fig.savefig(OUTPUT_DIR/'01_feature_ablation_heatmap.pdf',bbox_inches='tight');plt.close(fig)

# Plot 2: changes vs SNV, all using augmented training.
fig,axs=plt.subplots(1,3,figsize=(15.5,6.4),sharey=True)
pos=np.arange(len(FEATURES))
for ax,clf in zip(axs,MODELS):
    rr=results[results.classifier==clf].set_index('feature_method').loc[FEATURES]
    diff=rr.delta_bal_acc_vs_snv_pp.to_numpy()
    ax.barh(pos,diff,height=.64,color=np.where(diff>=0,'#427b84','#aa7468'))
    ax.axvline(0,color='black',lw=0.9)
    ax.set_yticks(pos,labels=FEATURES)
    ax.invert_yaxis();ax.set_title(clf)
    ax.set_xlabel('Change in balanced accuracy (pp)')
    lim=max(18,1.3*max(abs(diff.min()),abs(diff.max())))
    ax.set_xlim(-lim,lim)
    for ix,di in enumerate(diff):
        if abs(di)>0.15:ax.text(di+(.6 if di>0 else -.6),ix,f'{di:+.1f}',
                                va='center',ha=('left' if di>0 else 'right'),fontsize=7)
fig.tight_layout();fig.savefig(OUTPUT_DIR/'02_feature_ablation_deltas.pdf',bbox_inches='tight');plt.close(fig)

# Plot 3: trade-off between sensitivity and specificity for PLS-DA with all variants.
fig,ax=plt.subplots(figsize=(7,6))
for _,r in results[results.classifier=='PLS-DA'].iterrows():
    ax.scatter(r.specificity*100,r.sensitivity*100,s=45,
               facecolor='#3b7584' if r.feature_method==best else '#999999',
               edgecolor='white',linewidth=.65)
    if r.feature_method in [best,ref,'SG first derivative','ANOVA top 64','32-band statistics']:
        ax.annotate(r.feature_method,(r.specificity*100,r.sensitivity*100),xytext=(4,4),
                    textcoords='offset points',fontsize=8)
ax.set_xlabel('Specificity (healthy class, %)')
ax.set_ylabel('Sensitivity (sick class, %)')
ax.set_xlim(-4,105);ax.set_ylim(-4,105)
fig.tight_layout();fig.savefig(OUTPUT_DIR/'03_sensitivity_specificity.pdf',bbox_inches='tight');plt.close(fig)

# Plot 4: ROC and confusion for SNV vs post-hoc best features (same 17 held-out originals).
fig,axes=plt.subplots(2,2,figsize=(9.2,7.9))
for col,v in enumerate([ref,best]):
    df=pred[(pred.classifier=='PLS-DA')&(pred.feature_method==v)].sort_values('fold')
    cm=confusion_matrix(df.true,df.pred,labels=[0,1])
    ax=axes[0,col]
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=10)
    ax.set_xticks([0,1],labels=['Healthy','Sick']);ax.set_yticks([0,1],labels=['Healthy','Sick'])
    ax.set_xlabel('Predicted');ax.set_ylabel('True')
    ax.set_title(v+ (' (post-hoc)' if v!=ref else ' (reference)'))
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=17,
                                 color='white' if cm[i,j]>=6 else 'black')
    fpr,tpr,_=roc_curve(df.true,df.score)
    ax=axes[1,col]
    ax.plot(fpr,tpr,color='#315b71',linewidth=1.8,
            label=f'AUC {roc_auc_score(df.true,df.score):.3f}')
    ax.plot([0,1],[0,1],ls='--',lw=1,color='0.65')
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate')
    ax.legend(loc='lower right',frameon=False)
fig.tight_layout();fig.savefig(OUTPUT_DIR/'04_plsda_confusion_roc.pdf',bbox_inches='tight');plt.close(fig)

# Plot 5: explainability via contiguous spectral-band occlusion of held-out originals,
# without using the true patient in training of either generator or classifier.
bands=np.array_split(np.arange(len(wn)),OCCLUSION_BANDS)
occlusion=[]
selection=[]
for fold,train,held,Xreal,Xheld,Xaug,yaug in fold_cache:
    for v in dict.fromkeys([ref,best]):
        eng=FeatureEngineer(v,wn).fit(Xaug,yaug,Xreal=Xreal)
        a=eng.transform(Xaug)
        model=new_classifier('PLS-DA');model.fit(a,yaug)
        base=float(score_classifer(model,eng.transform(Xheld))[0])
        true=int(y[held[0]])
        training_mean=Xreal.mean(axis=0)
        for bi,idx in enumerate(bands):
            xocc=Xheld.copy()
            # Smooth raised cosine limits discontinuities at occluded band boundaries.
            blend=np.ones(len(idx),dtype=float)
            taper=min(8,max(1,len(idx)//5))
            ramp=(1-np.cos(np.linspace(0,np.pi,taper)))/2
            blend[:taper]=ramp;blend[-taper:]=ramp[::-1]
            xocc[:,idx]=(1-blend)*xocc[:,idx]+blend*training_mean[idx]
            val=float(score_classifer(model,eng.transform(xocc))[0])
            occlusion.append({'fold':fold,'patient':patients[held[0]],'true':true,
                'feature_method':v,'band':bi,'start_wn':float(wn[idx[0]]),
                'end_wn':float(wn[idx[-1]]),
                'absolute_score_change':abs(base-val),
                'true_class_margin_change':float((2*true-1)*(base-val))})
    # Frequency with which the selected ANOVA top-256 pipeline selects each original point.
    feat=FeatureEngineer('ANOVA top 256',wn).fit(Xaug,yaug,Xreal=Xreal)
    mask=feat.reducer.get_support()
    for k in np.flatnonzero(mask):
        selection.append({'fold':fold,'wavenumber_index':int(k),'wavenumber':float(wn[k])})

occ=pd.DataFrame(occlusion);occ.to_csv(OUTPUT_DIR/'spectral_occlusion.csv',index=False)
select=pd.DataFrame(selection);select.to_csv(OUTPUT_DIR/'anova_selected_features_by_fold.csv',index=False)
fig,axs=plt.subplots(2,1,figsize=(12,6.5),sharex=True,gridspec_kw={'height_ratios':[1.2,1]})
ax=axs[0]
ax.plot(wn,snv(X[y==0]).mean(0),color='#457ba0',lw=1.4,label='Healthy mean (real patients)')
ax.plot(wn,snv(X[y==1]).mean(0),color='#a35f44',lw=1.4,label='Sick mean (real patients)')
ax.set_ylabel('SNV intensity');ax.legend(loc='upper left',frameon=False)
ax=axs[1]
for v,color in [(ref,'#6c6c6c')]+([(best,'#315b71')] if best!=ref else []):
    gg=occ[occ.feature_method==v].groupby('band',sort=True).absolute_score_change.mean()
    val=gg.to_numpy()
    val=val/(val.max()+1e-12)
    centers=np.array([np.mean(wn[b]) for b in bands])
    ax.plot(centers,val,lw=1.7,marker='o',ms=2.7,label=v,color=color)
ax.set_xlim(float(wn[0]),float(wn[-1]))
ax.set_ylabel('Relative mean occlusion sensitivity')
ax.set_xlabel('Wavenumber (cm$^{-1}$)')
ax.legend(loc='upper left',frameon=False)
fig.tight_layout();fig.savefig(OUTPUT_DIR/'05_spectral_occlusion.pdf',bbox_inches='tight');plt.close(fig)

# Plot 6: fold stability of ANOVA top-256 feature selection and spectral patient variation.
frequency=np.bincount(select.wavenumber_index.to_numpy(int),minlength=len(wn))/len(X)
fig,(ax0,ax1)=plt.subplots(2,1,figsize=(11.8,6.6),sharex=True,
                             gridspec_kw={'height_ratios':[1.2,1]})
ax0.plot(wn,snv(X[y==0]).mean(0),color='#457ba0',lw=1.35,label='Healthy (mean)')
ax0.plot(wn,snv(X[y==1]).mean(0),color='#a35f44',lw=1.35,label='Sick (mean)')
ax0.legend(frameon=False,loc='upper left');ax0.set_ylabel('SNV intensity')
ax1.plot(wn,frequency,color='#315b71',lw=1.0)
ax1.set_ylabel('ANOVA top-256 selection frequency')
ax1.set_xlabel('Wavenumber (cm$^{-1}$)');ax1.set_ylim(0,1.04)
fig.tight_layout();fig.savefig(OUTPUT_DIR/'06_feature_selection_stability.pdf',bbox_inches='tight');plt.close(fig)

# Plot 7: uncertainty in differences vs SNV for PLS-DA (patient-stratified paired bootstrap).
sub=boot[boot.classifier=='PLS-DA'].set_index('feature_method').loc[FEATURES].reset_index()
fig,ax=plt.subplots(figsize=(9.8,6.2))
idx=np.arange(len(sub));delta=sub.delta_ba_pp.to_numpy()
errors=np.vstack([delta-sub.delta_ba_ci95_low_pp.to_numpy(),sub.delta_ba_ci95_high_pp.to_numpy()-delta])
ax.errorbar(delta,idx,xerr=errors,fmt='o',color='#315b71',elinewidth=1.2,capsize=3,markersize=4)
ax.axvline(0,lw=.9,color='black',ls=':')
ax.set_yticks(idx,labels=sub.feature_method)
ax.invert_yaxis();ax.set_xlabel('Change in balanced accuracy vs SNV (pp), 95% paired bootstrap interval')
fig.tight_layout();fig.savefig(OUTPUT_DIR/'07_paired_bootstrap_uncertainty.pdf',bbox_inches='tight');plt.close(fig)

# Optional exploratory full-data model, fitted with NEW flow samples trained on all originals.
# It has NO valid held-out test score and must not be used as independent validation.
if TRAIN_FINAL_MODEL:
    syn,sy=generate_flow(X,y,SEED+101001,SYNTHETIC_PER_CLASS)
    xx=np.vstack([snv(X),syn]); yy=np.r_[y,sy]
    eng=FeatureEngineer(best,wn).fit(xx,yy,Xreal=snv(X))
    model=new_classifier('PLS-DA').fit(eng.transform(xx),yy)
    with open(OUTPUT_DIR/'full_fit_plsda_with_flow.pkl','wb') as fh:
        cloudpickle.dump({'feature_engineer':eng,'classifier':model,'wavenumbers':wn,
            'feature_method':best,'note':'Exploratory full fit only; no independent test'},fh)
    np.savez_compressed(OUTPUT_DIR/'full_fit_synthetic_spectra.npz',wavenumbers=wn,
                        spectra_snv=syn,labels=sy)
    pd.DataFrame({'sample':list(patients)+[f'flow_{i:03d}' for i in range(len(syn))],
          'class':yy,'source':['real']*len(y)+['fold-independent flow']*len(syn)}).to_csv(
          OUTPUT_DIR/'full_fit_training_manifest.csv',index=False)

config={'seed':SEED,'n_real':len(y),'healthy':int(sum(y==0)),'sick':int(sum(y==1)),
        'latent_dim':LATENT_DIM,'flow_steps':FLOW_TRAIN_STEPS,
        'synthetic_per_class_every_fold':SYNTHETIC_PER_CLASS,
        'pool_per_class':SYNTHETIC_POOL_PER_CLASS,'features':FEATURES,
        'classifiers':MODELS,'outer_cv':'17-fold patient-wise leave-one-out',
        'selected_plsda_feature_posthoc':best,
        'all_classification_experiments_use_original_train_plus_flow_synthetic':True}
(OUTPUT_DIR/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')

print('\nFeature engineering ablation results (all models use Latent Flow):',flush=True)
print(results.pivot(index='feature_method',columns='classifier',values='balanced_accuracy')
            .loc[FEATURES,MODELS].mul(100).round(1).to_string(),flush=True)
print('\nSaved to',OUTPUT_DIR,flush=True)

# Plot 8: original-patient decision scores, and the 2x17 correctness map.
# This plot is deliberately post-hoc descriptive (no tuning on its outcomes).
a=pred[(pred.classifier=='PLS-DA')&(pred.feature_method==ref)].sort_values('fold').reset_index(drop=True)
b=pred[(pred.classifier=='PLS-DA')&(pred.feature_method==best)].sort_values('fold').reset_index(drop=True)
assert np.array_equal(a.patient.to_numpy(),b.patient.to_numpy())
fig,(ax0,ax1)=plt.subplots(1,2,figsize=(13.4,7.2),sharey=True,
                          gridspec_kw={'width_ratios':[3.3,1.2]})
for i in range(len(a)):
    ax0.plot([a.score[i],b.score[i]],[i,i],color='0.75',lw=1.15)
    ax0.scatter(a.score[i],i,color=('#b36456' if a.pred[i]!=a.true[i] else '#808080'),
                s=48,marker='o',zorder=2)
    ax0.scatter(b.score[i],i,color=('#b36456' if b.pred[i]!=b.true[i] else '#315b71'),
                s=49,marker='D',zorder=3)
ax0.axvline(0,color='black',ls=':',lw=1)
ax0.set_yticks(range(len(a)),[f'{r.patient} ({"sick" if r.true else "healthy"})' for r in a.itertuples()])
ax0.invert_yaxis();ax0.set_xlabel('Out-of-fold PLS-DA score (positive = sick)')
ax0.scatter([],[],color='0.75',marker='o',label='SNV reference')
ax0.scatter([],[],color='#315b71',marker='D',label='ANOVA 256')
ax0.scatter([],[],color='#b36456',marker='D',label='Incorrect prediction')
ax0.legend(loc='lower right',frameon=False)
correct=np.stack([(a.pred==a.true).to_numpy(int),(b.pred==b.true).to_numpy(int)],axis=1)
ax1.imshow(correct,cmap=matplotlib.colors.ListedColormap(['#d59a8f','#8fb5ae']),
           vmin=0,vmax=1,aspect='auto')
ax1.set_xticks([0,1],['SNV','ANOVA 256']);ax1.tick_params(axis='y',left=False,labelleft=False)
ax1.set_xlabel('Correctness on held-out patient')
for i in range(len(a)):
    for j in range(2):ax1.text(j,i,'✓' if correct[i,j] else '×',ha='center',va='center',fontsize=10)
fig.tight_layout();fig.savefig(OUTPUT_DIR/'08_patient_decision_changes.pdf',bbox_inches='tight');plt.close(fig)
