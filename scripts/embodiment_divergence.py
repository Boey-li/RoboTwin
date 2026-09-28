"""Per-task embodiment divergence from end-effector motion.

K-way classification accuracy (what was asked for) plus two statistics that do not
saturate, because accuracy turns out to sit at ceiling on every task:

  acc_lin / acc_rf   K-way balanced accuracy, 5-fold CV. Chance = 1/K.
  min_pair           lowest pairwise accuracy over all embodiment pairs - the most
                     confusable pair. Has spread even when K-way is 1.0.
  mmd                mean pairwise MMD^2 (RBF, median-heuristic bandwidth) on the
                     pooled-standardised features. Continuous, so it ranks.

Representation: EE only (joint vectors are not comparable - different base poses,
Franka is 7-DoF/arm vs 6). Positions centred on t=0 and scale-normalised, so the
known base offset (aloha z=0 + mast vs ARX z=0.784 on the table) and reach
magnitude do not do the separating. Quaternions sign-canonicalised. Grippers
min-max normalised per embodiment. Episodes resampled to a fixed length, since
embodiments differ 1.5-1.8x in episode duration.
"""
from __future__ import annotations
import glob, itertools, json, os, sys, warnings
warnings.filterwarnings("ignore")
import h5py, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, balanced_accuracy_score

DD="/coc/flash7/scratch/robotwin_shared/dataset"
EMB=["aloha-agilex","arx-x5","franka","piper","ur5"]
T=32
CV=StratifiedKFold(5,shuffle=True,random_state=0)
SC=make_scorer(balanced_accuracy_score)

def rs(a,n=T):
    s=np.linspace(0,1,len(a)); d=np.linspace(0,1,n)
    return np.stack([np.interp(d,s,a[:,j]) for j in range(a.shape[1])],1)

def load(task,emb,limit=50):
    eps=[]
    for f in sorted(glob.glob(f"{DD}/{task}/{emb}_clean_50/data/*.hdf5"))[:limit]:
        try:
            with h5py.File(f,"r") as h:
                cols=[]
                for s in ("left","right"):
                    ep=np.asarray(h[f"endpose/{s}_endpose"][()],dtype=np.float64)
                    if ep.ndim!=2 or ep.shape[1]<7: raise ValueError
                    p=ep[:,:3]-ep[0,:3]
                    sc=np.linalg.norm(p,axis=1).max()
                    p=p/(sc if sc>1e-6 else 1.0)
                    q=ep[:,3:7].copy(); q[q[:,0]<0]*=-1.0
                    g=np.asarray(h[f"endpose/{s}_gripper"][()],dtype=np.float64).reshape(-1,1)
                    cols += [p,q,g]
                a=np.concatenate(cols,1)
            if len(a)>=4 and np.isfinite(a).all(): eps.append(a)
        except Exception:
            continue
    if not eps: return None
    gi=[7,15]
    allg=np.concatenate([e[:,gi] for e in eps],0)
    lo,hi=allg.min(0),allg.max(0); rng=np.where(hi-lo<1e-9,1.0,hi-lo)
    X=[]
    for e in eps:
        e=e.copy(); e[:,gi]=(e[:,gi]-lo)/rng
        X.append(rs(e).ravel())
    return np.stack(X)

def mmd2(A,B):
    Z=np.vstack([A,B])
    d2=((Z[:,None,:]-Z[None,:,:])**2).sum(-1)
    med=np.median(d2[d2>0]) if (d2>0).any() else 1.0
    Kk=np.exp(-d2/(med if med>0 else 1.0))
    n,m=len(A),len(B)
    Kxx=Kk[:n,:n]; Kyy=Kk[n:,n:]; Kxy=Kk[:n,n:]
    return float((Kxx.sum()-np.trace(Kxx))/(n*(n-1))
                +(Kyy.sum()-np.trace(Kyy))/(m*(m-1))
                -2*Kxy.mean())

def run(task):
    data={}
    for e in EMB:
        if not os.path.isdir(f"{DD}/{task}/{e}_clean_50"): continue
        X=load(task,e)
        if X is not None and len(X)>=10: data[e]=X
    if len(data)<2: return None
    names=list(data); K=len(names); chance=1.0/K
    X=np.concatenate([data[e] for e in names]); y=np.concatenate(
        [np.full(len(data[e]),i) for i,e in enumerate(names)])
    Xs=StandardScaler().fit_transform(X)
    lin=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
    rf =make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=4))
    a_lin=float(cross_val_score(lin,X,y,cv=CV,scoring=SC,n_jobs=1).mean())
    a_rf =float(cross_val_score(rf ,X,y,cv=CV,scoring=SC,n_jobs=1).mean())
    pairs={}; mmds=[]
    off={e:sum(len(data[n]) for n in names[:i]) for i,e in enumerate(names)}
    for a,b in itertools.combinations(names,2):
        m=np.concatenate([np.full(len(data[a]),0),np.full(len(data[b]),1)])
        Xp=np.concatenate([data[a],data[b]])
        pa=float(cross_val_score(lin,Xp,m,cv=CV,scoring=SC,n_jobs=1).mean())
        pairs[f"{a}|{b}"]=pa
        A=Xs[off[a]:off[a]+len(data[a])]; B=Xs[off[b]:off[b]+len(data[b])]
        mmds.append(mmd2(A,B))
    return dict(task=task,K=K,emb=names,n=int(len(X)),chance=chance,
                acc_lin=a_lin,acc_rf=a_rf,
                min_pair=min(pairs.values()),pairs=pairs,
                mmd=float(np.mean(mmds)),mmd_min=float(np.min(mmds)))

if __name__=="__main__":
    tasks=sys.argv[1:] or sorted(os.listdir(DD))
    res=[]
    for t in tasks:
        try: r=run(t)
        except Exception as ex: print(f"{t:26s} ERROR {type(ex).__name__}",flush=True); continue
        if not r: print(f"{t:26s} SKIP",flush=True); continue
        res.append(r)
        print(f"{r['task']:26s} K={r['K']} lin={r['acc_lin']:.3f} rf={r['acc_rf']:.3f} "
              f"minpair={r['min_pair']:.3f} mmd={r['mmd']:.4f}",flush=True)
        json.dump(res,open("results/embodiment_divergence.json","w"),indent=1)
