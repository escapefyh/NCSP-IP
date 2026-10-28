"""Original repeated holdout construction and sampled ranking metrics."""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
SEED = 2026101930

def split_indices(count,repeat):
    out=[]
    for fold,(train,held) in enumerate(KFold(n_splits=5,shuffle=True,random_state=SEED+repeat).split(np.arange(count))):
        rng=np.random.default_rng(SEED+100+repeat*10+fold);held=rng.permutation(held);nv=len(held)//2;val=np.sort(held[:nv]);test=np.sort(held[nv:]);out.append((np.sort(train),val,test))
    return out

def unknown_pairs(cat,repeat):
    pos=np.array(cat['positive'],np.int32);n,m=len(cat['names']),len(cat['diseases']);known=set(map(tuple,pos));unknown=np.array([(i,d) for i in range(n) for d in range(m) if (i,d) not in known],np.int32);rng=np.random.default_rng(SEED+1000+repeat);uniform=unknown[rng.choice(len(unknown),len(pos),replace=False)];used=set(map(tuple,uniform));matched=np.empty_like(pos)
    for d in sorted(set(pos[:,1])):
        ids=np.flatnonzero(pos[:,1]==d);eligible=np.array([i for i in range(n) if (i,int(d)) not in known and (i,int(d)) not in used]);selected=rng.choice(eligible,len(ids),replace=False);matched[ids,0]=selected;matched[ids,1]=d
    assert len(set(map(tuple,matched)))==len(pos) and not (set(map(tuple,matched))&used);assert np.array_equal(matched[:,1],pos[:,1]);return uniform,matched

def sampled_metrics(y,s):
    p,r,_=precision_recall_curve(y,s)
    return dict(AUC=float(roc_auc_score(y,s)),AP=float(average_precision_score(y,s)),AUPR=float(auc(r,p)))
