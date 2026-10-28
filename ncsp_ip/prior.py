"""Original frozen-prior training and NumPy reconstruction routines."""
import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import rankdata

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-6
OFFSETS = [101, 202, 303]

class Net(torch.nn.Module):
 def __init__(self,width,m,dim,graph):
  super().__init__();self.graph=graph;self.encoder=torch.nn.Sequential(torch.nn.Linear(width,dim),torch.nn.Tanh(),torch.nn.Linear(dim,dim)).to(DEVICE);self.disease=torch.nn.Parameter(torch.randn(m,dim,device=DEVICE)*.1);self.bias=torch.nn.Parameter(torch.zeros(m,device=DEVICE))
 def forward(self,x,A):
  t=self.encoder(x);d=self.disease
  if self.graph:
   G=A/A.sum(1).clamp_min(1).sqrt()[:,None]/A.sum(0).clamp_min(1).sqrt()[None,:];t1=G@d;d1=G.T@t;t2=G@d1;d2=G.T@t1;t=(t+t1+t2)/3;d=(d+d1+d2)/3
  return t@d.T+self.bias

def normalized(A):return A/np.sqrt(np.maximum(A.sum(1),1))[:,None]/np.sqrt(np.maximum(A.sum(0),1))[None,:]

def rebuild(item,x,ann,A,arm):
 st={z:v.numpy().astype(float) for z,v in item['state'].items()};h=np.tanh(x.astype(float)@st['encoder.0.weight'].T+st['encoder.0.bias'])@st['encoder.2.weight'].T+st['encoder.2.bias']+ann.astype(float)@item['W'].numpy().astype(float);d=st['disease']
 if arm!='off':
  G=normalized(A).astype(float);h1=G@d;d1=G.T@h;h=(h+h1+G@d1)/3;d=(d+d1+G.T@h1)/3
 return h@d.T+st['bias']

def relation_loss(h,B):
    u=torch.nn.functional.normalize(h,dim=1,eps=1e-8)
    k=B.sum(0);valid=k>1
    if not valid.any():return h.sum()*0
    sums=B.T@u
    diag=B.T@u.square().sum(1)
    mean_cos=(sums.square().sum(1)-diag)/(k*(k-1)).clamp_min(1)
    return (1-mean_cos[valid]).mean()

def numpy_relation(h,B):
    u=h/np.maximum(np.linalg.norm(h,axis=1,keepdims=True),1e-8)
    vals=[]
    for d in range(B.shape[1]):
        ids=np.flatnonzero(B[:,d])
        if len(ids)>1:
            sims=u[ids]@u[ids].T
            vals.append(1-float((sims.sum()-np.trace(sims))/(len(ids)*(len(ids)-1))))
    return float(np.mean(vals)) if vals else 0.

def encoder(net,W,x,ann):return net.encoder(x)+ann@W

def fit_teacher(plan,x,ann,A,R,seed,arm,method,initial=None):
    torch.manual_seed(seed)
    if DEVICE=='cuda':torch.cuda.manual_seed_all(seed)
    device=DEVICE;net=Net(x.shape[1],plan['m'],plan['dimension'],False)
    W=torch.nn.Parameter(torch.zeros((92,plan['dimension']),device=device))
    D0=net.disease.detach().cpu().clone()
    if initial:
        assert torch.equal(initial['state']['disease'],D0)
        net.load_state_dict(initial['state']);W.data.copy_(initial['W'].to(device))
    params=list(net.parameters())+[W];xx=torch.tensor(x,device=device);aa=torch.tensor(ann,device=device)
    At=torch.tensor(A,device=device);Rt=torch.tensor(R,device=device);active=At.sum(1)>0;T=At[active]/At[active].sum(1,keepdim=True)
    G=torch.tensor(normalized(A),device=device);opt=torch.optim.Adam(params,lr=.01)
    weight=.01 if method.endswith('_joint') else 0.;checks=[];trace=[]
    for ep in range(1,161):
        h=encoder(net,W,xx,aa);d=net.disease
        aux=relation_loss(h,Rt) if weight else h.sum()*0
        if arm!='off':
            h1=G@d;d1=G.T@h;h=(h+h1+G@d1)/3;d=(d+d1+G.T@h1)/3
        logp=torch.log_softmax((torch.nn.functional.dropout(h,.2,True)@d.T+net.bias)[active],1)
        ce=-(T*logp).sum(1).mean();loss=ce+1e-4*sum(t.square().mean() for t in params)+weight*aux
        if ep in [1,80,160]:
            take=min(8,len(T));checks.append(dict(epoch=ep,logp=logp[:take].detach().cpu(),target=T[:take].cpu(),CE_slice=float(-(T[:take]*logp[:take]).sum(1).mean().detach().cpu())))
            trace.append(dict(epoch=ep,CE=float(ce.detach().cpu()),relation=float(aux.detach().cpu()),loss=float(loss.detach().cpu())))
        opt.zero_grad();loss.backward();opt.step()
    item=dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},W=W.detach().cpu(),checks=checks,trace=trace,parameters=sum(t.numel() for t in params),disease_changed=float(torch.linalg.norm(net.disease.detach().cpu()-D0)))
    raw=rebuild(item,x,ann,A,arm)
    h=encoder(net,W,xx,aa)
    item['final_relation_loss']=float(relation_loss(h,Rt).detach().cpu())
    st={k:v.numpy().astype(float) for k,v in item['state'].items()}
    hn=np.tanh(x.astype(float)@st['encoder.0.weight'].T+st['encoder.0.bias'])@st['encoder.2.weight'].T+st['encoder.2.bias']+ann.astype(float)@item['W'].numpy().astype(float)
    assert abs(numpy_relation(hn,R)-item['final_relation_loss'])<3e-5
    return item,raw

def neighbors(x,A,seed,random=False):
    n=len(A);warm=np.flatnonzero(A.sum(1)>0);sim=x.astype(float)@x.astype(float).T;out=np.zeros((n,n),np.float32);rng=np.random.default_rng(seed)
    for i in range(n):
        donors=warm[warm!=i];rank=np.argsort(-sim[i,donors],kind='stable');take=donors[rank[:3]];w=sim[i,take];ok=w>0;take,w=take[ok],w[ok]
        if random and len(take):take=rng.choice(donors,len(take),replace=False)
        if len(take):out[i,take]=(w/w.sum()).astype(np.float32)
    assert np.all(np.diag(out)==0) and np.all(out[:,A.sum(1)==0]==0)
    return out

def row_normal(A):return np.divide(A,A.sum(1,keepdims=True),out=np.zeros_like(A),where=A.sum(1,keepdims=True)>0)

def masked_context(A,P,rng,profile=None):
    ids=np.flatnonzero(A.sum(1)>0);C=A[ids].copy();targets=np.array([rng.choice(np.flatnonzero(A[i])) for i in ids]);C[np.arange(len(ids)),targets]=0;C[rng.random(len(ids))<.5]=0
    empty=C.sum(1)==0;C=row_normal(C);profile=P@row_normal(A) if profile is None else profile;C[empty]=profile[ids[empty]]
    return C,targets,empty

def completion_rebuild(item,C):
    st={k:v.numpy().astype(float) for k,v in item['state'].items()};return np.tanh(C.astype(float)@st['0.weight'].T+st['0.bias'])@st['2.weight'].T+st['2.bias']

def completion(A,P,seed):
    torch.manual_seed(seed+600000);net=torch.nn.Sequential(torch.nn.Linear(A.shape[1],64),torch.nn.Tanh(),torch.nn.Linear(64,A.shape[1])).to(DEVICE)
    torch.nn.init.zeros_(net[2].weight);torch.nn.init.zeros_(net[2].bias);params=list(net.parameters());opt=torch.optim.Adam(params,lr=.01);rng=np.random.default_rng(seed+610000);checks=[];profile=P@row_normal(A)
    for epoch in range(1,161):
        C,t,empty=masked_context(A,P,rng,profile);ct=torch.tensor(C,device=DEVICE);tt=torch.tensor(t,dtype=torch.long,device=DEVICE);lp=torch.log_softmax(net(ct),1);ce=-lp[torch.arange(len(t),device=DEVICE),tt].mean();loss=ce+1e-4*sum(p.square().mean() for p in params)
        if epoch in [1,80,160]:checks.append(dict(epoch=epoch,logp=lp[:8].detach().cpu(),target_index=t[:8],CE_slice=float(-lp[:8,][torch.arange(min(8,len(t)),device=DEVICE),tt[:8]].mean().detach().cpu()),empty_fraction=float(empty.mean())))
        opt.zero_grad();loss.backward();opt.step()
    item=dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},checks=checks,parameters=sum(p.numel() for p in params));C=P@row_normal(A);raw=completion_rebuild(item,C)
    with torch.no_grad():actual=net(torch.tensor(C,device=DEVICE)).cpu().numpy().astype(float)
    err=float(np.max(abs(softmax(raw,1)-softmax(actual,1))));assert err<3e-5;item['probability_error']=err
    return item,raw

def pava_decreasing(values,weights):
 levels=[];mass=[];sizes=[]
 for v,w in zip(values,weights):
  levels.append(float(v));mass.append(float(w));sizes.append(1)
  while len(levels)>1 and levels[-2]<levels[-1]:
   w=mass[-2]+mass[-1];v=(levels[-2]*mass[-2]+levels[-1]*mass[-1])/w;n=sizes[-2]+sizes[-1];levels[-2:]=[v];mass[-2:]=[w];sizes[-2:]=[n]
 return np.concatenate([np.full(n,v) for v,n in zip(levels,sizes)])

def project(P,A):
 result=np.empty_like(P);projected=np.empty_like(P)
 for i in range(len(P)):
  order=np.argsort(-P[i],kind='stable');x=P[i,order];y=A[i,order];starts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1];ends=np.r_[starts[1:],len(x)];weights=ends-starts;averages=np.array([y[a:b].mean() for a,b in zip(starts,ends)]);z=pava_decreasing(averages,weights);z=np.repeat(z,weights);q=np.empty_like(z);q[order]=z;projected[i]=q
  # Fixed numerical strictness restoration; preserve all original ties.
  ranks=rankdata(P[i],method='average');ordinal=ranks/ranks.sum();out=(1-EPS)*q+EPS*ordinal;result[i]=out/out.sum()
 return result,projected


def train_prior(A, x, ann, seed, P=None):
    """Train three seeds; average each view before order projection/completion."""
    if P is None:
        P = neighbors(x, A, seed)
    plan = dict(m=A.shape[1], dimension=32)
    bundle = {}
    N = np.zeros_like(A, dtype=float)
    G = N.copy()
    C = N.copy()
    for offset in OFFSETS:
        for arm, mean in [('off', N), ('real', G)]:
            item, raw = fit_teacher(plan, x, ann, A, A, seed + offset, arm, 'baseline')
            bundle[f'{offset}/{arm}'] = item
            mean += softmax(raw, 1) / 3
        item, raw = completion(A, P, seed + offset)
        bundle[f'{offset}/soft'] = item
        C += softmax(raw, 1) / 3
    prior = project(N, (N + G) / 2)[0]
    cold = A.sum(1) == 0
    prior[cold] = .9 * prior[cold] + .1 * C[cold]
    return prior, bundle
