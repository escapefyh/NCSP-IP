"""Adopted lambda=0 decoder, extracted from the formal experiment."""
import numpy as np
import torch
from scipy.special import logit
from .branches import Branch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def fit(p,m,off,c,negative):
 torch.manual_seed(p['seed']+off)
 if DEV=='cuda':torch.cuda.manual_seed_all(p['seed']+off)
 x,ann,G=[torch.tensor(c[k],device=DEV) for k in ['x','ann','G']];nets={arm:Branch(c['x'].shape[1],c['A'].shape[1],item=None if m=='no_warm_start' else c['bundle'][f'{off}/{arm}']).to(DEV) for arm in c['arms']}
 def raw():return sum(net(x,ann,G,arm=='real') for arm,net in nets.items())/len(nets)
 with torch.no_grad():initial=raw().clone()
 lp=torch.tensor(logit(np.clip(c['prior'],1e-9,1-1e-9)),dtype=torch.float32,device=DEV);loga=torch.nn.Parameter(torch.tensor(0.,device=DEV));b=torch.nn.Parameter(torch.tensor(0.,device=DEV));params=[v for net in nets.values() for v in net.parameters()];opt=torch.optim.Adam([dict(params=params,lr=.001),dict(params=[loga,b],lr=.03)])
 pi=torch.tensor(np.argwhere(c['A']>0),device=DEV);ni=torch.tensor(negative,device=DEV);trace=[]
 def forward():
  d=(raw()-initial)/5;delta=2*d if m=='no_residual_bound' else 2*torch.tanh(d);return torch.exp(loga)*lp+b+delta,delta
 for ep in range(1,101):
  u,delta=forward();pos=.5*torch.nn.functional.softplus(-u[pi[:,0],pi[:,1]]).mean();neg=.5*torch.nn.functional.softplus(u[ni[:,0],ni[:,1]]).mean();bce=pos if m=='no_negative_BCE' else pos+neg;reg=1e-4*sum(v.square().mean() for v in params);loss=bce+reg;opt.zero_grad();loss.backward()
  if ep==1:grads=dict(positive_logit_derivative=float(-.5*torch.sigmoid(-u[pi[:,0],pi[:,1]]).mean().detach().cpu()),negative_logit_derivative=float(.5*torch.sigmoid(u[ni[:,0],ni[:,1]]).mean().detach().cpu()) if m!='no_negative_BCE' else 0.)
  opt.step()
  if ep in [1,20,50,100]:trace.append(dict(epoch=ep,BCE=float(bce.detach().cpu()),loss=float(loss.detach().cpu()),delta_square_mean=float(delta.square().mean().detach().cpu()),penalty=0.))
 with torch.no_grad():u,delta=forward();score=torch.sigmoid(u).cpu().numpy().astype(float)
 state=dict(branches={arm:net.cpu().state_dict() for arm,net in nets.items()},initial=initial.cpu(),log_a=float(loga.detach().cpu()),bias=float(b.detach().cpu()),trace=trace,lambda_penalty=0.,method=m,offset=off,gradient_checks=grads)
 return state,score

def predict(context, states):
    """Reconstruct the mean independent sigmoid score from decoder checkpoints."""
    x, ann, G = [torch.tensor(context[k], device=DEV) for k in ['x', 'ann', 'G']]
    lp = torch.tensor(logit(np.clip(context['prior'], 1e-9, 1-1e-9)), dtype=torch.float32, device=DEV)
    scores = []
    for state in states.values():
        nets = {}
        for arm, weights in state['branches'].items():
            net = Branch(x.shape[1], context['A'].shape[1]).to(DEV)
            net.load_state_dict(weights)
            nets[arm] = net
        with torch.no_grad():
            raw = sum(net(x, ann, G, arm == 'real') for arm, net in nets.items()) / len(nets)
            d = (raw - state['initial'].to(DEV)) / 5
            delta = 2*d if state['method'] == 'no_residual_bound' else 2*torch.tanh(d)
            score = torch.sigmoid(np.exp(state['log_a'])*lp + state['bias'] + delta)
        scores.append(score.cpu().numpy().astype(float))
    return np.mean(scores, axis=0)
