"""Original name and graph decoder branches."""
import torch

class Branch(torch.nn.Module):
    def __init__(self,width,m,dim=32,item=None):
        super().__init__();self.encoder=torch.nn.Sequential(torch.nn.Linear(width,dim),torch.nn.Tanh(),torch.nn.Linear(dim,dim));self.disease=torch.nn.Parameter(torch.randn(m,dim)*.1);self.bias=torch.nn.Parameter(torch.zeros(m));self.W=torch.nn.Parameter(torch.zeros(92,dim))
        if item is not None:
            state=dict(item['state']);state['W']=item['W'];self.load_state_dict(state)
    def forward(self,x,ann,G,graph=False,training=False):
        h=self.encoder(x)+ann@self.W;d=self.disease
        if graph:
            h1=G@d;d1=G.T@h;h=(h+h1+G@d1)/3;d=(d+d1+G.T@h1)/3
        return torch.nn.functional.dropout(h,.2,training)@d.T+self.bias
