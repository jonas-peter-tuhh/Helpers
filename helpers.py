import numpy as np
import torch
from torch import device
from torch.autograd import grad, Variable
device = device("cuda:0" if torch.cuda.is_available() else "cpu")
def deriv(f,x,n):
    '''
    calculates the first n-th derivatives of f with respect to x
    '''
    out = []
    for _ in range(n):
        f = grad(f,x,create_graph = True, retain_graph = True, grad_outputs = torch.ones_like(f))[0]
        out.append(f)
    return out
def myconverter(x,grad = True):
    if isinstance(x,np.ndarray):
        x = Variable(torch.from_numpy(x).float(), requires_grad = grad).to(device)
    elif torch.is_tensor(x):
        x = x.cpu().detach().numpy().squeeze()
    return x
def errsum(fun,*args):
    out = 0
    for bc in args:
        out += fun(bc,Variable(torch.zeros(bc.shape), requires_grad = False).to(device))
    return out
    
