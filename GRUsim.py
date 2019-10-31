import torch as th
import torch.nn as nn
import numpy as np
npa = np.asarray

def tonumpy(ls):
    return [t.data.numpy() for t in ls]
class Erf(nn.Module):
    def forward(self, x):
        return x.erf()
class ErfSigmoid(nn.Module):
    def forward(self, x):
        return (1 + x.erf())/2


def simgru(inpseq, d_h, time=None, varUz=1, varUr=1, varUh=1,
           varWz=1, varWr=1, varWh=1,
           varbz=1, varbr=1, varbh=1,
           mubz=0, mubr=0, mubh=0,
           bias=True, nonlin=None, sigmoid=None, wt_tie=True,
           # get_h=False, get_update=False, get_reset=False, get_htilde=False,
           h_init=0):
    r'''Simulate a GRU on a sequence and obtain data

    A GRU evolves according to the equations
    
        \tilde z^t = W_z x^t + U_z h^{t-1} + b_z
        z^t = sigmoid(\tilde z^t)
        \tilde r^t = W_r x^t + U_r h^{t-1} + b_r
        r^t = sigmoid(\tilde r^t)
        \tilde h^t = W_h x^t + U_h(h^{t-1} \odot r^t) + b_h
        h^t = (1 - z^t) \odot h^{t-1} + z^t \odot nonlin(\tilde h^t)
        
    where
    
        h^t is state at time t
        x^t is input at time t
        z^t is ``update gate``: 1 means do update/forget previous h^{t-1}
                                0 means h^t = h^{t-1}
        r^t is ``reset gate'':
            the smaller, the easier to make proposed update not depend on h^{t-1}
        W_z, W_r, W_h are weights converting input to hidden states
        U_z, U_r, U_h are weights converting state to state
        b_z, b_r, b_h are biases

    This function simulates a randomly initialized GRU
    on the sequence `inpseq` and returns a record of
    several data.

    Inputs:
        inpseq: a matrix `seqlen x inputdim` where each row
            is a token
        d_h: dimension of state h^t
        time: run simulation for t up to `time`.
            If is None, then set of `seqlen`.
        varUz: each element of U_z has variance `varUz`/d
        varUr: each element of U_r has variance `varUr`/d
        varUh: each element of U_h has variance `varUh`/d
        varWz: each element of W_z has variance `varWz`/d
        varWr: each element of W_r has variance `varWr`/d
        varWh: each element of W_h has variance `varWh`/d
        varbz: each element of b_z has variance `varbz`
        varbr: each element of b_r has variance `varbr`
        varbh: each element of b_h has variance `varbh`
        mubz: each element of b_z has mean `mubz`
        mubr: each element of b_r has mean `mubr`
        mubh: each element of b_h has mean `mubh`
        bias: whether to turn on bias
            (if False, then `varb_`s and `mub_`s have no effect)
        nonlin: nonlinearity; has to be pytorch module or
            has the same ducktype
        sigmoid: sigmoid function; has to pytorch module
            or has the same ducktype
        wt_tie: whether to tie the weights
        h_init: the magnitude of the initial state (which
            is randomly initialized)

    Outputs:
        a dictionary with the following keys

        h: `time+1 x d_h` matrix containing all h^t
        th: `time x d_h` matrix containing all \tilde h^t
        tz: `time x d_h` matrix containing all \tilde z^t
        tr: `time x d_h` matrix containing all \tilde r^t
        hcov: `time+1 x time+1` matrix of 2nd moments of h^t
        thcov: `time x time` matrix of 2nd moments of \tilde h^t
        tzcov: `time x time` matrix of 2nd moments of \tilde z^t
        trcov: `time x time` matrix of 2nd moments of \tilde r^t
    '''
    d_i = inpseq.shape[1]
    if time is None:
        time = inpseq.shape[0]
    updates = []
    resets = []
    htildes = []
    def makelayer(nonlin=nonlin, sigmoid=sigmoid):
        W_update = nn.Linear(d_i, d_h, bias=False)
        update = nn.Linear(d_h, d_h, bias=bias)
        W_reset = nn.Linear(d_i, d_h, bias=False)
        reset = nn.Linear(d_h, d_h, bias=bias)
        W_htilde = nn.Linear(d_i, d_h, bias=False)
        htilde = nn.Linear(d_h, d_h, bias=bias)
        W_update.weight.data.normal_(0, np.sqrt(varWz/d_i))
        update.weight.data.normal_(0, np.sqrt(varUz/d_h))
        W_reset.weight.data.normal_(0, np.sqrt(varWr/d_i))
        reset.weight.data.normal_(0, np.sqrt(varUr/d_h))
        W_htilde.weight.data.normal_(0, np.sqrt(varWh/d_i))
        htilde.weight.data.normal_(0, np.sqrt(varUh/d_h))
        if bias:
            update.bias.data.normal_(mubz, np.sqrt(varbz))
            reset.bias.data.normal_(mubr, np.sqrt(varbr))
            htilde.bias.data.normal_(mubh, np.sqrt(varbh))
        if nonlin is None:
            nonlin = lambda: lambda x: x
        if sigmoid is None:
            sigmoid = nn.Sigmoid
        def r(h, inp):
            _u = update(h) + W_update(inp)
            # if get_update:
            updates.append(_u)
            u = sigmoid()(_u)
            _r = reset(h) + W_reset(inp)
            # if get_reset:
            resets.append(_r)
            r = sigmoid()(_r)
            _h = htilde(h * r) + W_htilde(inp)
            # if get_htilde:
            htildes.append(_h)
            ht = nonlin()(_h)
            return (1 - u) * h + u * ht
        return r
    if wt_tie:
        mylayer = makelayer()
#     if h_init is None:
#         x = 0
#     else:
    x = th.randn(1, d_h) * h_init
    xs = [x]
    for i in range(time):
        if not wt_tie:
            mylayer = makelayer()
        xx = mylayer(xs[-1], inpseq[i:i+1])
        xs.append(xx)

    ret = {}
    # if get_h:
    ret['h'] = npa(tonumpy(xs)).squeeze()
    # if get_update:
    ret['tz'] = npa(tonumpy(updates)).squeeze()
    # if get_reset:
    ret['tr'] = npa(tonumpy(resets)).squeeze()
    # if get_htilde:
    ret['th'] = npa(tonumpy(htildes)).squeeze()
#     print(ret['h'].shape)
    ret['tzcov'] = ret['tz'] @ ret['tz'].T / d_h
    ret['trcov'] = ret['tr'] @ ret['tr'].T / d_h
    ret['thcov'] = ret['th'] @ ret['th'].T / d_h
    ret['hcov'] = ret['h'] @ ret['h'].T / d_h
    # ret['hnorms'] = [th.mean(u**2).data.item() for u in xs]
    return ret



def simgru2(inpseq1, inpseq2, d_h, varUz=1, varUr=1, varUh=1,
           varWz=1, varWr=1, varWh=1,
           varbz=1, varbr=1, varbh=1,
           mubz=0, mubr=0, mubh=0,
           bias=True, nonlin=None, sigmoid=None, wt_tie=True,
           # get_h=False, get_update=False, get_reset=False, get_htilde=False,
           h_init=0):
    d_i = inpseq1.shape[1]
    updates = []
    resets = []
    htildes = []
    def makelayer(nonlin=nonlin, sigmoid=sigmoid):
        W_update = nn.Linear(d_i, d_h, bias=False)
        update = nn.Linear(d_h, d_h, bias=bias)
        W_reset = nn.Linear(d_i, d_h, bias=False)
        reset = nn.Linear(d_h, d_h, bias=bias)
        W_htilde = nn.Linear(d_i, d_h, bias=False)
        htilde = nn.Linear(d_h, d_h, bias=bias)
        W_update.weight.data.normal_(0, np.sqrt(varWz/d_i))
        update.weight.data.normal_(0, np.sqrt(varUz/d_h))
        W_reset.weight.data.normal_(0, np.sqrt(varWr/d_i))
        reset.weight.data.normal_(0, np.sqrt(varUr/d_h))
        W_htilde.weight.data.normal_(0, np.sqrt(varWh/d_i))
        htilde.weight.data.normal_(0, np.sqrt(varUh/d_h))
        if bias:
            update.bias.data.normal_(mubz, np.sqrt(varbz))
            reset.bias.data.normal_(mubr, np.sqrt(varbr))
            htilde.bias.data.normal_(mubh, np.sqrt(varbh))
        if nonlin is None:
            nonlin = lambda: lambda x: x
        if sigmoid is None:
            sigmoid = nn.Sigmoid
        def r(h, inp):
            _u = update(h) + W_update(inp)
            # if get_update:
            updates.append(_u)
            u = sigmoid()(_u)
            _r = reset(h) + W_reset(inp)
            # if get_reset:
            resets.append(_r)
            r = sigmoid()(_r)
            _h = htilde(h * r) + W_htilde(inp)
            # if get_htilde:
            htildes.append(_h)
            ht = nonlin()(_h)
            return (1 - u) * h + u * ht
        return r
    if wt_tie:
        mylayer = makelayer()
#     if h_init is None:
#         x = 0
#     else:
    rets = {1: {}, 2: {}}
    for i, seq in enumerate([inpseq1, inpseq2]):
        x = th.randn(1, d_h) * h_init
        xs = [x]
        for tok in seq:
            if not wt_tie:
                mylayer = makelayer()
            xx = mylayer(xs[-1], tok)
            xs.append(xx)

        ret = rets[i+1]
        # if get_h:
        ret['h'] = npa(tonumpy(xs)).squeeze()
        # if get_update:
        ret['tz'] = npa(tonumpy(updates)).squeeze()
        # if get_reset:
        ret['tr'] = npa(tonumpy(resets)).squeeze()
        # if get_htilde:
        ret['th'] = npa(tonumpy(htildes)).squeeze()
    #     print(ret['h'].shape)
        ret['tzcov'] = ret['tz'] @ ret['tz'].T / d_h
        ret['trcov'] = ret['tr'] @ ret['tr'].T / d_h
        ret['thcov'] = ret['th'] @ ret['th'].T / d_h
        ret['hcov'] = ret['h'] @ ret['h'].T / d_h
    ret = rets['x'] = {}
    ret['tzcov'] = rets[1]['tz'] @ rets[2]['tz'].T / d_h
    ret['trcov'] = rets[1]['tr'] @ rets[2]['tr'].T / d_h
    ret['thcov'] = rets[1]['th'] @ rets[2]['th'].T / d_h
    ret['hcov'] = rets[1]['h'] @ rets[2]['h'].T / d_h
    # ret['hnorms'] = [th.mean(u**2).data.item() for u in xs]
    
    rets['hcov'] =  np.block(
        [[rets[1]['hcov'][1:, 1:], rets['x']['hcov'][1:, 1:]],
         [rets['x']['hcov'][1:, 1:].T, rets[2]['hcov'][1:, 1:]]]
    )
    return rets