import rpy2
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector as FV
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from numpy import asarray as npa
import numbers

mvtnorm = importr('mvtnorm')
r'''
We require the R package `mvtnorm` to compute certain high dimensional expectations
involving erfs, by reducing such expectations to Gaussian orthant probabilities
and evaluating them with `mvtnorm`.
'''
def Eerf2(mu, var):
    r'''
    Computes
        E[erf(x)^2: x ~ N(mu, var)]
    when mu and var are scalars, or
        E[erf(x) erf(y): (x, y) ~ N(mu, var)]
    when mu is a length-2 vector and var is a 2x2 matrix
    Example:
        >>> Eerf2([0, 0], [[1, 1], [1, 1]])
        0.4645590543975399
        
    Inputs:
        mu: scalar or length-2 vector
        var: scalar, or 2x2 matrix
    Outputs:
        Gaussian expectation as explained above
    '''
    scalar = False
    if isinstance(var, numbers.Number) or \
            isinstance(var, np.ndarray) and var.size == 1:
        mean = FV([mu, mu])
        var = np.array([var] * 4).reshape(2, 2) + 0.5 * np.eye(2)
        scalar = True
    else:
        mean = FV(mu)
        var = np.asarray(var + 0.5 * np.eye(2))
    ppprob = npa(mvtnorm.pmvnorm(lower=FV(np.zeros(2)), mean=mean, sigma=var))
    pnprob = npa(mvtnorm.pmvnorm(lower=FV(np.array([0, -np.inf])),
                           upper=FV(np.array([np.inf, 0])),
                           mean=mean, sigma=var))
    if scalar:
        npprob = pnprob
    else:
        npprob = npa(mvtnorm.pmvnorm(lower=FV(np.array([-np.inf, 0])),
                               upper=FV(np.array([0, np.inf])),
                               mean=mean, sigma=var))
    nnprob = npa(mvtnorm.pmvnorm(lower=FV(np.array([-np.inf, -np.inf])),
                           upper=FV(np.array([0, 0])),
                           mean=mean, sigma=var))
    return (ppprob + nnprob - pnprob - npprob)[0]
    
def Esigmoidprod(signs, mu, cov):
    r'''
    Computes
        E[prod_i sigmoid(sgn_i * x_i): (x_i)_i ~ N(mu, cov)]
    where
        sigmoid(x) = 0.5 (1 + erf(x))
        sgn_i = `signs`[i]
    Inputs:
        signs: a vector of +/-1
        mu: vector of means
        cov: matrix of covariances
    '''
    cov = npa(cov)
    signs = npa(signs)
    mu = npa(mu)
    cov = signs[:, None] * cov * signs[None, :] + 0.5 * np.eye(cov.shape[0])
    mu *= signs
    return npa(mvtnorm.pmvnorm(lower=FV(np.zeros_like(mu)),
                           mean=FV(mu),
                           sigma=cov))[0]
def Esigmoid2prod(signs1, signs2, mu, cov):
    r'''
    Computes
        E[(prod_i s_i s'_i): (x_i)_i ~ N(mu, cov)]
    where
        s_i = 1 if `signs1[i] == 0` or else sigmoid(`signs1[i]` * x_i)
        s'_i = 1 if `signs2[i] == 0` or else sigmoid(`signs2[i]` * x_i)
        sigmoid(x) = 0.5 (1 + erf(x))
        sgn_i = `signs`[i]
    Inputs:
        signs1: a vector with entries from {-1, 0, 1}
        signs2: a vector with entries from {-1, 0, 1}
    '''
    # the lazy way:
    # duplicate cov to reduce to the case with no repeat variables.
    signs1 = npa(signs1)
    signs2 = npa(signs2)
    mu = npa(mu)
    cov = npa(cov)
    n = cov.shape[0]
    newcov = np.zeros([2 * n , 2 * n])
    newcov[:n, :n] = newcov[:n, n:] = newcov[n:, :n] = newcov[n:, n:] = cov
    newmu = np.concatenate([mu, mu], axis=0)
    signs = np.concatenate([signs1, signs2], axis=0)
    zero_idx = set(np.argwhere(signs==0).reshape(-1))
    nonzero_idx = list(set(list(range(0, 2*n))) - zero_idx)
    signs = signs[nonzero_idx]
    newmu = newmu[nonzero_idx]
    newcov = newcov[nonzero_idx, :][:, nonzero_idx]
    return Esigmoidprod(signs, newmu, newcov)

def thgru(ingram, varUz=1, varUr=1, varUh=1,
          varWz=1, varWr=1, varWh=1,
          varbz=1, varbr=1, varbh=1,
          mubz=0, mubr=0, mubh=0):
    r'''
    Computes the infinite-width GP kernel of an erf-GRU over an input sequence
    with covariance `ingram`.
    An erf-GRU evolves according to the equations
    
        \tilde z^t = W_z x^t + U_z h^{t-1} + b_z
        z^t = sigmoid(\tilde z^t)
        \tilde r^t = W_r x^t + U_r h^{t-1} + b_r
        r^t = sigmoid(\tilde r^t)
        \tilde h^t = W_h x^t + U_h(h^{t-1} \odot r^t) + b_h
        h^t = (1 - z^t) \odot h^{t-1} + z^t \odot erf(\tilde h^t)
        
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
        sigmoid(x) = (1 + erf(x)) / 2
        
    We use erf-derived activations so that we can compute
    the infinite-width GP kernel tractably.
    Note that we assume the initial state h^0 is 0.

    Below `d` is the dimension of the hidden state.
    Inputs:
        ingram: gram matrix of input tokens (divided by their dimension)
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
    Outputs:
        a dictionary with the following keys, which are associated to functions
        which are lazily evaluated:
        
        covhtht: covhtht(t, s) = covariance(\tilde h^t, \tilde h^s)
        covztzt: covztzt(t, s) = covariance(\tilde z^t, \tilde z^s)
        covrtrt: covrtrt(t, s) = covariance(\tilde r^t, \tilde r^s)
        Ess: Ess(t, s) = E erf(\tilde h^t) erf(\tilde h^s)
        Ezwzw: Ezwzw(a, t, b, s) = E zweight(a, t) zweight(b, s)
                where zweight(a, t) = z^a (1 - z^(a + 1)) ... (1 - z^t)
        Ehh: Ehh(t, s) = E h^t h^s
        Err: Err(t, s) = E r^t r^s
        
        Here, for two vectors of dimension m, let
            
            covariance(u, v) = u^T v / m

        The returned dictionary also contains dictionaries which hold the memoized values
        of the above functions. Their keys are the same but prefixed with `_`:
        
        _covhtht
        _covztzt
        _covrtrt
        _Ess
        _Ezwzw
        _Ehh
        _Err
        
    '''
    # covhtht(t, s) = covariance(\tilde h^t, \tilde h^s)
    _covhtht = {}
    def covhtht(t, s):
        if (t, s) in _covhtht:
            return _covhtht[(t, s)]
        if t < s:
            return covhtht(s, t)
        # covhtht(1, 1) inolves the first token, ingram[0, 0]
        _covhtht[(t, s)] = varWh * ingram[t-1, s-1] + \
            varUh * Ehh(t-1, s-1) * Err(t, s) + varbh
        return _covhtht[(t, s)]
    # Ess(t, s) = E erf(\tilde h^t) erf(\tilde h^s)
    _Ess = {}
    def Ess(t, s):
        if (t, s) in _Ess:
            return _Ess[(t, s)]
        if t < s:
            return Ess(s, t)
        elif t == s:
            _Ess[(t, s)] = Eerf2(mubh, covhtht(t, t))
            return _Ess[(t, s)]
        else:
            cov = [[covhtht(t, t), covhtht(t, s)],
                  [covhtht(t, s), covhtht(s, s)]]
            _Ess[(t, s)] = Eerf2([mubh, mubh], cov)
            return _Ess[(t, s)]
    # Ezwzw(a, t, b, s) = E zweight(a, t) zweight(b, s)
    #     where zweight(a, t) = z^a (1 - z^(a + 1)) ... (1 - z^t)
    _Ezwzw = {}
    def Ezwzw(a, t, b, s):
        if (a, t, b, s) in _Ezwzw:
            return _Ezwzw[(a, t, b, s)]
        if t < s:
            return Ezwzw(b, s, a, t)
        min_ = min(a, b)
        max_ = max(t, s)
        def getidx(i):
            return i - min_
        cov = [[covztzt(i, j) for i in range(min_, max_+1)]
              for j in range(min_, max_+1)]
        signs1 = np.zeros([max_ - min_ + 1])
        signs1[getidx(a)] = 1
        signs1[getidx(a)+1:getidx(t)+1] = -1
        signs2 = np.zeros([max_ - min_ + 1])
        signs2[getidx(b)] = 1
        signs2[getidx(b)+1:getidx(s)+1] = -1
        _Ezwzw[(a, t, b, s)] = Esigmoid2prod(signs1, signs2,
                                 np.zeros([max_ - min_ + 1]) + mubz,
                                 cov)
        return _Ezwzw[(a, t, b, s)]
    # Ehh(t, s) = E h^t h^s
    _Ehh = {}
    def Ehh(t, s):
        if (t, s) in _Ehh:
            return _Ehh[(t, s)]
        if t == 0 or s == 0:
            return 0
        elif t < s:
            return Ehh(s, t)
        else:
            _Ehh[(t, s)] = sum([
                    Ezwzw(a, t, b, s) * Ess(a, b)
                    for a in range(1, t+1)
                    for b in range(1, s+1)
            ])
            return _Ehh[(t, s)]
    # covztzt(t, s) = covariance(\tilde z^t, \tilde z^s)
    _covztzt = {}
    def covztzt(t, s):
        if (t, s) in _covztzt:
            return _covztzt[(t, s)]
        if t < s:
            return covztzt(s, t)
        # covztzt(1, 1) inolves the first token, ingram[0, 0]
        _covztzt[(t, s)] = varWz * ingram[t-1, s-1] \
                + varUz * Ehh(t-1, s-1) + varbz
        return _covztzt[(t, s)]
    # covrtrt(t, s) = covariance(\tilde r^t, \tilde r^s)
    _covrtrt = {}
    def covrtrt(t, s):
        if (t, s) in _covrtrt:
            return _covrtrt[(t, s)]
        if t < s:
            return covrtrt(s, t)
        # covrtrt(1, 1) inolves the first token, ingram[0, 0]
        _covrtrt[(t, s)] = varWr * ingram[t-1, s-1] \
                + varUr * Ehh(t-1, s-1) + varbr
        return _covrtrt[(t, s)]
    # Err(t, s) = E r^t r^s
    _Err = {}
    def Err(t, s):
        if (t, s) in _Err:
            return _Err[(t, s)]
        if t < s:
            return Err(s, t)
        elif t == s:
            _Err[(t, s)] = Esigmoid2prod([1], [1], [mubr], [covrtrt(t, t)])
            return _Err[(t, s)]
        else:
            cov = [[covrtrt(i, j) for i in [t, s]]
                  for j in [t, s]]
            _Err[(t, s)] = Esigmoidprod([1, 1], [mubr, mubr], cov)
            return _Err[(t, s)]
    return dict(covhtht=covhtht, Ess=Ess, Ezwzw=Ezwzw,
               Ehh=Ehh, covztzt=covztzt, covrtrt=covrtrt,
               Err=Err,
               _covhtht=_covhtht, _Ess=_Ess, _Ezwzw=_Ezwzw,
               _Ehh=_Ehh, _covztzt=_covztzt, _covrtrt=_covrtrt,
               _Err=_Err)

def thgru2(in1covs, in2covs, ingramx, 
            varUz=1, varUr=1, varUh=1,
            varWz=1, varWr=1, varWh=1,
            varbz=1, varbr=1, varbh=1,
            mubz=0, mubr=0, mubh=0):
    r'''Same as `thgru` but over 2 sequences.

    For two vectors of dimension d, let
        
        covariance(u, v) = u^T v / d

    Inputs:
        `in1covs`: dict returned by `thgru` applied to 1st sequence
        `in2covs`: dict returned by `thgru` applied to 2nd sequence
        `ingramx`: the covariance btw the 1st and 2nd sequence
            ingramx[i, j] = covariance(seq1[i], seq2[j])

    Outputs:
        a dictionary with the following keys, which are associated to functions
        which are lazily evaluated:
        
        covhtht: covhtht(t, s) = covariance(\tilde h1^t, \tilde h2^s)
        covztzt: covztzt(t, s) = covariance(\tilde z1^t, \tilde z2^s)
        covrtrt: covrtrt(t, s) = covariance(\tilde r1^t, \tilde r2^s)
        Ess: Ess(t, s) = E erf(\tilde h1^t) erf(\tilde h2^s)
        Ezwzw: Ezwzw(a, t, b, s) = E zweight1(a, t) zweight2(b, s)
                where zweight1(a, t) = z1^a (1 - z1^(a + 1)) ... (1 - z1^t)
                and zweight2(a, t) = z2^a (1 - z2^(a + 1)) ... (1 - z2^t)
        Ehh: Ehh(t, s) = E h1^t h2^s
        Err: Err(t, s) = E r1^t r2^s

        Here ?1 and ?2 refer to the vectors obtained from the 1st or the 2nd sequence
        
        the dictionary also contains dictionaries which hold the memoized values
        of the above functions. Their keys are the same but prefixed with `_`:
        
        _covhtht
        _covztzt
        _covrtrt
        _Ess
        _Ezwzw
        _Ehh
        _Err

    '''
    # covhtht(t, s) = covariance(\tilde h1^t, \tilde h2^s)
    _covhtht = {}
    def covhtht(t, s):
        if (t, s) in _covhtht:
            return _covhtht[(t, s)]
        # covhtht(1, 1) inolves the first token, ingram[0, 0]
        _covhtht[(t, s)] = varWh * ingramx[t-1, s-1] + \
            varUh * Ehh(t-1, s-1) * Err(t, s) + varbh
        return _covhtht[(t, s)]
    # Ess(t, s) = E erf(\tilde h1^t) erf(\tilde h2^s)
    _Ess = {}
    def Ess(t, s):
        if (t, s) in _Ess:
            return _Ess[(t, s)]
        else:
            cov = [[in1covs['covhtht'](t, t), covhtht(t, s)],
                  [covhtht(t, s), in2covs['covhtht'](s, s)]]
            _Ess[(t, s)] = Eerf2([mubh, mubh], cov)
            return _Ess[(t, s)]
    # Ezwzw(a, t, b, s) = E zweight1(a, t) zweight2(b, s)
    #     where zweight1(a, t) = z1^a (1 - z1^(a + 1)) ... (1 - z1^t)
    #           zweight2(a, t) = z2^a (1 - z2^(a + 1)) ... (1 - z2^t)
    _Ezwzw = {}
    def Ezwzw(a, t, b, s):
        if (a, t, b, s) in _Ezwzw:
            return _Ezwzw[(a, t, b, s)]
        min_ = min(a, b)
        max_ = max(t, s)
        def getidx(i):
            return i - min_
        cov1 = npa([[in1covs['covztzt'](i, j) for i in range(a, t+1)]
                for j in range(a, t+1)])
        cov2 = npa([[in2covs['covztzt'](i, j) for i in range(b, s+1)]
                for j in range(b, s+1)])
        covx = npa([[covztzt(i, j) for j in range(b, s+1)]
              for i in range(a, t+1)])
        cov = np.block(
            [[cov1, covx],
             [covx.T, cov2]])
        signs1 = [1] + [-1] * (t - a)
        signs2 = [1] + [-1] * (s - b)
        _Ezwzw[(a, t, b, s)] = Esigmoidprod(signs1 + signs2,
                                 np.zeros([cov.shape[0]]) + mubz,
                                 cov)
        return _Ezwzw[(a, t, b, s)]
    # Ehh(t, s) = E h1^t h2^s
    _Ehh = {}
    def Ehh(t, s):
        if (t, s) in _Ehh:
            return _Ehh[(t, s)]
        if t == 0 or s == 0:
            return 0
        else:
            _Ehh[(t, s)] = sum([
                    Ezwzw(a, t, b, s) * Ess(a, b)
                    for a in range(1, t+1)
                    for b in range(1, s+1)
            ])
            return _Ehh[(t, s)]
    # covztzt(t, s) = covariance(\tilde z1^t, \tilde z2^s)
    _covztzt = {}
    def covztzt(t, s):
        if (t, s) in _covztzt:
            return _covztzt[(t, s)]
        # covztzt(1, 1) inolves the first token, ingram[0, 0]
        _covztzt[(t, s)] = varWz * ingramx[t-1, s-1] \
                + varUz * Ehh(t-1, s-1) + varbz
        return _covztzt[(t, s)]
    # covrtrt(t, s) = covariance(\tilde r1^t, \tilde r2^s)
    _covrtrt = {}
    def covrtrt(t, s):
        if (t, s) in _covrtrt:
            return _covrtrt[(t, s)]
        # covrtrt(1, 1) inolves the first token, ingram[0, 0]
        _covrtrt[(t, s)] = varWr * ingramx[t-1, s-1] \
                + varUr * Ehh(t-1, s-1) + varbr
        return _covrtrt[(t, s)]
    # Err(t, s) = E r1^t r2^s
    _Err = {}
    def Err(t, s):
        if (t, s) in _Err:
            return _Err[(t, s)]
        else:
            cov = [[in1covs['covrtrt'](t, t), covrtrt(t, s)],
                  [covrtrt(t, s), in2covs['covrtrt'](s, s)]]
            _Err[(t, s)] = Esigmoidprod([1, 1], [mubr, mubr], cov)
            return _Err[(t, s)]
    return dict(covhtht=covhtht, Ess=Ess, Ezwzw=Ezwzw,
               Ehh=Ehh, covztzt=covztzt, covrtrt=covrtrt,
               Err=Err,
               _covhtht=_covhtht, _Ess=_Ess, _Ezwzw=_Ezwzw,
               _Ehh=_Ehh, _covztzt=_covztzt, _covrtrt=_covrtrt,
               _Err=_Err)