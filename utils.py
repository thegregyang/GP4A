import numpy as np

def getCor(cov):
    d = np.diag(cov)**-0.5
    return d[:, None] * cov * d


def seqs2cov(inputs):
    inputs = np.array(inputs)
    inputcov = inputs @ inputs.T / len(inputs[0])
    
    inputidxs = [0]
    for seq in inputs:
        inputidxs.append(inputidxs[-1] + len(seq))
    inputidxs.pop()
    return inputs, inputcov, inputidxs
	
def VStep(cov):
    '''
    Computes E[step(z) step(z)^T | z ~ N(0, `cov`)]
    where step is the function takes positive numbers to 1 and
    all else to 0, and 
    z is a multivariate Gaussian with mean 0 and covariance `cov`
    
    Inputs:
        `cov`: An array where the last 2 dimensions contain covariance matrix of z (and the first dimensions are "batch" dimensions)
    Output:
        a numpy array of the same shape as `cov` that equals the
        expectation above in the last 2 dimensions.
    '''
    ll = list(range(cov.shape[-1]))
    d = np.sqrt(cov[..., ll, ll])
    c = d[..., None]**(-1) * cov * d[..., None, :]**(-1)
    return 2./np.pi * np.arcsin(np.clip(c, -1, 1))

def VErf(cov):
    '''
    Computes E[erf(z) erf(z)^T | z ~ N(0, `cov`)]
    where z is a multivariate Gaussian with mean 0 and covariance `cov`
    
    Inputs:
        `cov`: An array where the last 2 dimensions contain covariance matrix of z (and the first dimensions are "batch" dimensions)
    Output:
        a numpy array of the same shape as `cov` that equals the
        expectation above in the last 2 dimensions.
    '''
    ll = list(range(cov.shape[-1]))
    d = np.sqrt(cov[..., ll, ll] + 0.5)
    
    c = d[..., None]**(-1) * cov * d[..., None, :]**(-1)
    return 2./np.pi * np.arcsin(np.clip(c, -1, 1))

def VDerErf(cov):
    '''
    Computes E[erf'(z) erf'(z)^T | z ~ N(0, `cov`)]
    where erf' is the derivative of erf and
    z is a multivariate Gaussian with mean 0 and covariance `cov`
    
    Inputs:
        `cov`: An array where the last 2 dimensions contain covariance matrix of z (and the first dimensions are "batch" dimensions)
    Output:
        a numpy array of the same shape as `cov` that equals the
        expectation above in the last 2 dimensions.
    '''
    ll = list(range(cov.shape[-1]))
    d = np.sqrt(cov[..., ll, ll])
    dd = 1 + 2 * d
    return 4/np.pi * (dd[..., None] * dd[..., None, :] - 4 * cov**2)**(-1./2)

def J1(c, eps=1e-10):
    c[c > 1-eps] = 1-eps
    c[c < -1+eps] = -1+eps
    return (np.sqrt(1-c**2) + (np.pi - np.arccos(c)) * c) / np.pi

def VReLU(cov, eps=1e-5):
    ll = list(range(cov.shape[-1]))
    d = np.sqrt(cov[..., ll, ll])
    c = d[..., None]**(-1) * cov * d[..., None, :]**(-1)
    return np.nan_to_num(0.5 * d[..., None] * J1(c, eps=eps) * d[..., None, :])

from mpl_toolkits.axes_grid1 import make_axes_locatable
def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)