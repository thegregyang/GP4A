import numpy as np

def thrnn(ingram, inputidxs, Vphi,
    varw=1, varu=1, varb=0, varv=1,
    maxlength=None):
    '''
    Computes the infinite-width GP kernel of an erf-RNN over input sequences
    with normalized Gram matrix `ingram`.

    A simple RNN with scalar output every time step evolves like

        s^t = nonlin(W s^{t-1} + U x^{t} + b)
        y^t = <v, s^t>

    where

        x^t is the input at time t
        s^t is the state of the RNN at time t
        W is the state-to-state weight matrix
        U is the input-to-state weight matrix
        b is the bias
        v is the readout weight
        y^t is the output at time t
        nonlin = erf in the case of erf-RNN here

    If n is the number of neurons and x^t has dimension m, and

        W_{pq} ~ N(0, varw / n)
        U_{pq} ~ N(0, varu / m)
        b_p ~ N(0, varb)
        v_p ~ N(0, varv / n)

    then all outputs of an RNN over all time steps and all input sequences
    are jointly Gaussian in the limit as n -> infinity.
    This Gaussian distribution has dimension

        \sum_seq length(seq)

    It has zero mean, and in this function we calculate its kernel.
        
    Inputs:
        `ingram`: normalized Gram matrix between all tokens across all input sequences
        `inputidxs`: indices of `ingram` that indicate starts of input sequences
        `Vphi`: V transform of the nonlinearity of the RNN (e.g. arcsin for step function, etc)
        `varw`: variance of state-to-state weights
        `varu`: variance of input-to-state weights
        `varb`: variance of biases
        `maxlength`: max length of all sequences. Default: None.
            In this case, it is calculated from `inputidxs`
    Outputs:
        The kernel of the Gaussian distribution described above.
    '''
    if maxlength is None:
        maxlength = 0
        for i in range(len(inputidxs)-1):
            maxlength = max(maxlength, inputidxs[i+1]-inputidxs[i])
    hcov = np.zeros(ingram.shape)
    hhcov = np.zeros(ingram.shape)
    for _ in range(maxlength):
        hhcov[1:, 1:] = hcov[:-1, :-1]
        hhcov[inputidxs, :] = hhcov[:, inputidxs] = 0
        hhcov += varu * ingram + varb
        hcov = varw * Vphi(hhcov)
    return varv * hcov
