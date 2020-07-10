import numpy as np

def thbirnn(ingram, inputidxs, Vphi,
            varw=1, varu=1, varb=0, varv=1,
            maxlength=None):
    '''
    Computes the infinite-width GP kernel of a bidirectional erf-RNN over input sequences
    with normalized Gram matrix `ingram`.

    A bidirectional RNN with scalar output every time step evolves like

        s^t_f = nonlin(W s^{t-1}_f + U x^{t}_f + b)
        s^t_b = nonlin(W s^{t-1}_b + U x^{t}_b + b)
        y^t = <v, (s^t_f, s^t_b) >

    where

        x^t_f is the forward input at time t
        x^t_b is the backward input at time t
        s^t_f is the forward state of the RNN at time t
        s^t_b is the backward state of the RNN at time t
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

    ingram_b = np.zeros(ingram.shape)
    loc = inputidxs[1]
    ingram_b[:loc, :loc] = ingram[:loc, :loc][::-1, ::-1].T
    ingram_b[loc:, loc:] = ingram[loc:, loc:][::-1, ::-1].T
    ingram_b[:loc, loc:] = ingram[loc:, :loc][::-1, ::-1].T
    ingram_b[loc:, :loc] = ingram[:loc, loc:][::-1, ::-1].T

    if maxlength is None:
        maxlength = 0
        for i in range(len(inputidxs)-1):
            maxlength = max(maxlength, inputidxs[i+1]-inputidxs[i])

    hcov_f = np.zeros(ingram.shape)
    hhcov_f = np.zeros(ingram.shape)

    hcov_b = np.zeros(ingram.shape)
    hhcov_b = np.zeros(ingram.shape)

    for _ in range(maxlength):
        hhcov_f[1:, 1:] = hcov_f[:-1, :-1]
        hhcov_f[inputidxs, :] = hhcov_f[:, inputidxs] = 0
        hhcov_f += varu * ingram + varb
        hcov_f = varw * Vphi(hhcov_f)
        
        hhcov_b[1:, 1:] = hcov_b[:-1, :-1]
        hhcov_b[inputidxs, :] = hhcov_b[:, inputidxs] = 0
        hhcov_b += varu * ingram_b + varb
        hcov_b = varw * Vphi(hhcov_b)
    return varv * (hcov_f+hcov_b) / 2