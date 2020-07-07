import numpy as np


def thbirnn(ingram, inputidxs, Vphi,
            varw=1, varu=1, varb=0, varv=1,
            maxlength=None):
    '''
    Computes the infinite-width GP kernel of a bidirectional erf-RNN over input sequences
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

    ingram_b = np.zeros(ingram.shape)
    ingram_b[:7, :7] = ingram[:7, :7][::-1, ::-1].T
    ingram_b[7:, 7:] = ingram[7:, 7:][::-1, ::-1].T
    ingram_b[:7, 7:] = ingram[7:, :7][::-1, ::-1].T
    ingram_b[7:, :7] = ingram[:7, 7:][::-1, ::-1].T
    # ingram_b[:7, 7:] = ingram[:7, 7:][::-1, ::-1]
    # ingram_b[7:, :7] = ingram[7:, :7][::-1, ::-1]
    

    if maxlength is None:
        maxlength = 0
        for i in range(len(inputidxs)-1):
            maxlength = max(maxlength, inputidxs[i+1]-inputidxs[i])

    hcov1 = np.zeros(ingram.shape)
    hhcov1 = np.zeros(ingram.shape)

    hcov2 = np.zeros(ingram.shape)
    hhcov2 = np.zeros(ingram.shape)

    for _ in range(maxlength):
        hhcov1[1:, 1:] = hcov1[:-1, :-1]
        hhcov1[inputidxs, :] = hhcov1[:, inputidxs] = 0
        hhcov1 += varu * ingram + varb
        hcov1 = varw * Vphi(hhcov1)
        
        hhcov2[1:, 1:] = hcov2[:-1, :-1]
        hhcov2[inputidxs, :] = hhcov2[:, inputidxs] = 0
        hhcov2 += varu * ingram_b + varb
        hcov2 = varw * Vphi(hhcov2)

        # hcov1 = hcov2= hcov1+hcov2
        
    return varv * (hcov1+hcov2) / 2
