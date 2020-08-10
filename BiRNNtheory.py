import numpy as np

def thbirnn(ingram, inputidxs, Vphi, mergemode,
            varw=1, varu=1, varb=0, varv=1,
            maxlength=None):
    '''
    Computes the infinite-width GP kernel of a bidirectional erf-RNN over input sequences
    with normalized Gram matrix `ingram`.

    A bidirectional RNN with scalar output every time step evolves like

        s^t_f = nonlin(W s^{t-1}_f + U x^{t}_f + b)
        s^t_b = nonlin(W s^{t-1}_b + U x^{t}_b + b)
        y^t = <v, mergemode(s^t_f, s^t_b) >

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
        mergemode is the method of merging s^t_f and s^t_b, (s^t_f, s^t_b) for concatenation and (s^t_f + s^t_b) for sum
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
        mergemode: method of merging the two hidden states, either "concat" for concatenation or "sum" for sum
        `varw`: variance of state-to-state weights
        `varu`: variance of input-to-state weights
        `varb`: variance of biases
        `maxlength`: max length of all sequences. Default: None.
            In this case, it is calculated from `inputidxs`
    Outputs:
        The kernel of the Gaussian distribution described above.
    '''
    
    if mergemode is "concat":
        coingram = np.repeat(ingram[None, ...], 2, axis=0)

        loc = inputidxs[1]
        coingram[1][:loc, :loc] = ingram[:loc, :loc][::-1, ::-1]
        coingram[1][loc:, loc:] = ingram[loc:, loc:][::-1, ::-1]
        coingram[1][:loc, loc:] = ingram[:loc, loc:][::-1, ::-1]
        coingram[1][loc:, :loc] = ingram[loc:, :loc][::-1, ::-1]

        if maxlength is None:
            maxlength = 0
            for i in range(len(inputidxs)-1):
                maxlength = max(maxlength, inputidxs[i+1]-inputidxs[i])

        hcov = np.zeros(coingram.shape)
        hhcov = np.zeros(coingram.shape)

        for _ in range(maxlength):
            hhcov[..., 1:, 1:] = hcov[..., :-1, :-1]
            hhcov[..., inputidxs, :] = hhcov[..., :, inputidxs] = 0
            hhcov += varu * coingram + varb
            hcov = varw * Vphi(hhcov)
        return varv * (hcov[0]+hcov[1]) / 2

    elif mergemode is "sum":
        loc = inputidxs[1]
        ingram_input1 = np.zeros((loc*2,loc*2))
        ingram_input1[:loc, :loc] = ingram[:loc, :loc]
        ingram_input1[loc:, loc:] = ingram[:loc, :loc][::-1, ::-1]
        ingram_input1[:loc, loc:] = ingram[:loc, :loc][:, ::-1]
        ingram_input1[loc:, :loc] = ingram[:loc, :loc][::-1, :]
        
        hcov_input1 = np.zeros(ingram_input1.shape)
        hhcov_input1 = np.zeros(ingram_input1.shape)
        for _ in range(loc):
            hhcov_input1[1:, 1:] = hcov_input1[:-1, :-1]
            hhcov_input1[inputidxs, :] = hhcov_input1[:, inputidxs] = 0
            hhcov_input1 += varu * ingram_input1 + varb
            hcov_input1 = varw * Vphi(hhcov_input1)

        loc2 = ingram.shape[0]-loc
        
        ingram_input2 = np.zeros((loc2*2, loc2*2))
        ingram_input2[:loc2, :loc2] = ingram[loc:, loc:]
        ingram_input2[loc2:, loc2:] = ingram[loc:, loc:][::-1, ::-1]
        ingram_input2[:loc2, loc2:] = ingram[loc:, loc:][:, ::-1]
        ingram_input2[loc2:, :loc2] = ingram[loc:, loc:][::-1, :]

        hcov_input2 = np.zeros(ingram_input2.shape)
        hhcov_input2 = np.zeros(ingram_input2.shape)
        for _ in range(loc2):
            hhcov_input2[1:, 1:] = hcov_input2[:-1, :-1]
            hhcov_input2[[0, loc2], :] = hhcov_input2[:, [0, loc2]] = 0
            hhcov_input2 += varu * ingram_input2 + varb
            hcov_input2 = varw * Vphi(hhcov_input2)
        
        coingram = np.repeat(ingram[None, ...], 4, axis=0)

        loc = inputidxs[1]
        coingram[1][:loc, :loc] = ingram[:loc, :loc][::-1, ::-1]
        coingram[1][loc:, loc:] = ingram[loc:, loc:][::-1, ::-1]
        coingram[1][:loc, loc:] = ingram[:loc, loc:][::-1, ::-1]
        coingram[1][loc:, :loc] = ingram[loc:, :loc][::-1, ::-1]

        coingram[2][:loc, :loc] = ingram[:loc, :loc]
        coingram[2][loc:, loc:] = ingram[loc:, loc:][::-1, ::-1]
        coingram[2][:loc, loc:] = ingram[:loc, loc:][::-1, :]
        coingram[2][loc:, :loc] = (ingram[:loc, loc:][::-1, :]).T
        
        coingram[3][:loc, :loc] = ingram[:loc, :loc][::-1, ::-1]
        coingram[3][loc:, loc:] = ingram[loc:, loc:]
        coingram[3][:loc, loc:] = ingram[:loc, loc:][:, ::-1]
        coingram[3][loc:, :loc] = (ingram[:loc, loc:][:, ::-1]).T

        if maxlength is None:
            maxlength = 0
            for i in range(len(inputidxs)-1):
                maxlength = max(maxlength, inputidxs[i+1]-inputidxs[i])

        hcov = np.zeros(coingram.shape)
        hhcov = np.zeros(coingram.shape)

        for _ in range(maxlength):
            hhcov[..., 1:, 1:] = hcov[..., :-1, :-1]
            hhcov[..., inputidxs, :] = hhcov[..., :, inputidxs] = 0
            hhcov += varu * coingram + varb
            hcov = varw * Vphi(hhcov)

        hcov[2][:loc, :loc] = hcov_input1[:loc, loc:]
        hcov[3][:loc, :loc] = hcov_input1[loc:, :loc]

        hcov[2][loc:, loc:] = hcov_input2[:loc2, loc2:]
        hcov[3][loc:, loc:] = hcov_input2[loc2:, :loc2]

        return varv * (np.sum(hcov, axis=0))

    else:
        raise NotImplementedError()