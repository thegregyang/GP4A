import numpy as np
import scipy as sp
import torch as th

def relu(c):
    return c * (c > 0)
    
class Transformer():
    def __init__(self, d, d_in, depth, temp=1, vw=1, vb=0):
        self.d = d
        self.depth = depth
        self.temp = temp
        self.vw = vw
        self.vb = vb
        self.W1s = [np.random.randn(d, d) * np.sqrt(vw/d) 
                   for _ in range(depth)]
        self.b1s = [np.random.randn(d) * np.sqrt(vb)
                    for _ in range(depth)]
        self.W2s = [np.random.randn(d, d) * np.sqrt(vw/d) 
                   for _ in range(depth)]
        self.b2s = [np.random.randn(d) * np.sqrt(vb)
                    for _ in range(depth)]
        self.embedding = np.random.randn(d_in, d) * np.sqrt(1/d_in)
        
    def __call__(self, seq):
        '''
        Input:
            seq: seqlen x tokensize array, for any seqlen and tokensize
        Output:
            out: seqlen x self.d_in array, for the same seqlen as input
        '''
        inseq = seq @ self.embedding
        for l in range(self.depth):
            # self attn
            gram = inseq @ inseq.T / inseq.shape[1]
            gram[np.triu_indices(gram.shape[0], 1)] = -np.inf
            weights = sp.special.softmax(gram / self.temp, axis=1)
            # weights @ inseq gives vectors returned by attention
            # inseq + weights @ inseq is the residual connection
            post_attn = self.layernorm(inseq + weights @ inseq)
            # self.post_attn = post_attn
            
            # FF
            inseq = relu(post_attn @ self.W1s[l] + self.b1s[l])
            inseq = inseq @ self.W2s[l] + self.b2s[l]
            inseq = self.layernorm(inseq + post_attn)
            
        return inseq
    def layernorm(self, seq):
        '''inplace layernorm
        Input:
            seq: seqlen x tokensize array, for any seqlen and tokensize
        Output:
            out: seqlen x tokensize array
                Means and standard deviation computed over the `tokensize` dimension
        '''
        seq -= np.mean(seq, axis=1, keepdims=True)
        seq /= np.std(seq, axis=1, keepdims=True)
        return seq
        
    def randomize(self, vw=None, vb=None):
        if vw is None:
            vw = self.vw
        if vb is None:
            vb = self.vb
        for p in self.W1s + self.W2s:
            # numpy has no way of sampling in place
            th.from_numpy(p).normal_(std=np.sqrt(vw / self.d))
        for p in self.b1s + self.b2s:
            th.from_numpy(p).normal_(std=np.sqrt(vb))