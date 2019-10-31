import numpy as np
import scipy as sp
from utils import VReLU_m, getCor

def _trsfmr(ingram, temp=1, vw=1, vb=0):
	ingram_ = np.copy(ingram)
	ingram_[np.triu_indices(ingram.shape[0], 1)] = -np.inf
	K = np.eye(ingram.shape[0]) + sp.special.softmax(ingram_ / temp, axis=1)
	post_res_attn = K @ ingram @ K.T
	# do layernorm
	layer1 = getCor(post_res_attn)
	
	post_ffn = vw * VReLU_m(vw * layer1 + vb) + vb
	layer2 = getCor(post_ffn)
	return layer2

def _trsfmr2(ingram, seq2idx, temp=1, vw=1, vb=0):
	ingram1 = np.copy(ingram[:seq2idx, :seq2idx])
	ingram2 = np.copy(ingram[seq2idx:, seq2idx:])
	ingram1[np.triu_indices(ingram1.shape[0], 1)] = -np.inf
	ingram2[np.triu_indices(ingram2.shape[0], 1)] = -np.inf
	K1 = np.eye(ingram1.shape[0]) \
		+ sp.special.softmax(ingram1 / temp, axis=1)
	K2 = np.eye(ingram2.shape[0]) \
		+ sp.special.softmax(ingram2 / temp, axis=1)
	post_res_attn = np.zeros_like(ingram)
	post_res_attn[:seq2idx, :seq2idx] = K1 @ ingram[:seq2idx, :seq2idx] @ K1.T
	post_res_attn[seq2idx:, seq2idx:] = K2 @ ingram[seq2idx:, seq2idx:] @ K2.T
	post_res_attn[:seq2idx, seq2idx:] = K1 @ ingram[:seq2idx, seq2idx:] @ K2.T
	post_res_attn[seq2idx:, :seq2idx] = post_res_attn[:seq2idx, seq2idx:].T
	# do layernorm
	layer1 = getCor(post_res_attn)
	
	post_ffn = vw * VReLU_m(vw * layer1 + vb) + vb
	layer2 = getCor(post_ffn + layer1)
	return {'layer1': layer1, 'layer2': layer2}

def th_trsfmr(ingram, seq2idx, depth, temp=1, vw=1, vb=0):
	r'''
	Computes the gram matrix for a single-head transformer
	running over 2 sequences
	Inputs:
		ingram: matrix.
			The gram matrix for the tokens of the two sequences
		seq2idx: the index such that
			`ingram[:seq2idx, :seq2idx]` is the gram matrix
			 of the first sequence, and
			 `ingram[seq2idx:, seq2idx:]` is the gram matrix
			 of the second sequence.
		depth: number of layers
		temp: temperature for softmax
		vw: variance of weights
		vb: variance of bias
	Outputs:
		the gram matrix of the same size as `ingram`
		of the output tokens of the transformer
	'''
	for l in range(depth):
		ingram = _trsfmr2(ingram, seq2idx)['layer2']
	return ingram