from utils import getCor, colorbar
from utils import VErf, VReLU, VStep
from BiRNNtheory import thbirnn
from BiRNNsim import simbirnn
import numpy as np
import scipy.special as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

sent1 = "The brown fox jumps over the dog".split()
sent2 = "The quick brown fox jumps over the lazy dog".split()

with open('ExampleGloVeVecs.npy', 'rb') as f:
    exampleGloveVecs = np.load(f)
with open('ExampleGloVeCov.npy', 'rb') as f:
    exampleGloveCov = np.load(f)

varw = 1
varu = 1
varb = 0
# if want to do ReLU then use `VReLU` instead of `VErf`; similarly for step function
thcov = thbirnn(exampleGloveCov, [0, 7], VErf, varw, varu, varb)

plt.figure(figsize=(8, 8))

plt.subplot(221)
ax = plt.gca()
im_thcov = plt.imshow(thcov, cmap='PuBu_r')
span = np.linspace(-.5, 15.5)
plt.plot(span, [6.5]*len(span), 'r')
plt.plot([6.5]*len(span), span, 'r')
plt.yticks(np.arange(16), sent1+sent2)
plt.xticks([])
plt.title('RNN covariances (theory)')
plt.ylabel('sent2                       sent1')
plt.grid()
colorbar(im_thcov)

plt.subplot(222)
ax = plt.gca()
im_thcor = plt.imshow(getCor(thcov), cmap='viridis')
span = np.linspace(-.5, 15.5)
plt.plot(span, [6.5]*len(span), 'r')
plt.plot([6.5]*len(span), span, 'r')
plt.yticks([])
plt.xticks([])
plt.title('RNN correlations (theory)')
plt.grid()
colorbar(im_thcor)


plt.subplot(223)
ax = plt.gca()
im_glove = plt.imshow(exampleGloveCov, cmap='PuBu_r')
span = np.linspace(-.5, 15.5)
plt.plot(span, [6.5]*len(span), 'r')
plt.plot([6.5]*len(span), span, 'r')
plt.yticks(np.arange(16), sent1+sent2)
plt.xticks(np.arange(16), sent1+sent2, rotation=90)
plt.title('GloVe covariances')
plt.xlabel('sent1                       sent2')
plt.ylabel('sent2                       sent1')
plt.grid()
colorbar(im_glove)

plt.subplot(224)
ax = plt.gca()
im_glovecor = plt.imshow(getCor(exampleGloveCov), cmap='viridis')
span = np.linspace(-.5, 15.5)
plt.plot(span, [6.5]*len(span), 'r')
plt.plot([6.5]*len(span), span, 'r')
plt.yticks([])
plt.xticks(np.arange(16), sent1+sent2, rotation=90)
plt.title('GloVe correlations')
plt.xlabel('sent1                       sent2')
plt.grid()
colorbar(im_glovecor)

plt.tight_layout()

plt.show()

# with open('RNN.kernel', 'wb') as f:
# np.save(f, thcov)

nsamples = 100
widths = [2**i for i in range(5, 13)]
# widths = [2**i for i in range(5, 6)]
mysimcovs = {}
for width in widths:
    mysimcovs[width] = np.array([
        simbirnn([exampleGloveVecs[:7], exampleGloveVecs[7:]],
                 width, F.erf, varw, varu, varb, seed=seed)[1]
        for seed in range(nsamples)])
frobs = []
for width in widths:
    _frobs = np.sum((mysimcovs[width] - thcov)**2,
                    axis=(1, 2)) / np.linalg.norm(thcov)**2
    for f in _frobs:
        frobs.append(dict(
            relfrob=f,
            width=width
        ))


# plt.figure(figsize=(8, 8))

# plt.subplot(221)
# ax = plt.gca()
# im_thcov = plt.imshow( np.sum(mysimcovs[2**5], axis=(0,)) / 100, cmap='PuBu_r')
# span = np.linspace(-.5, 15.5)
# plt.plot(span, [6.5]*len(span), 'r')
# plt.plot([6.5]*len(span), span, 'r')
# plt.yticks(np.arange(16), sent1+sent2)
# plt.xticks([])
# plt.title('RNN covariances (theory)')
# plt.ylabel('sent2                       sent1')
# plt.grid()
# colorbar(im_thcov)

# plt.tight_layout()

# plt.show()













frob_df = pd.DataFrame(frobs)

frob_df.to_pickle('RNN.df')


sns.boxplot(x='width', y='relfrob', data=frob_df)
plt.semilogy()
# plt.legend()
plt.title('Deviation From Infinite-width Theory')
_ = plt.ylabel(
    u'Relative Squared Frob. Norm\n $\|K_{\infty} - K_{width}\|_F^2/\|K_{\infty}\|_F^2$')

frob_df.groupby('width', as_index=False).mean(
).plot.line(x='width', y='relfrob')
plt.plot(widths, np.array(widths, dtype='float')
         ** -1, '--', label=u'${width}^{-1}$')
plt.ylabel(
    u'Mean Relative Squared Frob. Norm\n $\|K_{\infty} - K_{width}\|_F^2/\|K_{\infty}\|_F^2$')
plt.loglog()
plt.legend()
_ = plt.title(
    u'Deviation from theory in (Frobenius norm)$^2$ drops like $width^{-1}$')

plt.show()
