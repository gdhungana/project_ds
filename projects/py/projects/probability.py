import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn import mixture

#- 2d Gaussian and conditional probability

def prob2d(XY, lim=(-2.5,+2.5), n=100):
    grid = np.linspace(*lim, n)
    xy = np.stack(np.meshgrid(grid, grid)).reshape(2, -1).T

    data = pd.DataFrame(XY, columns=['x', 'y'])
    fitxy = mixture.GaussianMixture(n_components=1).fit(data)
    fitx = mixture.GaussianMixture(n_components=1).fit(data.drop(columns='y'))
    fity = mixture.GaussianMixture(n_components=1).fit(data.drop(columns='x'))
    
    pdfxy = np.exp(fitxy.score_samples(xy)).reshape(n, n)
    pdfx = np.exp(fitx.score_samples(grid.reshape(-1, 1))).reshape(-1)
    pdfy = np.exp(fity.score_samples(grid.reshape(-1, 1))).reshape(-1)
    xmarg = pdfxy[:, n // 3].copy()
    xmarg /= np.trapz(xmarg, grid)
    ymarg = pdfxy[n // 2, :].copy()
    ymarg /= np.trapz(ymarg, grid)
    
    g = sns.JointGrid('x', 'y', data, ratio=2, xlim=lim, ylim=lim, size=8)
    g.ax_joint.imshow(pdfxy, extent=lim+lim, origin='lower', interpolation='none')
    g.ax_joint.text(-1.6, 1.4, 'A', color='w', fontsize=18)
    g.ax_marg_x.plot(grid, pdfx, label='B')
    g.ax_marg_x.plot(grid, ymarg, 'r--', label='C')
    g.ax_marg_x.legend(fontsize='x-large')
    g.ax_marg_y.plot(pdfy, grid, label='D')
    g.ax_marg_y.plot(xmarg, grid, 'r--', label='E')
    g.ax_marg_y.legend(fontsize='x-large')

gen = np.random.RandomState(seed=123)
prob2d(gen.multivariate_normal([0,0], [[2,1],[1,1]], size=5000))
plt.show()

#- independent data- use covariance
gen = np.random.RandomState(seed=123)
prob2d(gen.multivariate_normal([0,0], [[1,0],[0,0.4]], size=5000))
plt.show()


