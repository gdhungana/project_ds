import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import scipy.linalg
from sklearn import linear_model

#- Generate some pulse data with smooth Gaussian response
def plot_response(R, ax):
    lim = np.percentile(np.abs(R), 99)
    img = ax.imshow(R, interpolation='none', cmap='bwr', vmin=-lim, vmax=+lim)
    plt.colorbar(img, ax=ax, orientation='horizontal', pad=0.01, fraction=0.1)
    ax.axis('off')

def generate(N=5000, n=50, tlo=0, thi=5, nplot=3, sigma_y = 0., seed=123):
    gen = np.random.RandomState(seed=seed)
    t_range = thi - tlo
    t0 = gen.uniform(tlo + 0.4 * t_range, thi - 0.4 * t_range, size=(N, 1))
    sigma = gen.uniform(0.05 * t_range, 0.15 * t_range, size=(N, 1))
    y0 = 1 + gen.rayleigh(size = (N, 1))
    t_grid = np.linspace(tlo, thi, n)
    X = y0 * np.exp(-0.5 * (t_grid - t0) ** 2 / sigma ** 2)
    r = np.exp(-0.5 * t_grid ** 2 / (t_range / 10) ** 2)
    R = scipy.linalg.toeplitz(r)
    Y = X.dot(R)
    Y += gen.normal(scale=sigma_y, size=Y.shape)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(nplot):
        ax[0].plot(t_grid, X[i], '.-', lw=1)
        ax[2].plot(t_grid, Y[i], '.-', lw=1)
    plot_response(R, ax[1])
    return X, Y

X,Y= generate()
plt.show()

n=X.shape[1]
fit1 = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
fit2 = linear_model.LinearRegression(fit_intercept=False).fit(X[:n**2], Y[:n**2])
fit3 = linear_model.LinearRegression(fit_intercept=False).fit(X[:n], Y[:n])
R1, R2, R3 = fit1.coef_, fit2.coef_, fit3.coef_

def plot_responses(*Rlist):
    n = len(Rlist)
    fig, ax = plt.subplots(1, n, figsize=(4 * n, 4))
    for i, R in enumerate(Rlist):
        plot_response(R, ax[i])
        
plot_responses(R1, R2, R3)
plt.show()

#- Add noise
X,Y=generate(sigma_y=1)
plt.show()

fit1 = linear_model.LinearRegression(fit_intercept=False).fit(X, Y)
fit2 = linear_model.LinearRegression(fit_intercept=False).fit(X[:n**2], Y[:n**2])
fit3 = linear_model.LinearRegression(fit_intercept=False).fit(X[:n], Y[:n])
R1, R2, R3 = fit1.coef_, fit2.coef_, fit3.coef_

plot_responses(R1,R2,R3)
plt.show()

#- Ridge regression
alpha = 0.1
fit1 = linear_model.Ridge(fit_intercept=False, alpha=alpha).fit(X, Y)
fit2 = linear_model.Ridge(fit_intercept=False, alpha=alpha).fit(X[:n**2], Y[:n**2])
fit3 = linear_model.Ridge(fit_intercept=False, alpha=alpha).fit(X[:n], Y[:n])
R1, R2, R3 = fit1.coef_, fit2.coef_, fit3.coef_

plot_responses(R1,R2,R3)
plt.show()

