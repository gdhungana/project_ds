import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn import preprocessing, pipeline, model_selection, linear_model
import tensorflow as tf

#import edward as ed
#import edward.models

#- generate some data for a projectile
xlo, xhi = 0., 1.
poly_coefs = np.array([-1., 2., -1.])
f = lambda X: np.dot(poly_coefs, [X ** 0, X ** 1, X ** 2])
sigma_y = 0.2


def generate(N, seed=123):
    gen = np.random.RandomState(seed=seed)
    X = gen.uniform(xlo, xhi, size=N)
    y = f(X) + gen.normal(scale=sigma_y, size=N)
    return X, y

#Compare the results with different samples of the same model

Xa, ya = generate(N=15)
Xb, yb = generate(N=150)

def plotXy(X, y, ax=None, *fits):
    ax = ax or plt.gca()
    ax.scatter(X, y, s=25, lw=0)
    x_grid = np.linspace(xlo, xhi, 100)
    ax.plot(x_grid, f(np.array(x_grid)), '-', lw=10, alpha=0.2)
    for fit in fits:
        y_fit = fit.predict(x_grid.reshape(-1, 1))
        ax.plot(x_grid, y_fit, lw=2, alpha=1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ylo, yhi = np.percentile(y, (0, 100))
    dy = yhi - ylo
    ax.set_ylim(ylo - 0.1 * dy, yhi + 0.1 * dy)
    
_, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
plotXy(Xa, ya, ax[0])
plotXy(Xb, yb, ax[1])
plt.show()

#- Combine preprocessing and linear fit in a pipeline
def poly_fit(X, y, degree):
    degree_is_zero = (degree == 0)
    model = pipeline.Pipeline([
        ('poly', preprocessing.PolynomialFeatures(degree=degree, include_bias=degree_is_zero)),
        ('linear', linear_model.LinearRegression(fit_intercept=not degree_is_zero))])
    return model.fit(X.reshape(-1, 1), y)


#- see for different deg of polynomials: 0,1,2,14
_, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
plotXy(Xa, ya, ax[0], poly_fit(Xa, ya, 0), poly_fit(Xa, ya, 1), poly_fit(Xa, ya, 2), poly_fit(Xa, ya, 14))
plotXy(Xb, yb, ax[1], poly_fit(Xb, yb, 0), poly_fit(Xb, yb, 1), poly_fit(Xb, yb, 2), poly_fit(Xb, yb, 14))
plt.show()

#- Train-test split

def train_test_split(X, y, degree, test_fraction=0.2, ax=None, seed=123):
    gen = np.random.RandomState(seed=seed)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_fraction, random_state=gen)
    train_fit = poly_fit(X_train, y_train, degree)
    plotXy(X, y, ax, train_fit)
    test_R2 = train_fit.score(X_test.reshape(-1, 1), y_test)
    ax.scatter(X_test, y_test, marker='x', color='r', s=40, zorder=10)
    ax.text(0.7, 0.1, '$R^2={:.2f}$'.format(test_R2), transform=ax.transAxes, fontsize='x-large', color='r')

_, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
train_test_split(Xa, ya, 2, ax=ax[0])
train_test_split(Xb, yb, 2, ax=ax[1])
plt.show()


#-Scan the splitting
def test_fraction_scan(degree=2, seed=123):
    gen = np.random.RandomState(seed=seed)
    test_fractions = np.arange(0.05, 0.6, 0.025)
    R2 = np.empty((2, len(test_fractions)))
    for i, test_fraction in enumerate(test_fractions):
        for j, (X, y) in enumerate(((Xa, ya), (Xb, yb))):
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=test_fraction, random_state=gen)
            fit = poly_fit(X_train, y_train, degree)
            R2[j, i] = fit.score(X_test.reshape(-1, 1), y_test)
    plt.plot(test_fractions, R2[0], 'o:', label='$N = {}$'.format(len(ya)))
    plt.plot(test_fractions, R2[1], 'o:', label='$N = {}$'.format(len(yb)))
    plt.xlabel('Test fraction')
    plt.ylabel('Test score $R^2$')
    plt.ylim(-2, 1)
    plt.legend()
    
test_fraction_scan()
plt.show()

