import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import scipy.linalg
from sklearn import linear_model

#- Read data 
dest='/home/govinda/learn/MachineLearningStatistics/mls/data/'
D = pd.read_csv(dest+'line_data.csv')
X = D['x'].values.reshape(-1, 1)
y = D['y'].values
y_true = pd.read_csv(dest+'line_targets.csv')['y_true'].values

#- fit
fit = linear_model.LinearRegression(fit_intercept=True).fit(X, y)

#- params:
W = fit.coef_
y0 = fit.intercept_

#- Plot

plt.scatter(X[:, 0], y, label='D=(X,y)', lw=0, s=5, c='gray')
plt.plot(X[:, 0], X.dot(W) + y0, 'r-', lw=8, alpha=0.5, label='fit')
plt.plot(X[:, 0], y_true, 'k-', label='truth')
plt.legend(fontsize='x-large')
plt.xlabel('$X$')
plt.ylabel('$y$')
plt.xlim(-1., +1.);
plt.show()

#- with errors
Dvalid = D.dropna()
X = Dvalid['x'].values.reshape(-1, 1)
y = Dvalid['y'].values
dy = Dvalid['dy'].values

fit_w = linear_model.LinearRegression().fit(X, y, dy ** 2)
W_w = fit_w.coef_ 
y0_w = fit_w.intercept_

#- predict:
X_new = np.array([[-1], [0], [+1]])
y_new = fit_w.predict(X_new) #- these are mean point estimates 

slope = W_w[0]
plt.scatter(X_new[:, 0], y_new);
plt.plot([-1, +1], [y0-slope, y0+slope], 'r--');
plt.show()

#- Errors on the parameters: linear_fit doesn't do this
C = np.diag(dy ** 2)
E = np.linalg.inv(np.dot(X.T, np.dot(C, X)))

dslope=np.sqrt(E[0,0])

plt.scatter(X_new[:, 0], y_new);
plt.fill_between([-1, +1], [y0-slope+dslope, y0+slope-dslope],
                 [y0-slope-dslope, y0+slope+dslope], color='r', alpha=0.5)
plt.xlabel('$X$')
plt.ylabel('$y$');
plt.show()

