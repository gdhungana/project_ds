import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats
import time
from mls import plot_rosenbrock, plot_posterior
#- Optimization

#- Derivatives
#- 1. by hand

def f(x):
    return np.cos(np.exp(x)) / x ** 2

def fp(x):
    return -2 * np.cos(np.exp(x)) / x ** 3 - np.exp(x) * np.sin(np.exp(x)) / x ** 2

x = np.linspace(1, 3, 50)
plt.plot(x, f(x), label='$f(x)$')
plt.plot(x, fp(x), '.-', lw=1, label='$f\'(x)$')
plt.legend();
plt.show()

#- 2. numerically : finite difference
fp_numeric=np.gradient(f(x),x)
plt.plot(x, (fp_numeric - fp(x)), '.-', lw=1, label='absolute error')
plt.plot(x, (fp_numeric - fp(x)) / fp(x), '.', label='relative error')
plt.legend();
plt.show()

#- 3. Automatic Differentiation
from autograd import grad, elementwise_grad
import autograd.numpy as anp

def f_auto(x):
    return anp.cos(anp.exp(x)) / x ** 2

fp_auto = elementwise_grad(f_auto)
plt.plot(x, fp_auto(x) - fp(x), '.-', lw=1);
plt.show()

def sinc(x):
    return anp.sin(x) / x if x != 0 else 1.

#- methods
def rosenbrock(x):
    x0, x1 = x
    return (1 - x0) ** 2 + 100.0 * (x1 - x0 ** 2) ** 2
plot_rosenbrock()
plt.show()

#- rosenbrock is not a convex function
x0 = np.linspace(-1.5, 1.5, 100)
plt.plot(x0, rosenbrock([x0, 1.0]))
plt.show()

#- scipy optimize
opt = minimize(rosenbrock, [-1, 0], method='Nelder-Mead', tol=1e-4)
print(opt.message, opt.x)

#- using jacobian
rosenbrock_grad = grad(rosenbrock)
opt = minimize(rosenbrock, [-1, 0], method='CG', jac=rosenbrock_grad, tol=1e-4)
print(opt.message, opt.x)


#- see the optimization develop
def optimize_rosenbrock(method, use_grad=False, x0=-1, y0=0, tol=1e-4):
    
    all_calls = []
    def rosenbrock_wrapped(x):
        all_calls.append(x)
        return rosenbrock(x)
    
    path = [(x0,y0)]
    def track(x):
        path.append(x)

    jac = rosenbrock_grad if use_grad else False
    
    start = time.time()
    opt = minimize(rosenbrock_wrapped, [x0, y0], method=method, jac=jac, tol=tol, callback=track)
    stop = time.time()
    
    assert opt.nfev == len(all_calls)
    njev = opt.get('njev', 0)    
    print('Error is ({:+.2g},{:+.2g}) after {} iterations making {}+{} calls in {:.2f} ms.'
          .format(*(opt.x - np.ones(2)), opt.nit, opt.nfev, njev, 1e3 * (stop - start)))    

    xrange, yrange = plot_rosenbrock(path=path, all_calls=all_calls)

optimize_rosenbrock(method='Nelder-Mead', use_grad=False)
plt.show()

#- Can use different methods: Nelder-Mead, CG, Newton-CG, Powell, BFGS
#- initial point can have big effect on optimization cost

def cost_map(method, tol=1e-4, ngrid=50):
    xrange, yrange = plot_rosenbrock(shaded=False)
    x0_vec = np.linspace(*xrange, ngrid)
    y0_vec = np.linspace(*yrange, ngrid)
    cost = np.empty((ngrid, ngrid))
    for i, x0 in enumerate(x0_vec):
        for j, y0 in enumerate(y0_vec):
            opt = minimize(rosenbrock, [x0, y0], method=method, tol=tol)
            cost[j, i] = opt.nfev
    plt.imshow(cost, origin='lower', extent=[*xrange, *yrange],
               interpolation='none', cmap='magma', aspect='auto', vmin=0, vmax=250)
    plt.colorbar().set_label('Number of calls')

cost_map('Nelder-Mead')
plt.show()

cost_map('BFGS')
plt.show()

#- stochastic optimization: loops on data
D = scipy.stats.norm.rvs(loc=0, scale=1, size=200, random_state=123)
x = np.linspace(-4, +4, 100)
plt.hist(D, range=(x[0], x[-1]), bins=20, normed=True)
plt.plot(x, scipy.stats.norm.pdf(x,loc=0,scale=1))
plt.xlim(x[0], x[-1]);
plt.show()

#- likelihood
def NLL(theta, D):
    mu, sigma = theta
    return anp.sum(0.5 * (D - mu) ** 2 / sigma ** 2 + 0.5 * anp.log(2 * anp.pi) + anp.log(sigma))

#-priors
def NLP(theta):
    mu, sigma = theta
    return -anp.log(sigma) if sigma > 0 else -anp.inf

#- posterior
def NLpost(theta, D):
    return NLL(theta, D) + NLP(theta)

plot_posterior(D);
plt.show()

#- Optimization with gradient decent
NLpost_grad = grad(NLpost)

#-step
def step(theta, D, eta):
    return theta - eta * NLpost_grad(theta, D) / len(D)

def GradientDescent(mu0, sigma0, eta, n_steps):
    path = [np.array([mu0, sigma0])]
    for i in range(n_steps):
        path.append(step(path[-1], D, eta))
    return path

plot_posterior(D, path=GradientDescent(mu0=-0.2, sigma0=1.3, eta=0.2, n_steps=15));
plt.show()

#- Stochastic Gradient Descent
def StochasticGradientDescent(mu0, sigma0, eta, n_minibatch, eta_factor=0.95, seed=123, n_steps=15):
    gen = np.random.RandomState(seed=seed)
    path = [np.array([mu0, sigma0])]
    for i in range(n_steps):
        minibatch = gen.choice(D, n_minibatch, replace=False)
        path.append(step(path[-1], minibatch, eta))
        eta *= eta_factor
    return path

plot_posterior(D, path=StochasticGradientDescent(
    mu0=-0.2, sigma0=1.3, eta=0.2, n_minibatch=100, n_steps=100));
plt.show()

#- with no decay of learning rate
plot_posterior(D, path=StochasticGradientDescent(
    mu0=-0.2, sigma0=1.3, eta=0.2, eta_factor=1, n_minibatch=100, n_steps=100))

#- smaller minibatch
plot_posterior(D, path=StochasticGradientDescent(
    mu0=-0.2, sigma0=1.3, eta=0.15, eta_factor=0.97, n_minibatch=20, n_steps=75))


