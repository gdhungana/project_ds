import lifelines
print(lifelines.__version__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import datetime
import json
import sys,os
from lifelines.datasets import load_waltons
from lifelines import KaplanMeierFitter, NelsonAalenFitter

#- lifelines from https://lifelines.readthedocs.io/en/latest/index.html
print(lifelines.__version__)

def plot_kmf(kmfmodel,xrange=None,label=r'$KM-estimate$'):
    sf=kmfmodel.survival_function_
    plt.step(sf.index,sf.iloc[:,0].values,label=label)#sf.shape
    kcfup=kmfmodel.confidence_interval_.iloc[:,0].values
    kcfdn=kmfmodel.confidence_interval_.iloc[:,1].values
    if xrange is not None:
        plt.xlim(xrange)
    plt.legend()
    plt.xlabel(r'$Time\ [days]$')
    plt.ylabel(r'$Survival\ function$')
    print(len(sf.index),len(kcfdn),len(kcfup))
    plt.fill_between(sf.index,kcfdn,kcfup,color='blue',step='pre',alpha=0.3)



