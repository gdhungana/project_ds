import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score

#- load data
def get_data(corr=False):
    white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
    red=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
    #- combine with a target varialbe
    white['type']=0.
    red['type']=1.
    wines = pd.concat([red,white],join='outer')
    
    #- check the correlation:
    if corr:
        import seaborn as sns 
        corr = wines.corr() 
        sns.heatmap(corr,  
                    xticklabels=corr.columns.values, 
                    yticklabels=corr.columns.values) 
        plt.show()    

    return wines



