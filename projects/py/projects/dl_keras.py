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


def train_test_data(dataDF):
    # Specify the data 
    X=dataDF.ix[:,0:-1]
    # Specify the target labels and flatten the array 
    y=np.ravel(dataDF.ix[:,-1])
    # Split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
    #- scale/normalization
    # Define the scaler 
    scaler = StandardScaler().fit(X_train)
    # Scale the train set
    X_train = scaler.transform(X_train)
    # Scale the test set
    X_test = scaler.transform(X_test)

    return X_train,X_test,y_train,y_test

def deep_learning_model_inst(inshape=11,hidshape=8):

    # Initialize the constructor
    model = Sequential()

    # Add an input layer 
    model.add(Dense(12, activation='relu', input_shape=(inshape,)))

    # Add one hidden layer 
    model.add(Dense(hidshape, activation='relu'))

    # Add an output layer 
    model.add(Dense(1, activation='sigmoid'))
    return model

def fit_model(model,XX_train,yy_train,epochs=5,loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']):
    model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    model.fit(XX_train,yy_train,epochs=epochs,batch_size=1,verbose=1)
    return model


def evaluate_model(model,XX_test,yy_test,round=False):
    y_pred = model.predict(XX_test)
    if round:
       y_pred=np.round(y_pred)
    score=model.evaluate(XX_test,yy_test,verbose=1)
    print(score)
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
    print("Confusion Matrix: ", confusion_matrix(yy_test, y_pred))
    print("Precision: ",precision_score(yy_test,y_pred))
    print("Recall: ",recall_score(yy_test,y_pred))
    print("F1 Score: ", f1_score(yy_test,y_pred))
    print("Cohen Kappa: ", cohen_kappa_score(yy_test, y_pred))
    return


