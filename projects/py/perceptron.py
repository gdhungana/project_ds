import numpy as np

class Perceptron(object):
	""" Perceptron Classifier

	Parameters
	------------
	rate : float
		Learning rate (ranging from 0.0 to 1.0)
	number_of_iteration : int
		Number of iterations over the input dataset.

	Attributes:
	------------

	weight_matrix : 1d-array
		Weights after fitting.

	error_matrix : list
		Number of misclassification in every epoch(one full training cycle on the training set)

	"""

	def __init__(self, rate = 0.01, number_of_iterations = 100):
		self.rate = rate
		self.number_of_iterations = number_of_iterations

	def fit(self, X, y):
		""" Fit training data
		
		Parameters:
		------------
		X : array-like, shape = [number_of_samples, number_of_features]
			Training vectors.
		y : array-like, shape = [number_of_samples]
			Target values.

		Returns
		------------
		self : object

		"""
		
		self.weight_matrix = np.zeros(1 + X.shape[1])
		self.errors_list = []

		for _ in range(self.number_of_iterations):
			errors = 0
			for xi, target in zip(X, y):
				update = self.rate * (target - self.predict(xi))
				self.weight_matrix[1:] += update * xi
				self.weight_matrix[0] += update
				errors += int(update != 0.0)
				#print("this error",errors)
			self.errors_list.append(errors)
		return self

	def dot_product(self, X):
		""" Calculate the dot product """
		return (np.dot(X, self.weight_matrix[1:]) + self.weight_matrix[0])

	def predict(self, X):
		""" Predicting the label for the input data """
		return np.where(self.dot_product(X) >= 0.0, 1, 0)

"""
if __name__ == '__main__':
	X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
	y = np.array([0, 1, 1, 1, 1, 1, 1])
	p = Perceptron()
	p.fit(X, y)
	print("Predicting the output of [1, 1, 1] = {}".format(p.predict([1, 1, 1])))
"""

import pandas as pd
import numpy as np

def preprocess_df(X,target):
    (X-X.min())/(X.max()-X.min())
    bool={"Up":1, "NoSignal":0}
    y=target.map(bool)
    X= pd.concat([X,y],axis=1)
    #X=X.replace(np.nan,0)
    X=X.dropna()
    print(X.shape)
    print(X.columns)
    return X


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def do_random_forest(X):
    X_train,X_test,y_train,y_test=train_test_split(X.iloc[:,:-1],X.iloc[:,-1],test_size=0.2,random_state=12345)
    classifier=RandomForestClassifier(max_depth=2,random_state=12345)
    classifier.fit(X_train,y_train)
    y_pred=classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)  
    print(cm)  
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    return cm

def plot_confusion_matrix(cm, classes,normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    fig=plt.figure()
    ax=fig.add_subplot(121)
    im=ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im,ax=ax,fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes,rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = 'd' #if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ax1=fig.add_subplot(122)
        im1=ax1.imshow(cm, interpolation='nearest', cmap=cmap)
        ax1.set_title(title+' (normalized)')
        fig.colorbar(im1,ax=ax1,fraction=0.046,pad=0.04)
        tick_marks = np.arange(len(classes))
        ax1.set_xticks(tick_marks)
        ax1.set_xticklabels(classes,rotation=45)
        ax1.set_yticks(tick_marks)
        ax1.set_yticklabels(classes)

        fmt = '.2f' #if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax1.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        ax1.set_ylabel('True label')
        ax1.set_xlabel('Predicted label')
        
    fig.tight_layout()