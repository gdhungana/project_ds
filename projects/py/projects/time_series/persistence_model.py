from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
 
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')
 
def load_data(csvfile):
    series = read_csv(csvfile, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    return series

# Create lagged dataset
def get_train_test_data(series,trnsize=0.66):
    values = DataFrame(series.values)
    dataframe = concat([values.shift(1), values], axis=1)
    dataframe.columns = ['t-1', 't+1']
    print(dataframe.head(5))
 
    # split into train and test sets
    X = dataframe.values
    train_size = int(len(X) * trnsize)
    train, test = X[1:train_size], X[train_size:]
    train_X, train_y = train[:,0], train[:,1]
    test_X, test_y = test[:,0], test[:,1]
    return train_X,train_y,test_X,test_y
 
# persistence model
def model_persistence(x):
    return x
 
# walk-forward validation
def get_prediction(test_X,test_y):
    predictions = list()
    for x in test_X:
        yhat = model_persistence(x)
        predictions.append(yhat)
    test_score = mean_squared_error(test_y, predictions)
    print('Test MSE: %.3f' % test_score)
    return predictions
 
# plot predictions and expected results
def plot_predictions(train_y,test_y,predictions):
    pyplot.plot(train_y)
    pyplot.plot([None for i in train_y] + [x for x in test_y])
    pyplot.plot([None for i in train_y] + [x for x in predictions])
    pyplot.show()

