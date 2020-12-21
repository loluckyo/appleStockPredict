from DataUtils import *
from Model import *
import pandas_datareader as web

if __name__ == "__main__":
    df = web.DataReader('GOOG', data_source='yahoo', start='2012-01-01', end='2020-12-17')

    # pre-processing data
    x_train, y_train, x_test, y_test = split_data(df, features='Close')

    # train model
    input_shape = (x_train.shape[1], 1)
    model = Model(input_shape)
    model.train_model(x_train, y_train)

    # generate predictions for test set
    predictions = model.generate_predictions(x_test)
    predictions = inverse_normal(predictions)

    # plot predictions
    plot_predictions(df, features='Close', predictions=predictions, company_name='GOOG')

    # predict next day
    next_day = predict_next_day(df, model)
    print('prediction of next day: ' + next_day)
