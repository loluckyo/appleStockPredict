import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

scaler = MinMaxScaler(feature_range=(0, 1))


def split_data(df, features):
    data = df.filter([features])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    scaled_data = normalize(dataset)
    train_data = scaled_data[0:training_data_len, :]

    x_train = []
    y_train = []

    # use prev 60 days to predict next day
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # reshape x_train for lstm model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    # reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test


def normalize(dataset):
    return scaler.fit_transform(dataset)


def inverse_normal(dataset):
    return scaler.inverse_transform(dataset)


def plot_predictions(df, features, predictions, company_name):
    data = df.filter([features])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    # plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title(company_name + ' Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USC ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

def predict_next_day(df, model):
    new_df = df.filter(['Close'])

    last_60_days = new_df[-60:].values
    last_60_days_scaled = normalize(last_60_days)

    x_test = []
    x_test.append(last_60_days_scaled)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pred_price = model.predict(x_test)
    pred_price = inverse_normal(pred_price)

    return pred_price