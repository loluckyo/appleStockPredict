# lstm model class using previous 60 days to predict next day
from keras.models import Sequential
from keras.layers import Dense, LSTM


class Model:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, x_train, y_train, batch_size=1, epochs=1):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    def generate_predictions(self, x_set):
        predictions = self.model.predict(x_set)
        return predictions
