#Code from Amer Kumar and adapted by me for volitlity factor
import math 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib.dates import YearLocator, DateFormatter
from arch import arch_model
from GARCH import GARCH

#implimenting the file path
csv_file = 'Downloads/stock.csv'
# Load the data
stock = pd.read_csv('csv_file')
closing_prices = stock['Close']


values = closing_prices.values

training_data_len = math.ceil(len(values) * .8)
print(training_data_len)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))

train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])

x_test = np.array (x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = keras.Sequential()
model.add(layers.LTSM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LTSM(100, return_sequences=False))
model.add(layers.Dense(25))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=3)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)


df = stock [:training_data_len]
up = 0
up_list = []
down = 0
down_list = []
same = 0
same_list = []
prev_price = df['Close'].values[0]

threshold = 0.01

for index in range(1, df['Close'].count()):
    curent_price = df.loc[index, 'Close']
    price_movement = (curent_price - prev_price)/ prev_price
    state = 0
    if price_movement > threshold:
        state = 1
        up_list.append(curent_price)
        up += 1
    elif price_movement < -threshold:
        state = -1
        down_list.append(curent_price)
        down += 1
    else:
        same += 1
        same_list.append(curent_price)
        state = 0
print(up, down, same)

transition_matrix = []

up_up = 0
up_down = 0
up_same = 0
for index in range(0, len(up_list) - 1):
    current_price = up_list[index]
    next_price = up_list[index + 1]
    price_movement = (next_price - current_price) / current_price
    if price_movement > threshold:
        up_up += 1
    elif price_movement < -threshold:
        up_down += 1
    else:
        up_same += 1
up_len = len(up_list) - 1 
transition_matrix.append(up_up/up_len)
transition_matrix.append(up_down/up_len)
transition_matrix.append(up_same/up_len)

down_up = 0
down_down = 0
down_same = 0
for index in range(0, len(down_list) - 1):
    current_price = down_list[index]
    next_price = down_list[index + 1]
    price_movement = (next_price - current_price) / current_price
    if price_movement > threshold:
        down_up += 1
    elif price_movement < -threshold:
        down_down += 1
    else:
        down_same += 1
down_len = len(down_list) - 1
transition_matrix.append(down_up/down_len)
transition_matrix.append(down_down/down_len)
transition_matrix.append(down_same/down_len)

same_up = 0
same_down = 0
same_same = 0
for index in range(0, len(same_list) - 1):
    current_price = same_list[index]
    next_price = same_list[index + 1]
    price_movement = (next_price - current_price) / current_price
    if price_movement > threshold:
        same_up += 1
    elif price_movement < -threshold:
        same_down += 1
    else:
        same_same += 1

same_len = len(same_list) - 1
transition_matrix.append(same_up/same_len)
transition_matrix.append(same_down/same_len)
transition_matrix.append(same_same/same_len)

print(transition_matrix)
def matrix_power(A, n):
    A = np.reshape(A, (3,3))
    if A.shape != (3,3):
        print("Invalid matrix")
        return None
    if not isinstance(n, int):
        print("Invalid power")
        return None
    if n == 0:
        return np.eye(3)
    if n > 0:
        return np.linalg.matrix_power(A, n)
    if n < 0:
        return np.linalg.matrix_power(np.linalg.inv(A), -n)
print(matrix_power(transition_matrix, 10) #Steady State Matrix

