import json
import matplotlib.pyplot as plt
from pprint import pprint
from LSTM.data_processor import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# with open('./saved_models_cantho/C1-adadelta-history.json') as f:
#     data = json.load(f)
filenames = ["C1", "C2", "C3", "C4", "C5", "C6"]
optimizers = ['rmsprop', 'adagrad', 'adadelta', 'adam']
# i = 0
# for filename in filenames:
#     i+=1
#     plt.figure(i)
#     for optimizer in optimizers:
#         with open('./saved_models_cantho/' + filename + '-' + optimizer + '-history.json') as f:
#             data = json.load(f)
#         plt.plot(data)
#     plt.title("Can Tho  + filename)
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['rmsprop', 'adagrad', 'adadelta', 'adam'], loc='upper right')

project_path="./"
save_dir = project_path + "saved_models/"
data_path = project_path + "C1-6/C1-6_CanTho/"
lstm_model  = load_model('./saved_models_cantho/C1-rmsprop.h5')
i = 0
for filename in filenames:
    training_data = DataLoader(data_path + filename + "/Training_Input.txt", data_path + filename + "/Training_Target.txt")
    test_data = DataLoader(data_path + filename + "/Testing_Input.txt", data_path + filename + "/Testing_Target.txt")

    x_train, y_train = training_data.get_data()
    x_test, y_test = test_data.get_data()

    train = np.concatenate((x_train, y_train), axis=1)
    test = np.concatenate((x_test, y_test), axis=1)
    scaler, train_scaled, test_scaled = scale(train, test)

    for optimizer in optimizers:
        i+=1
        plt.figure(i)
        print("[model] {}, {}".format(filename, optimizer))
        lstm_model  = load_model('./saved_models_cantho/' + filename + '-' + optimizer + '.h5')
        train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), 1, train_scaled.shape[1]-1)
        predict = lstm_model.predict(train_reshaped, batch_size=1)
        plt.plot(train_scaled[:, -1])
        plt.plot(predict)
        plt.title("Can Tho " + filename)


plt.show()