from LSTM.data_processor import DataLoader
from LSTM.model import fit_lstm, scale, invert_scale, forecast_lstm, r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error


project_path = "./"
save_dir = project_path + "saved_models/2-12-saved-models-cantho-03/"
data_path = project_path + "C1-6/C1-6_CanTho/"
filenames = ["C1", "C2", "C3", "C4", "C5", "C6"]
neurons = [5, 8, 10, 12, 15]
optimizers = ['rmsprop', 'adagrad', 'adadelta', 'adam']

i = 0
res_list = []
for filename in filenames:
    for optimizer in optimizers:
        training_data = DataLoader(
            data_path + filename + "/Training_Input.txt", data_path + filename + "/Training_Target.txt")
        test_data = DataLoader(data_path + filename + "/Testing_Input.txt",
                               data_path + filename + "/Testing_Target.txt")

        x_train, y_train = training_data.get_data()
        x_test, y_test = test_data.get_data()

        train = np.concatenate((x_train, y_train), axis=1)
        test = np.concatenate((x_test, y_test), axis=1)
        scaler, train_scaled, test_scaled = scale(train, test)
#         yhat = invert_scale(scaler, X, yhat)
        neuron_list = []
        for neuron in neurons:
            i += 1
            plt.figure(i)
            print("[model] {}, {}, {}".format(neuron, filename, optimizer))
            lstm_model = load_model(
                save_dir + str(neuron) + '-' + filename + '-' + optimizer + '.h5')
            train_reshaped = train_scaled[:, 0:-1].reshape(
                len(train_scaled), 1, train_scaled.shape[1]-1)
            predict = lstm_model.predict(train_reshaped, batch_size=1)
            #
            predictions = list()
#             predictions = lstm_model.predict(test_scaled[i, 0:-1], batch_size=1)
            for i in range(len(test_scaled)):
                # make one-step forecast
                X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
                yhat = forecast_lstm(lstm_model, 1, X)
                # invert scaling
                yhat = invert_scale(scaler, X, yhat)
                # invert differencing
#                 yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
                # store forecast
                yhat = math.fabs(yhat)
                predictions.append(yhat)
#                 expected = raw_values[len(train) + i + 1]
                expected = y
            predictions = np.array(predictions)
            print("R: {}.RMSE: {}. MAE: {}. ".format(round(r(test[:, -1], predictions), 4),
                                                     round(np.sqrt(mean_squared_error(
                                                         test[:, -1], predictions)), 4),
                                                     round(mean_absolute_error(test[:, -1], predictions), 4)))
            neuron_list.append([round(r(test[:, -1], predictions), 4),
                                round(np.sqrt(mean_squared_error(
                                    test[:, -1], predictions)), 4),
                                round(mean_absolute_error(test[:, -1], predictions), 4)])
            plt.plot(test[:, -1])
            plt.plot(predictions)
            plt.title("Can Tho {} {}".format(filename, optimizer))
            plt.show()
        res_list.append(list(map(list, zip(*neuron_list))))

res_list = np.array(res_list).reshape(72, 5)
df = pd.DataFrame.from_records(res_list)
df.to_csv(project_path+"res00.csv")
