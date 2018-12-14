import numpy as np
import json
from LSTM.data_processor import DataLoader
from LSTM.model import fit_lstm, scale

project_path = "./"
save_dir = project_path + "chau_doc_saved_models/"
data_path = project_path + "C1-6/C1-6_CanTho/"
filenames = ["C1", "C2", "C3" "C4", "C5", "C6"]

neurons = [5, 8, 10, 12, 15]
optimizers = ['rmsprop', 'adagrad', 'adadelta', 'adam']

for neuron in neurons:
    for filename in filenames:
        training_data = DataLoader(
            data_path + filename + "/Training_Input.txt", data_path + filename + "/Training_Target.txt")
        test_data = DataLoader(data_path + filename + "/Testing_Input.txt",
                               data_path + filename + "/Testing_Target.txt")

        x_train, y_train = training_data.get_data()
        x_test, y_test = test_data.get_data()

        train = np.concatenate((x_train, y_train), axis=1)
        test = np.concatenate((x_test, y_test), axis=1)
        for optimizer in optimizers:
            print("[Model] data: {}, optimizer: {}, neuron: {}".format(
                filename, optimizer, str(neuron)))
            scaler, train_scaled, test_scaled = scale(train, test)
            lstm_model, history = fit_lstm(
                train_scaled, 1, 50, neuron, optimizer, 0.1)

            lstm_model.save(save_dir + '%s-%s-%s.h5' %
                            (str(neuron), filename, optimizer))
            with open(save_dir + '%s-%s-%s-history.json' % (str(neuron), filename, optimizer), 'w') as f:
                json.dump(history, f)
