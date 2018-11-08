__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from LSTM.data_processor import DataLoader
from LSTM.model import Model
from keras.utils import plot_model
import numpy as np

def main():
    batch_size=8
    epochs = 100
    root = './C1-6/C1-6_CanTho/'
    x_dir = '/Training_Input.txt'
    y_dir = '/Training_Target.txt'
    save_dir = "./saved_models/"
    # filenames = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    filenames = ['C1']
    # optimizers = ['rmsprop', 'adagrad', 'adadelta', 'adam']
    optimizers = ['rmsprop']
    i = 1
    for filename in filenames:
        data = DataLoader(root + filename + x_dir, root+filename+y_dir)
        plt.figure(i)
        i+=1
        for optimizer in optimizers:
            model = Model()
            x, y = data.get_train_data()
            # y = y[:,:,0]
            print('[Model] Build model. Filename: ', filename, '. Optimizer: ', optimizer)
            model.build_model(x.shape[2],optimizer)
            print(x.shape, y.shape)
            steps_per_epoch = math.ceil((data.len - 100) /  batch_size)
            history, yhat = model.train_generator(
                x, 
                y,
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                save_dir=save_dir,
                data=filename,
                optimizer=optimizer
            )
            with open(save_dir + '%s-%s.json' % (filename, optimizer), 'w') as f:
                json.dump(history.history, f)

            # plt.plot(history.history['loss'])
            print(yhat)
            print(yhat.shape)
            plt.plot(y.ravel(), color='blue')
            plt.plot(yhat.ravel(), color='red')
            del model
        plt.title(filename)
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['rmsprop', 'adagrad', 'adadelta', 'adam'], loc='upper right')
    
    plt.show()
if __name__ == '__main__':
    main()