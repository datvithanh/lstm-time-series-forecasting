__author__ = "Jakob Aungiers"
__copyright__ = "Jakob Aungiers 2018"
__version__ = "2.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
from keras.utils import plot_model



def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    print(data)
    '''
	# in-memory training
	model.train(
		x,
		y,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)
	'''
    # out-of memory generative training
    model = Model()
    model.build_model(configs)
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    print(x.shape)
    print(data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                normalise=configs['data']['normalise']
            ))
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    # plot_model(model, to_file='model.png')
    for optimizer in ['rmsprop', 'adagrad', 'adadelta', 'adam']:
        configs['model']['optimizer'] = optimizer
        model = Model()
        model.build_model(configs)
        x, y = data.get_train_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
        history = model.train_generator(
            data_gen=data.generate_train_batch(
                seq_len=configs['data']['sequence_length'],
                batch_size=configs['training']['batch_size'],
                normalise=configs['data']['normalise']
            ),
            epochs=configs['training']['epochs'],
            batch_size=configs['training']['batch_size'],
            steps_per_epoch=steps_per_epoch,
            save_dir=configs['model']['save_dir']
        )

        x_test, y_test = data.get_test_data(
            seq_len=configs['data']['sequence_length'],
            normalise=configs['data']['normalise']
        )
        plt.plot(history.history['loss'])
        del model
    
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['rmsprop', 'adagrad', 'adadelta', 'adam'], loc='upper left')
    # plt.show()

if __name__ == '__main__':
    main()