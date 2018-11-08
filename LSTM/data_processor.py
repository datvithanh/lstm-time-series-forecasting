import math
import numpy as np
import pandas as pd

class DataLoader():
    '''Class for loading and transforming data for LSTM model'''
    def __init__(self, filename_x, filename_y):
        dataframe_x = pd.read_csv(filename_x, sep='\s+',header=None)
        dataframe_y = pd.read_csv(filename_y, sep='\s+',header=None)
        self.x = dataframe_x.values[:]
        self.y = dataframe_y.values[:]
        self.len = len(self.x)
        self.seq_len = 30

    def get_train_data(self):
        train_x = []
        train_y = []
        train_dim = math.floor(self.len/self.seq_len)
        for i in range(train_dim):
            start = i*self.seq_len
            end = (i+1)*self.seq_len
            train_x.append(self.x[start:end])
            train_y.append(self.y[start:end])
            # print(i*self.seq_len, (i+1)*self.seq_len)
        return np.array(train_x), np.array(train_y)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.x[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
    
    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)
        