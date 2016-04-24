from random_forest.random_forest import *
# import random_forest
# import random_forest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import sys, os


def to_record_array(np_arr):
    '''
    take in a numpy array, return a labeled numpy array of tuples with the proper data type
    :param np_arr: numpy array
    :return: labeled structured array of tuples with proper data type
    '''

    # Get column labels
    data_cols = []
    with open('../data/letter.names.txt') as f:
        for line in f:
            data_cols.append(str(line.strip('\n')))

    # create list of column data types
    data_col_types = ['i1', 'S1']
    for i in range(2, len(data_cols)):
        data_col_types.append('i1')

    # zip them together in a list
    dtypes = [(d, t) for d, t in zip(data_cols, data_col_types)]

    # create data type object
    data_dtype = np.dtype(dtypes)

    # return new structured array (will be a numpy array of tuples)
    return np.core.records.array(list(tuple(np_arr.transpose())), dtype=data_dtype)


def format_array(arr):
    arr[: 1] = [ord(c) - 97 for c in arr[:,1]]
    return arr.astype('i1')


def format_data_for_auto_context(data, window_size):
    N, d = data.shape

    f_start = 6  # start of features
    x_data = []
    y_data = []
    tmp_x = np.empty((d - f_start), dtype='i1')
    tmp_y = []
    word_id = 1
    wind_count = 0

    overwritten_empty_flag = False

    for i in range(N):
        print(data[i, 3] == word_id)
        if data[i, 3] == word_id and wind_count <= window_size:
            wind_count += 1
            if tmp_x.shape[0] == 1 and not overwritten_empty_flag:
                overwritten_empty_flag = True
                tmp_x[0] = data[i, f_start:]
            else:
                print(tmp_x.shape)
                print(data[i, f_start:].shape)
                np.concatenate((tmp_x, data[i, f_start:]), 0)
            tmp_y.append(data[i, 1])
        elif data[i, 3] == word_id and wind_count > window_size:
            word_id += 1
        elif wind_count <= window_size and data[i, 3] is not word_id:
            wind_count += 1
            if tmp_x.shape[0] == 1 and not overwritten_empty_flag:
                tmp_x[0] = np.zeros((d - f_start), dtype='i1')
                overwritten_empty_flag = True
            else:
                np.concatenate((tmp_x, np.zeros((d - f_start), dtype='i1')), 0)
            tmp_y.append(-1)
        elif tmp_x.any() and tmp_y and wind_count > window_size and data[i, 3] is not word_id:
            x_data.append(tmp_x)
            y_data.append(tmp_y)
            tmp_x = np.empty((d - f_start), dtype='i1')
            tmp_y = []

        if data[i, 2] == -1:
            wind_count = 0
            overwritten_empty_flag = False

    return np.array(x_data), np.array(y_data)


def main():
    '''
    Main method
    '''
    # genfromtxt does the same thing as my implementation, but slightly slower.
    # letter_data = np.genfromtxt('../data/letter.data', dtype=None)

    # Acquire data
    letter_data_obj = Data('../data/letter.data', filetype='CSV')
    letter_data = letter_data_obj.read_data()

    # Convert characters to 1 byte integers
    letter_data[:,1] = [ord(c) - 97 for c in letter_data[:,1]]  # char --> intx
    letter_data = letter_data.astype('i1')

    formatted_x, formatted_y = format_data_for_auto_context(letter_data, 2)


    print(formatted_x.shape)
    print(formatted_y.shape)
    print(formatted_x[:2])
    print(formatted_y[:2])



if __name__ == '__main__':
    main()
    sys.exit()
