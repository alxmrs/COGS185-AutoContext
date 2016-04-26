from random_forest.random_forest import *
from sklearn import svm
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


def concat_ocr_data(data, window_size):
    f_start = 6     # start of features
    pos_idx = 4     # word position index
    label_idx = 2   # label index

    data_new = data.copy()
    data_new = data_new[data_new[:, pos_idx] <= window_size]

    X = data_new[0::window_size, f_start:]
    Y = data_new[0::window_size, label_idx]
    for i in range(1, window_size):
        X = np.hstack((X, data_new[i::window_size, f_start:]))
        Y = np.vstack((Y, data_new[i::window_size, label_idx]))

    return X, Y.T


class AutoContext(object):
    '''
    C1: p(1)(y | x)               --> p(1)(y | ..)

    C2: p(2)(y | x, p(1)(y | ..)) --> p(2)(y | ..)

    C3: p(3)(y | x, p(2)(y | ..)) --> p(3)(y | ..)

    ...
    -- The auto context algorithm using a multiclass random forest instead of a structure random forest --

    Strategy 1:

    Train one random forest on all of the input data, performing traditional multiclass classification.
    Then, when classifying each label of the structured input (which will be concatenated x values), predict one label
    at a time and output its confidence value. When predicting the next label, sum the previous labels confidence value:
    Augment the confidence of each classification with the confidence of its neighbors.

    Strategy 2:

    Multiclass SVM approach using sci-kit learn's LinearSVC classifier (using crammer_singer implementation).

    '''
    def __init__(self, unstructured_data, n_classes, n_iter, w_size, dataset='OCR'):
        # _, self.n_label = structured_label.shape
        # self.Ns, self.ds = structured_train.shape
        self.Nu, self.du = unstructured_data.shape
        # self.X = structured_train
        # self.Y = structured_label
        self.unstructured_data = unstructured_data
        # self.train = unstructured_data
        self.n_classes = n_classes
        self.num_iterations = n_iter
        self.window_size = w_size
        self.forest = None
        self.models = []

        if dataset == 'OCR':
            self.train, self.test, self.N, self.dtr, self.Ntr, self.Ntst = self.prep_ocr_data()

    def prep_ocr_data(self, fold=[0, 1], test_fold=2, fold_idx=5, f_start=6, next_id_idx=2, label_idx=1, one_hot=False):
        if one_hot:
            y_space = self.n_classes
        else:
            y_space = 1

        if type(fold) is int:
            fold = [fold]

        if type(test_fold) is int:
            test_fold = [test_fold]

        selected_unstructured_data = self.unstructured_data

        Nt, du = selected_unstructured_data.shape
        dt = du - f_start
        data_tmp = np.zeros((1, dt + y_space))
        train = []
        test  = []

        Ntr = 0
        Ntst = 0

        for i in range(Nt):
            if one_hot:
                y = np.ones((1, self.n_classes))*-1             # one-hot encoding of label
                y[selected_unstructured_data[i, label_idx]] = 1
            else:
                y = selected_unstructured_data[i, label_idx]    # scalar encoding of label

            data_tmp = np.vstack((data_tmp, np.hstack((selected_unstructured_data[i, f_start:], y))))
            data_len = data_tmp.shape[0] - 1
            if selected_unstructured_data[i, next_id_idx] == -1:
                if selected_unstructured_data[i, fold_idx] in fold:
                    train.append(data_tmp[1:, :])
                    Ntr += data_len
                elif selected_unstructured_data[i, fold_idx] in test_fold:
                    test.append(data_tmp[1:, :])
                    Ntst += data_len
                data_tmp = np.zeros((1, dt + y_space))

        return train, test, Nt, dt, Ntr, Ntst

    def train_forest(self, n_trees=10, subsample=.10, linspace=20, n_workers=None):
        print('Training forest...')
        self.forest = RandomForest(self.train, n_trees, subsample=subsample, linspace=linspace,n_workers=n_workers)
        print('Forest trained.')

    def run(self, strat=2):
        if strat == 1:
            self.strategy1()
        else:
            self.strategy2()

    def strategy1(self):
        # Train one random forest on all input data via multiclassification
        if self.forest is None:
            self.train_forest()

        # Predict one label at a time in the structured input, augmenting the confidence of each class with the
        # confidence of its neighbors
        predictions = []
        for i in range(self.Ns):
            tmp_pred = []
            for j in range(self.window_size):
                x = self.X[i, j*self.du:(j + 1) * self.du]
                y = self.Y[i, j]
                datapoint = np.hstack((x, y))
                prediction = self.forest.predict(datapoint)
                tmp_pred.append(prediction)
            predictions.append(tmp_pred)

        return predictions

    def strategy2(self):
        '''
        SVM-based auto-context
        :return:
        '''
        # prep data
        if self.train is None:
            self.prep_ocr_data()

        confidence = np.zeros((self.Ntr, self.n_classes))
        accurracy1 = []
        accurracy2 = []

        for i in range(self.num_iterations):
            print('Iteration number ' + str(i+1) + ' out of ' + str(self.num_iterations))

            W = np.zeros((self.Ntr, self.dtr + self.n_classes * self.window_size * 2))    # Weight matrix: X + confidence
            Y = np.zeros(self.Ntr)                                                      # Cached predictions

            # print(W.shape)

            curr_line = 0

            print('Prepping data')
            for j in range(len(self.train)):
                word = self.train[j]        # get current word (X, which consists of x_1, x_2, ... x_m
                word_len = word.shape[0]    # find num letters in X (i.e. m)

                W[curr_line:curr_line+word_len, :self.dtr] = word[:, :self.dtr]
                W[curr_line:curr_line+word_len, self.dtr:] = self.extend_context(
                        confidence[curr_line:curr_line+word_len, :]
                )

                Y[curr_line:curr_line+word_len] = self.train[j][:, -1]
                curr_line += word_len

            print('Building model')
            svm_class = svm.LinearSVC(multi_class='crammer_singer', random_state=0)  # multiclass, consistent seed
            svm_class.fit(W, Y)

            self.models.append((svm_class, W))

            print('Performing prediction')
            # Perform prediction
            if i < self.num_iterations:
                acc1, acc2, confidence = self.svm_inference(self.train, confidence, svm_class)
                accurracy1.append(acc1)
                accurracy2.append(acc2)

        return accurracy1, accurracy2, confidence

    def svm_inference(self, data, confidence, svm, norm=True):
        print('\tPerforming SVM inference')
        Nt = len(data)
        print(Nt)
        acc1 = 0
        acc2 = 0
        total1 = 0
        total2 = 0
        conf_new = np.zeros(confidence.shape)

        cur_line = 0
        for i in range(Nt):

            word = data[i]
            word_len = word.shape[0]
            # print(word.shape)
            Y = word[:, -1]

            # TODO: implemented iterative context inference
            W_prime = np.zeros((word_len, self.dtr + self.n_classes * self.window_size * 2))
            # W_prime : [X | extended context]
            W_prime[:, :self.dtr] = word[:, :self.dtr]
            W_prime[:, self.dtr:] = self.extend_context(confidence[cur_line:(cur_line + word_len), :])

            # y_hat = svm.predict(W_prime)          # Predictions
            conf = svm.decision_function(W_prime)   # Confidence measures of predictions

            if norm:
                conf = (1 + np.exp(-1*conf))**-1    # Sigmoid function --> Normalization

            conf_new[cur_line : cur_line+word_len, :] = conf
            cur_line += word_len

            # Calculate error rates
            total1 += word_len
            total2 += 1
            subtask_acc = svm.score(W_prime, Y)
            acc2 += subtask_acc
            acc1 += subtask_acc * word_len
            print('\t\tShort-term accuracy: ' + str(subtask_acc))

        return acc1/total1, acc2/total2, conf_new

    def predict(self, test_data=None):
        if test_data is None:
            test_data = self.test
        n_iter = len(self.models)

        confidence = np.zeros((self.Ntst, self.n_classes))

        accuracy1 = []
        accuracy2 = []

        for i in range(n_iter):
            curr_model, _ = self.models[i]
            acc1, acc2, confidence = self.svm_inference(test_data, confidence, curr_model)
            accuracy1.append(acc1)
            accuracy2.append(acc2)

        return accuracy1, accuracy2, confidence

    def extend_context(self, conf, window_size=None, n_classes=None):
        if window_size is None:
            window_size = self.window_size
        if n_classes is None:
            n_classes = self.n_classes

        word_len = conf.shape[0]
        W = np.zeros((word_len, 2*window_size*n_classes))
        for i in range(word_len):
            for w in range(-window_size, window_size):
                if 0 <= i + w < word_len:
                    if w < 0:
                        W[i, (window_size + w)*n_classes : (window_size+w)*n_classes + n_classes] =\
                            conf[i + w, :n_classes]
                    elif w > 0:
                        W[i, (window_size + w - 1)*n_classes : (window_size + w - 1)*n_classes + n_classes] =\
                            conf[i + w, :n_classes]

        return W


def main():
    '''
    Main method
    '''
    # genfromtxt does the same thing as my implementation, but slightly slower.
    # letter_data = np.genfromtxt('../data/letter.data', dtype=None)

    # Acquire data
    letter_data_obj = Data('../data/letter.data', filetype='CSV')
    letter_data = letter_data_obj.read_data()
    print('data loaded')

    # Convert characters to 1 byte integers
    letter_data[:,1] = [ord(c) - 97 for c in letter_data[:,1]]  # char --> intx
    letter_data = letter_data.astype('i1')
    print('data converted to ints')

    # strategy 1: concat letters into words of a particular window size
    # X, Y = concat_ocr_data(letter_data, 3)
    #
    # train = letter_data[:, 6:]
    # print(train.shape, letter_data[:,2].shape)
    # train = np.hstack((train, letter_data[:,2].reshape((letter_data.shape[0], 1))))  # [X | y]
    #
    # print(train.shape)
    # print(X.shape, Y.shape)
    #
    # ac = AutoContext(X[:25, :], Y[:25, :], train[:50, :])
    # predictions = ac.strategy1()
    #
    # print(predictions[:20])

    # strategy 2
    print('Creating AutoContext object, prepping OCR dataset')
    ac = AutoContext(letter_data,26,4,3)
    # print(ac.train[1].shape)  # sanity check
    # print(ac.Ntr, ac.dtr)

    print('Training Strategy 2: SVM-based Auto Context')
    tr_accuracy1, tr_accuracy2, conf = ac.strategy2()
    print('Training accuracy (by word and letter):')
    print(tr_accuracy1)
    print(tr_accuracy2)
    print('Testing Strategy 2')
    ts_accuracy1, ts_accuracy2, conf = ac.predict()
    print('Testing accuracy (by word and by letter):')
    print(ts_accuracy1)
    print(ts_accuracy2)



if __name__ == '__main__':
    main()
    sys.exit()
