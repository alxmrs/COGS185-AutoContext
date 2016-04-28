import numpy as np
import scipy.io
from queue import *
from multiprocessing import Pool


class Node(object):
    '''
    Node class: basic unit of a decision tree
    '''
    def __init__(self, left=None, right=None, data=None, rule=(None, None), label=None):
        self.left = left
        self.right = right
        self.data = data
        self.rule = rule
        self.label = label

    def copy(self, node):
        self.left = node.left
        self.right = node.right
        self.data = node.data
        self.rule = node.rule


class ID3(object):
    '''
    ID3 Decision Tree
    '''
    def __init__(self, train, linspace=100, n_features=-1):
        '''
        :param train: Training data set
        :param linspace: number of evenly spaced points over the feature space to search through. Default=100
        :param n_features: number of features to search through when making splitting rules. When entering a smaller
        number than the dimensions fo the training set, the algorithm will randomly selected this many features to
        search through when deciding the best feature to use for the splitting rule. -1 means search through all the
        features. Default: -1
        '''
        self.train = train
        self.linspace = linspace
        self.num_features = n_features
        self.root = self.generate_tree(self.train)

    def information_gain(self, f, t, data):
        '''
        Calculates the decrease in conditional entropy given condition Z (if X[f] <= t): Information gain is
        IG(Z) = H(X) - H(X|Z) where H(X) is the entropy distributed at a node and H(X|Z) is the entropy given the node
        splitting rule.
        :param f: feature
        :param t: threshold
        :param data: data
        :return: information gain
        '''

        N, d = np.shape(data)
        label_freq = {}
        left = []
        left_freq = {}
        right = []
        right_freq = {}

        for i in range(N):

            # Initialize dictionaries
            if data[i, -1] not in label_freq:
                label_freq[data[i, -1]] = 0.0

            if data[i, -1] not in right_freq:
                right_freq[data[i, -1]] = 0.0

            if data[i, -1] not in left_freq:
                left_freq[data[i, -1]] = 0.0

            label_freq[data[i, -1]] += 1.0

            if data[i, f] <= t:
                left.append(data[i, -1])
                left_freq[data[i, -1]] += 1.0
            else:
                right.append(data[i,-1])
                right_freq[data[i, -1]] += 1.0

        # Calculate entropy, H(X)
        H_x = 0
        for k, v in label_freq.items():
            f = v/N
            if f != 0:
                H_x -= f*np.log(f)

        # Calculate H(X | Z = 0)
        H_x_z0 = 0
        if len(left) != 0:
            for k, v in left_freq.items():
                f = v/len(left)
                if f != 0:
                    H_x_z0 -= f*np.log(f)

        # Calculate H(X | Z = 1)
        H_x_z1 = 0
        if len(right) != 0:
            for k, v in right_freq.items():
                f = v/len(right)
                if f != 0:
                    H_x_z1 -= f*np.log(f)

        # Calculate H(X | Z)
        Pz0 = float(len(left))/N
        Pz1 = float(len(right))/N
        H_x_z = Pz0 * H_x_z0 + Pz1 * H_x_z1

        # Return IG: H(X) - H(X|Z)
        return H_x - H_x_z

    def generate_tree(self, data):
        '''
        Build a decision tree from the training data using best information_gain as the criteria for splitting. Does not
        include pruning.
        :param data: training data
        :return: root of newly constructed decision tree
        '''
        self.root = Node(data=data, rule=(None, None))

        to_explore = Queue(maxsize=np.shape(data)[0])

        to_explore.put(self.root)

        count = 0
        while not to_explore.empty():

            current_node = to_explore.get()

            N, d = np.shape(current_node.data)

            # Bag features or use all the features
            if self.num_features == -1:
                feature_space = range(d-1)
            else:
                feature_space = np.random.choice(d-1, size=self.num_features, replace=False)

            # test if data is impure:
            if not np.all(current_node.data == current_node.data[0, :], axis=0)[-1]:

                max_ent = -9999999999999999999999
                feature = 0
                thr_i = 0   # Initial threshold
                thr_f = 0   # Final threshold

                for f in feature_space:
                    f_min = np.min(current_node.data[:, f])
                    f_max = np.max(current_node.data[:, f])
                    for t in np.linspace(f_min, f_max, self.linspace):

                        # Calculate entropy
                        new_ent = self.information_gain(f, t, current_node.data)

                        # Find first acceptable threshold and feature
                        if new_ent > max_ent:
                            max_ent = new_ent
                            feature = f
                            thr_i = t
                        # Find last acceptable threshold
                        elif new_ent == max_ent:
                            # feature = f
                            thr_f = t

                # Calculate midpoint of range of acceptable thresholds, if applicable
                if thr_f >= thr_i:
                    thresh = (thr_f+thr_i)/2
                else:
                    thresh = thr_i

                data_left = np.zeros((1, d))
                data_right = np.zeros((1, d))
                left = 0
                right = 0
                for i in range(np.shape(current_node.data)[0]):
                    if current_node.data[i, feature] <= thresh:
                        data_left = np.vstack((data_left, current_node.data[i, :]))
                        left += 1
                    else:
                        data_right = np.vstack((data_right, current_node.data[i, :]))
                        right += 1

#                 print(feature, thresh, N, max_ent)

                if np.shape(data_left)[0] == 1:
                    left_child = None
                else:
                    data_left = data_left[1:, :]
                    left_child = Node(data=data_left)

                if np.shape(data_right)[0] == 1:
                    right_child = None
                else:
                    data_right = data_right[1:, :]
                    right_child = Node(data=data_right)

                current_node.left = left_child
                current_node.right = right_child
                current_node.rule = (feature, thresh)

                if left_child is not None:
                    to_explore.put(left_child)

                if right_child is not None:
                    to_explore.put(right_child)

                count += 1
            else:
                current_node.label = current_node.data[0, -1]

        return self.root

    def parse(self, data_point, root):
        '''
        Recursively parse tree to find the class of a data point.
        :param data_point: non-labeled data point
        :param root: root of the subtree
        :return: label of data point
        '''
        if root.label is not None:
            return root.label
        else:
            (f, t) = root.rule
            if data_point[f] <= t:
                if root.left:
                    return self.parse(data_point, root.left)
            else:
                if root.right:
                    return self.parse(data_point, root.right)

    def predict(self, data_point):
        '''
        Find the label or class of a data point.
        :param data_point:
        :return:
        '''
        prediction = self.parse(data_point, self.root)
        # Return "don't know" class if no class/label found, otherwise return prediction
        if prediction is None:
            return -1
        else:
            return prediction

    def error_rate(self, test):
        '''
        Calculate error rate of decision tree on testing data set
        :param test: test data set --> [X | y]
        :return: proportion of wrong classifications over size of testing data set
        '''
        N, d = np.shape(test)

        err = 0
        for i in range(N):
            if self.predict(test[i, :-1]) != test[i, -1]:
                err += 1.0
        return err/N


def read_data(data_file_name, n_features=None, n_datapoints=-1):
    """
    Slightly Modified by Alex Rosengarten
    Source: https://github.com/cjlin1/libsvm/blob/master/python/svmutil.py
    svm_read_problem(data_file_name) -> [y, x]
    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = []
    prob_x = []
    i = 0
    for line in open(data_file_name):
        if i is n_datapoints:
            break
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        if n_features is None:
            xi = [0 for _ in range(len(features.split()))]
        else:
            xi = [0 for _ in range(n_features)]
        for e in features.split():
            ind, val = e.split(":")
            xi[int(ind)-1] = float(val)
        i += 1
        prob_y += [float(label)]
        prob_x += [xi]

    return prob_y, prob_x


class Data(object):
    '''
    A parallelized way of importing LIBSVM-formatted data from a file. To use, create a Data object with the proper
    parameters and call read_data().
    '''
    def __init__(self, data_file_name, n_features=None, n_datapoints=-1, n_workers=None, filetype='SVM', delim=None):
        '''
        :param data_file_name: data file
        :param n_features: number of features in the data set
        :param n_datapoints: [non-functional feature] number of data points to import before stopping
        :param n_workers: number of threads or processes working to import the data
        :param filetype:
        :param delim:
        :return:
        '''
        self.file = data_file_name
        self.n_features = n_features
        self.n_datapoints = n_datapoints
        self.n_threads = n_workers
        self.filetype = filetype
        self.delimiter = delim

    def process_svm_line(self, line):
        '''
        Process one line of data from a SVM formated data file.
        See following for example datasets:
        http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
        :param line: the line of the file to process
        :return: An array of data with the label at the right most column --> [X | y]
        '''
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        if self.n_features is None:
            xi = [0.0 for _ in range(len(features.split()))]
        else:
            xi = [0.0 for _ in range(self.n_features)]
        for e in features.split():
            ind, val = e.split(":")
            xi[int(ind)-1] = float(val)

        return xi + [float(label)]

    def process_csv_line(self, line):
        return [self.determine_data_type(v) for v in line.strip().split(self.delimiter) if v is not None]

    def determine_data_type(self, elem):
        if self.is_float(elem) and '.' in elem:
            return float(elem)
        elif self.is_int(elem):
            return int(elem)
        else:
            return str(elem)

    def is_float(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_int(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def is_complex(self, s):
        try:
            complex(s)
            return True
        except ValueError:
            return False


    def read_data(self):
        '''
        Reads the data into program in parallel via a thread pool. Uses the higher-order function Map to call
        process_line on the input file.
        :return: dataset, in numpy array format.
        '''
        if self.n_threads is None:
            pool = Pool()
        else:
            pool = Pool(self.n_threads)

        with open(self.file) as f:
            if self.filetype == 'SVM':
                results = pool.map(self.process_svm_line, f)
            else:
                results = pool.map(self.process_csv_line, f)
            pool.close()
            pool.join()

        return np.array(results)


class RandomForest(object):
    '''
    Random Forest, composed of ID3 decision trees.
    '''
    def __init__(self, train, trees=100, subsample=.10, linspace=100, n_features=None, n_workers=None):
        '''

        :param train: training set, required
        :param trees: Number of trees. Default: 100
        :param subsample: proportion of the training set each used to build each decision tree. Default: 10%
        :param linspace: number of evenly spaced points over the feature space to search through. Default=100
        :param n_features: number of features in dataset (training set)
        :param n_workers: number of workers in thread pool
        '''
        self.training_set = train
        self.N, self.D = self.training_set.shape
        self.num_trees = trees
        self.subsample_size = int(self.N*subsample)
        self.linspace = linspace
        if n_features is None:
            self.num_features = int(np.sqrt(self.D))
        else:
            self.num_features = n_features

        self.subsamples = self.bootstrap()
        self.forest = self.generate_forest(n_workers)

    def bootstrap(self):
        '''
        Generate random (with replacement) subsample of data set.
        :return:
        '''
        inds = [np.random.choice(self.N, self.subsample_size) for _ in range(self.num_trees)]
        return [np.array([self.training_set[i] for i in ind]) for ind in inds]

    def create_tree(self, sample):
        '''
        Create a decision tree
        :param sample: data set to build tree
        :return: Trained decision tree
        '''
        return ID3(sample, linspace=self.linspace, n_features=self.num_features)

    def generate_forest(self, n_workers=None):
        '''
        Create forest of decision trees. Works in parallel via the higher order function Map. Work is split up in a pool
        of workers that each call the create_tree function.
        :param n_workers: number of workers in the thread/process pool
        :return: list of decision trees in the forest.
        '''
        if n_workers is None:
            pool = Pool()
        else:
            pool = Pool(n_workers)

        forest = pool.map(self.create_tree, self.subsamples)
        pool.close()
        pool.join()
        return forest

    def predict(self, data_point):
        '''
        Map the data point to a class/label. Classifies the input based on a equally weighted voting from the ensemble
        of decision trees in the forest.
        :param data_point: test data point to be classified
        :return: (label with the most votes, confidence of the prediction)
        '''
        labels = {}
        if self.forest is None:
            return None

        total = 0.0
        for tree in self.forest:
            p = tree.svm_predict(data_point)

            if p not in labels:
                labels[p] = 0.0

            labels[p] += 1.0
            total += 1.0

        max_label = max(labels.keys(), key=(lambda key: labels[key]))  # return label with highest frequency in dictionary
        # print('RF predict, probability: ')
        # print((labels[max_label]), labels[max_label] / total)
        return max_label, labels[max_label] / total

    def error_rate(self, test):
        '''
        Calculates the error rate of the trained forest against a test data set
        :param test: a testing data set, [X | y]
        :return: number of errors over total number of examples
        '''
        N, d = np.shape(test)

        err = 0
        for i in range(N):
            if self.predict(test[i, :-1])[0] != test[i, -1]:
                err += 1.0
        return err/N

    def confusion_mx(self, test, n_classes):
        '''
        Cij/Nj where Cij is the number of test examples that have label
        j but are classified as label i by the classifier, and Nj is the
        number of test examples that have label j.
        :param test:
        :param n_classes:
        :return:
        '''
        N, d = np.shape(test)

        # cmx = np.zeros(n_classes+1, n_classes)
        Cij = np.zeros((n_classes+1, n_classes))
        Nj  = np.zeros((n_classes, 1))

        for i in range(N):
            prediction = self.predict(test[i, :-1])[0]
            actual     = test[i, -1]

            Cij[prediction+1, actual] += 1
            Nj[actual] += 1

        return Cij / Nj.T


def evenly_distribute_data(data):
    data_by_class = {}
    freq = {}

    for d in data:
        if d[-1] not in freq:
            data_by_class[d[-1]] = [d]
            freq[d[-1]] = 0.0
        else:
            data_by_class[d[-1]] = np.concatenate((data_by_class[d[-1]], [d]), 0)

        freq[d[-1]] += 1.0

    print(freq)
    smallest_class = min(freq.keys(), key=(lambda key: freq[key]))
    smallest_class_size = freq[smallest_class]
    smallest_class_data = data_by_class[smallest_class]

    new_dataset = smallest_class_data

    for k in data_by_class.keys():
        curr_class = data_by_class[k]

        inds = np.random.choice(curr_class.shape[0], smallest_class_size, replace=False)
        rand_sample = [curr_class[i, :] for i in inds]
        rand_sample = np.array(rand_sample)

        new_dataset = np.concatenate((new_dataset, rand_sample), 0)
    #
    # return new_dataset

    np.savetxt('./even_data', new_dataset, delimiter=',')


def ova_confusion_err(predictions, test, n_classes):
    '''
    Calucates confusion matrix and error rate for one versus all classification. Uses confidence measures to pick the
    best prediction out of the competing classifiers.
    :param predictions: a matrix of predictions. Each entry in the matrix is a tuple of predictions and confidence vals
    :param test: the testing data set
    :param n_classes: number of classes
    :return: (confusion matrix, error rate)
    '''
    Cij = np.zeros((n_classes+1, n_classes)) # Include "Don't Know" Class
    Nj  = np.zeros((n_classes, 1))

    N, d = test.shape
    err = 0.0

    for i in range(N):

        # Search through each classifiers prediction, return
        total_space = 0.0
        max_c = -1  # max confidence
        min_c = 1   # min confidence (if all negative)
        ppred = -1  # final prediction
        npred = -1
        all_neg = True
        for j, p_tup in enumerate(predictions[i, :]):
            pred, conf = p_tup
            total_space += conf

            # print(p_tup, end=' ')

            # find negative prediction with min confidence
            if conf < min_c and pred == -1:
                npred = pred
                min_c = conf

            # if two negative classes have equal confidence, return "don't know"
            elif conf == min_c and pred == -1:
                npred = -1

            # find positive prediction with max confidence
            elif conf > max_c and pred == 1:
                all_neg = False
                max_c = conf
                ppred = j

            # if there are two positive classes with equal confidence, return "don't know"
            elif conf == max_c and pred == 1:
                ppred = -1

        # print('fred: ' + str(max_c) + ' ' + str(fpred))

        # If there were all negative predictions, choose prediction with best confidence
        if all_neg:
            prediction = npred
        else:
            prediction = ppred

        actual = test[i, -1]

        if prediction != actual:
            err += 1.0

        Cij[prediction + 1, actual] += 1
        Nj[actual] += 1

    return Cij / Nj.T, err/N


def calc_freq(ys):
    '''
    Calculates frequency of every class, prints to standard out.
    :param ys: labels, or y values
    '''
    label = {}
    total = 0
    for y in ys:
        if y not in label:
            label[y] = 0.0
        label[y] += 1
        total += 1
    print(label)

    for k in label.keys():
        label[k] = label[k] / total

    print(label)


def suit_rank_to_card_number(poker_data):
    '''
    maps the (suit, rank) format of poker data into a single number representing 'card number'.
    :param poker_data: poker data set
    :return: new data set matrix, in numpy format
    '''
    N, d = poker_data.shape
    D_new = np.zeros((N, 6))
    D_new[:, -1] = poker_data[:, -1]
    for row in range(N):
        for col in range(5):
            suit, rank = poker_data[row, col*2:col*2+2]
            card = (suit-1)*13+rank
            D_new[row, col] = card

    return D_new

if __name__ == '__main__':
    '''
    Main method
    '''

    dataset = 'MNIST'

    print('Reading data...')

    if dataset == 'MNIST':
        ## MNIST dataset prep
        mat_label = scipy.io.loadmat('./MNSIT_mats/training_labels.mat')
        mat_imgs  = scipy.io.loadmat('./MNSIT_mats/training_images.mat')

        mat_t_label = scipy.io.loadmat('./MNSIT_mats/test_labels.mat')
        mat_t_imgs  = scipy.io.loadmat('./MNSIT_mats/test_images.mat')

        print('Data loaded.')

        D_x = mat_imgs['training_images']
        D_y = mat_label['training_labels']

        Dt_x = mat_t_imgs['test_images']
        Dt_y = mat_t_label['test_labels']


        D = np.concatenate((D_x, D_y), 1)
        Dt = np.concatenate((Dt_x, Dt_y), 1)

        train = D
        test = Dt

        # print(D.shape)
        # print(Dt.shape)

    else:
        ## Poker dataset prep
        poker = Data('poker', n_features=10)
        poker_test = Data('poker.t', n_features=10)

        D = poker.read_data()
        Dt = poker_test.read_data()
        print('Data loaded.')

        # D = suit_rank_to_card_number(D)

        N, d = D.shape
        Nt, dt = Dt.shape

        inds_test = set(np.random.choice(Nt, int(N*0.2), replace=False))
        test = [Dt[i, :] for i in inds_test]
        test = np.array(test)
        train = D

        # test = suit_rank_to_card_number(test)
        # train = suit_rank_to_card_number(train)

    # calc_freq(D[:,-1])
    print('Data prepped.')


    ## Muliclass prediction
    print('Generating Multiclass random forests')

    for j in [.0001, .001, .01, .1]:
        for i in [1, 10, 100]:
            print('parameters: ', end='')
            print((i, j, 20))
            forest = RandomForest(train, i, j, 20)
            print('Forest generated. Now calculating test error...')
            print('test error: ' + str(forest.error_rate(test)))
            print('Confusion Matrix: ')
            print(forest.confusion_mx(test, n_classes=10))

    ## OVA
    print('Generating OVA Random Forests')
    n_classes = 10

    Nt, dt = test.shape

    for j in [.2]:
        for i in [1, 10, 100]:
            print('parameters: ', end='')
            param = (i, j, 20)
            print(param)

            # Train OVA classes
            classifiers = []
            for c in range(n_classes):

                tmp_D = np.copy(D)
                D_label = tmp_D[:, -1]
                D_label[D_label != c] = -1.0
                D_label[D_label == c] = 1.0
                # print(np.sum(D_label))
                tmp_D[:, -1] = D_label

                tmp_forest = RandomForest(train=tmp_D, trees=i, subsample=j, linspace=20)
                classifiers.append(tmp_forest)

            predictions = np.zeros((Nt, n_classes), dtype=tuple)

            # Test OVA classes
            for l, c in enumerate(classifiers):
                for m in range(Nt):
                    predictions[m, l] = c.svm_predict(test[m, :-1])

            # Gen Confusion Matrix
            CMX, err = ova_confusion_err(predictions, test, n_classes)
            print(err)
            print(CMX)

