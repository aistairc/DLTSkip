# -*- coding: utf-8 -*-

import data
import comp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import util
from sklearn import metrics

iter_num = 100  # too small?


class GridSearch:
    def __init__(self):
        pass

    def fit(self, grid_params):
        self.grid_params = grid_params
        print
        print('*** fit')
        # grid search for each division
        for division in range(self.num_divisions):
            if division == 0:
                results = self.grid_search(division)
            else:
                results += self.grid_search(division)
        # calculate error rates
        error_rates = 1.0 - 1.0*results[:, 0]/results[:, 1]
        # output cross validation result as csv
        f = open('validation_result_' + self.name + '.csv', 'a')
        string = 'name'
        string += ', error_rate'
        for key in self.params[0]:
            if key != 'classifier':
                string += ', ' + key
        f.write(string)
        f.write('\n')
        for (error_rate, param) in zip(error_rates, self.params):
            string = self.name
            string += ', ' + str(error_rate)
            for (key, value) in param.items():
                if key != 'classifier':
                    string += ', ' + str(value)
            f.write(string)
            f.write('\n')
        f.close()
        # return best param
        best_param = self.params[error_rates.argmin()]
        return best_param

    def grid_search(self, division):
        ts = self.ts
        # assign training / validation data
        ts.assign_training_validation_data([division])
        print()
        print('division =', division+1, '/', self.num_divisions)
        print('training =', ts.training_data_list)
        print('validation =', ts.validation_data_list)
        # grid search loop
        params = []
        results = []
        # loop for training & validation data extraction
        for time_scale in self.grid_params['time_scale']:
            if time_scale < 1:
                continue
            for window_size in self.grid_params['window_size']:
                if window_size < 1:
                    continue
                if (window_size-1)*time_scale >= ts.min_length:
                    continue
                ts_extract_params = {
                    'target': ts.normalized,
                    'time_scale': time_scale,
                    'window_size': window_size,
                    'window_stride': 1,
                }
                (Xs_training, windowses_training, labels_training
                 ) = ts.extract_training_data(ts_extract_params)
                (Xs_validation, windowses_validation, labels_validation
                 ) = ts.extract_validation_data(ts_extract_params)
                # loop for dictionar learning
                for num_atoms_ratio in self.grid_params['num_atoms_ratio']:
                    num_atoms = int(window_size * num_atoms_ratio)
                    if num_atoms <= 0:
                        continue
                    print('time_scale={:g}, window_size={:g}, num_atoms={:g}'.format(
                            time_scale, window_size, num_atoms))
                    sd_fit_params = {
                        'lambda1': 0.1,
                        'num_atoms': num_atoms,
                        'iter_min': iter_num,
                        'iter_max': iter_num,
                    }
                    sd = comp.SparseDecomposition()
                    sd.fit(Xs_training, sd_fit_params)
                    # loop for compression & extraction
                    for lambda1 in self.grid_params['lambda1']:
                        if lambda1 < 0.0:
                            continue
                        for interval_ratio in self.grid_params['interval_ratio']:
                            interval = int(
                                    window_size * time_scale * interval_ratio)
                            if interval < 1:
                                continue
                            for extract_mode in self.grid_params['extract_mode']:
                                sd_compress_params = {
                                    'lambda1': lambda1,
                                    'mode': 'fixed',  # 'fixed', 'variable'
                                    'interval': interval,
                                }
                                sd_extract_params = {
                                    'mode': extract_mode
                                }
                                # training
                                sd.compress(Xs_training, windowses_training, labels_training, sd_compress_params)
                                (X, y_actual) = sd.extract_augmented(sd_extract_params)
                                for classifier in self.grid_params['classifier']:
                                    classifier.fit(X, y_actual)
                                # validation
                                sd.compress(Xs_validation, windowses_validation, labels_validation, sd_compress_params)
                                (X, y_actual) = sd.extract(sd_extract_params)
                                for (classifier, classifier_name) in zip(self.grid_params['classifier'], self.grid_params['classifier_name']):
                                    y_predicted = classifier.predict(X)
                                    confusion_matrix = metrics.confusion_matrix(y_actual, y_predicted)
                                    num_accuracy = int(np.diag(confusion_matrix).sum())
                                    num_total = int(confusion_matrix.sum())
                                    result = [num_accuracy, num_total]
                                    results.append(result)
                                    param = {
                                        'time_scale': time_scale,
                                        'window_size': window_size,
                                        'lambda1': lambda1,
                                        'num_atoms': num_atoms,
                                        'interval': interval,
                                        'extract_mode': extract_mode,
                                        'classifier': classifier,
                                        'classifier_name': classifier_name,
                                    }
                                    params.append(param)
        self.params = params
        return np.array(results)

    def predict(self, params, print_predict=True, file_name=None):
        if print_predict:
            print('\n*** predict')
        ts = self.ts
        total_test_length = np.array(ts.lengths)[
                np.array(ts.modes) == data.test].sum()
        compression_rates = []
        RMSEs = []
        error_rates = []
        for param in params:
            if print_predict:
                print()
                print(param)
            # set param
            time_scale = param['time_scale']
            window_size = param['window_size']
            ts_extract_params = {
                'target': ts.normalized,
                'time_scale': time_scale,
                'window_size': window_size,
                'window_stride': 1,
            }
            lambda1 = param['lambda1']
            num_atoms = param['num_atoms']
            sd_fit_params = {
                'lambda1': lambda1,
                'num_atoms': num_atoms,
                'iter_min': iter_num,
                'iter_max': iter_num,
            }
            interval = param['interval']
            extract_mode = param['extract_mode']
            sd_compress_params = {
                'lambda1': lambda1,
                'mode': 'fixed',  # 'fixed', 'variable'
                'interval': interval,
            }
            sd_extract_params = {
                'mode': extract_mode
            }
            ts_reconstruct_params = {
                'coef_noised': None,
                'interp1d_kind': 'cubic',  # 'linear', 'cubic', ... cf. scipy.interpolate.interp1d
            }
            classifier = param['classifier']
            # training
            sd = comp.SparseDecomposition()
            (Xs_training, windowses_training, labels_training
             ) = ts.extract_training_validation_data(ts_extract_params)
            sd.fit(Xs_training, sd_fit_params)
            sd.compress(Xs_training, windowses_training, labels_training,
                        sd_compress_params)
            (X, y_actual) = sd.extract_augmented(sd_extract_params)
            classifier.fit(X, y_actual)
            # test
            (Xs_test, windowses_test, labels_test
             ) = ts.extract_test_data(ts_extract_params)
            num_nonzeros = sd.compress(Xs_test, windowses_test, labels_test,
                                       sd_compress_params)
            compression_rate = 1.0 * num_nonzeros / total_test_length
            (Zs_test, windows_indices_test) = sd.decompress()
            RMSE = ts.reconstruct(Zs_test, windowses_test,
                                  ts_reconstruct_params, windows_indices_test)
            (X, y_actual) = sd.extract(sd_extract_params)
            y_predicted = classifier.predict(X)
            confusion_matrix = metrics.confusion_matrix(y_actual, y_predicted)
            error_rate = 1.0 - metrics.accuracy_score(y_actual, y_predicted)
            if print_predict:
                print('compression_rate =', compression_rate)
                print('RMSE =', RMSE)
                print(confusion_matrix)
                print('error_rate =', error_rate)
            compression_rates.append(compression_rate)
            RMSEs.append(RMSE)
            error_rates.append(error_rate)
        # output test result as csv
        if file_name is None:
            f = open('test_result_' + self.name + '.csv', 'a')
        else:
            f = open(file_name, 'a')
        string = 'name'
        string += ', error_rate'
        for (key, value) in params[0].items():
            if key != 'classifier':
                string += ', ' + key
        f.write(string)
        f.write('\n')
        for (param, error_rate) in zip(params, error_rates):
            string = self.name
            string += ', ' + str(error_rate)
            for (key, value) in param.items():
                if key != 'classifier':
                    string += ', ' + str(value)
            f.write(string)
            f.write('\n')
        f.close()
        self.sd = sd
        self.classifier = classifier
        return compression_rates, RMSEs, error_rates


class GridSearchUCR(GridSearch):
    def __init__(self, name, num_divisions=1):
        self.name = name
        self.num_divisions = num_divisions
        print()
        print('**************************************************************')
        print(name)
        ts_read_params = {
            'type': 'UCR',
            'dir_name': '../dataset/UCR_TS_Archive_2015/' + name + '/',
        }
        ts_normalize_params = {
            'mode': 'whole',  # 'whole', 'window', None
        }
        ts_divide_params = {
            'num_divisions': num_divisions,
            'mode': 'label_balanced_except_test',  # 'label_balanced_except_test', 'label_balanced', 'balanced_except_test', 'balanced'
            'seed': 0,
        }
        ts = data.TimeSeries()
        ts.read(ts_read_params)
        ts.normalize(ts_normalize_params)
        ts.divide(ts_divide_params)
        self.ts = ts


def generate_params(target, grid, best_param):
    params = []
    for value in grid:
        param = best_param.copy()
        param[target] = value
        params.append(param)
    return params


def plot_dependency(target, grid, compression_rates, RMSEs, error_rates,
                    best_param, best_param_compression_rate, best_param_RMSE,
                    best_param_error_rate, xscale='linear'):
    if target == 'interval':
        interval_line = (best_param['window_size'] - 1
                         ) * best_param['time_scale'] + 1
    plt.figure(figsize=(util.plot_unit_width * 3, util.plot_unit_height * 3))
    '''
    ax = plt.subplot(3, 1, 1)
    plt.ylabel('compression rate')
    if target == 'interval':
        plt.axvline(interval_line, linestyle='--', color='gray')
    plt.plot(grid, compression_rates, 'o-')
    plt.plot(best_param[target], best_param_compression_rate, '*',
             markersize=10, color='red')
    plt.ylim(ymin=0.0)
    '''
#    plt.subplot(3, 1, 2, sharex=ax)
    ax = plt.subplot(2, 1, 1)
    plt.ylabel('average RMSE')
    if target == 'interval':
        plt.axvline(interval_line, linestyle='--', color='gray')
    plt.plot(grid, RMSEs, 'o-')
    plt.plot(best_param[target], best_param_RMSE, '*',
             markersize=10, color='red')
    plt.ylim(ymin=0.0)
#    plt.subplot(3, 1, 3, sharex=ax)
    plt.subplot(2, 1, 2, sharex=ax)
    plt.xlabel(target)
    plt.ylabel('TSC test error rate')
    if target == 'interval':
        plt.axvline(interval_line, linestyle='--', color='gray')
    plt.plot(grid, error_rates, 'o-')
    plt.plot(best_param[target], best_param_error_rate, '*',
             markersize=10, color='red')
    plt.ylim(ymin=0.0)
    plt.xscale(xscale)
    plt.tight_layout()
#    plt.show()
    plt.savefig(target+'_dependency.pdf')
