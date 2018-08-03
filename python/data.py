# -*- coding: utf-8 -*-

import os
import random
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import util
from scipy import stats
from scipy import interpolate


def read_ucr(file_name):
    arr = np.loadtxt(file_name, delimiter=',')
    num_data = arr.shape[0]
    dfs = []
    labels = []
    for i in range(num_data):
        df = pd.DataFrame(arr[i, 1:])
        dfs.append(df)
        labels.append(int(arr[i, 0]))
    return (dfs, labels)


def read_mocap(file_name):
    if os.path.splitext(file_name)[1] == '.amc':
        print(file_name)
        f = open(file_name, 'r')
        for i in range(4):
            line = f.readline()
        values = []
        while line:
            value = []
            for i in range(29):
                line = f.readline()
                value += map(float, line.split(' ')[1:])
            values.append(value)
            line = f.readline()
        f.close()
        columns = [
            'root0', 'root1', 'root2', 'root3', 'root4', 'root5',
            'lowerback0', 'lowerback1', 'lowerback2',
            'upperback0', 'upperback1', 'upperback2',
            'thorax0', 'thorax1', 'thorax2',
            'lowerneck0', 'lowerneck1', 'lowerneck2',
            'upperneck0', 'upperneck1', 'upperneck2',
            'head0', 'head1', 'head2',
            'rclavicle0', 'rclavicle1',
            'rhumerus0', 'rhumerus1', 'rhumerus2',
            'rradius0',
            'rwrist0',
            'rhand0', 'rhand1',
            'rfingers0',
            'rthumb0', 'rthumb1',
            'lclavicle0', 'lclavicle1',
            'lhumerus0', 'lhumerus1', 'lhumerus2',
            'lradius0',
            'lwrist0',
            'lhand0', 'lhand1',
            'lfingers0',
            'lthumb0', 'lthumb1',
            'rfemur0', 'rfemur1', 'rfemur2',
            'rtibia0',
            'rfoot0', 'rfoot1',
            'rtoes0',
            'lfemur0', 'lfemur1', 'lfemur2',
            'ltibia0',
            'lfoot0', 'lfoot1',
            'ltoes0',
        ]
        df = pd.DataFrame(values, columns=columns)
        return df


def read_ecg(file_name):
    pass


def read_csv(file_name):
    pass


training = 0
validation = 1
test = 2


class TimeSeries:
    def __init__(self, plot_params=None):
        if plot_params is None:
            self.plot_params = {
                'reconstruction': False,
                'num_reconstructions': 10,
            }
        else:
            self.plot_params = plot_params

    def read(self, read_params):
        self.read_params = read_params
        read_type = read_params['type']
        dir_name = read_params['dir_name']
        if read_type == 'UCR':
            training_file_name = glob.glob(dir_name + '*_TRAIN')
            test_file_name = glob.glob(dir_name + '*_TEST')
            if len(training_file_name) == 1 and len(test_file_name) == 1:
                (dfs_training, labels_training) = read_ucr(
                        training_file_name[0])
                (dfs_test, labels_test) = read_ucr(test_file_name[0])
                self.dfs = dfs_training + dfs_test
                self.labels = labels_training + labels_test
                self.modes = [training] * len(
                        dfs_training) + [test] * len(dfs_test)
            else:
                print('error: cannot read UCR time series')
        elif read_type == 'CUSTOM':
            func = read_params['func']
            label_names = read_params['label_names']
            label_numbers = read_params['label_numbers']
            dfs = []
            labels = []
            for (label_name, label_number) in zip(label_names, label_numbers):
                file_names = glob.glob(dir_name + label_name + '/*.*')
                for file_name in file_names:
                    df = func(file_name)
                    dfs.append(df)
                    labels.append(label_number)
            self.dfs = dfs
            self.labels = labels
            self.modes = [training] * len(dfs)
        # set time series information
        dfs = self.dfs
        self.num_data = len(dfs)
        self.num_training_data = ((np.array(self.modes) != test) * 1).sum()
        self.num_test_data = ((np.array(self.modes) == test) * 1).sum()
        self.sensor_names = list(dfs[0].columns)
        self.num_sensors = dfs[0].columns.size
        lengths = np.array([len(x) for x in dfs])
        self.min_length = lengths.min()
        self.max_length = lengths.max()
        self.lengths = list(lengths)
        self.num_labels = len(set(self.labels))
        # print time series information
        print()
        print('num_data =', self.num_data)
        print('num_training_data =', self.num_training_data)
        print('num_test_data =', self.num_test_data)
        print('num_sensors =', self.num_sensors)
        print('min_length =', self.min_length)
        print('max_length =', self.max_length)
        print('num_labels =', self.num_labels)
        # convert to arrays
        original = []
        for df in dfs:
            original.append(df.as_matrix())
        self.original = original

    def normalize(self, normalize_params):
        self.normalize_params = normalize_params
        normalize_mode = normalize_params['mode']  # if normalize_mode == 'window', normalize at extract() !!!
        if normalize_mode == 'whole':
            normalized = []
            means = []
            stds = []
            for arr in self.original:
                mean = arr.mean(axis=0)
                means.append(mean)
                std = arr.std(axis=0)
                stds.append(std)
                norm = stats.zscore(arr, axis=0)
                norm[np.isnan(norm)] = 0.0  # norm[i] = nan if std[i] = 0
                normalized.append(norm)
            self.normalized = normalized
            self.means = means
            self.stds = stds
        else:
            self.normalized = self.original
            self.means = [0.0] * self.num_data
            self.stds = [1.0] * self.num_data

    def add_noise(self, noise_params):
        self.noise_params = noise_params
        noise_type = noise_params['type']
        # initialize seed
        seed = noise_params['seed']
        if seed >= 0:
            np.random.seed(seed)
        # add noise
        noised = []
        for arr in self.normalized:
            if noise_type == 'Gaussian':
                mu = noise_params['mu']
                sigma = noise_params['sigma']
                noised.apeend(arr + np.random.normal(mu, sigma, arr.shape))
        self.noised = noised

    def reconstruct(self, Zs, windowses, reconstruct_params,
                    windows_indices=None):
        self.reconstruct_params = reconstruct_params
        if windows_indices is None:
            windows_indices = []
            for windows in windowses:
                windows_index = range(len(windows))
                windows_indices.append(windows_index)
        num_sensors = self.num_sensors
        time_scale = self.extract_params['time_scale']
        coef_noised = reconstruct_params['coef_noised']
        interp1d_kind = reconstruct_params['interp1d_kind']
        reconstructed = [None] * self.num_data
        if self.normalize_params['mode'] != 'window':
            for (Z, windows, windows_index, i) in zip(
                    Zs, windowses, windows_indices, self.extract_index):
                length = windows[-1][1]
                arr = np.zeros([length, num_sensors])
                if coef_noised is None:
                    weight = np.zeros([length, num_sensors])
                else:  # if coef_noised is not None (for denoising)
                    weight = coef_noised * np.ones([length, num_sensors])
                for w in windows_index:
                    window = windows[w]
                    if time_scale == 1:
                        interpolated = Z[w, :, :]
                    else:
                        interpolated = []
                        for s in range(num_sensors):
                            interp_x = range(window[0], window[1], time_scale)
                            interp_y = Z[w, :, s]
                            interp = interpolate.interp1d(
                                    interp_x, interp_y, kind=interp1d_kind)
                            interpolated.append(interp(range(
                                    window[0], window[1])))
                        interpolated = np.array(interpolated).T
                    arr[window[0]:window[1], :] += interpolated
                    weight[window[0]:window[1], :] += np.ones(
                            [window[1]-window[0], num_sensors])
                arr /= weight
                arr[np.isnan(arr)] = 0.0
                reconstructed[i] = arr
        else:  # self.normalize_params['mode'] == 'window':
            for (Z, windows, windows_index, i, means, stds) in zip(
                    Zs, windowses, windows_indices,
                    self.extract_index, self.meanss, self.stdss):
                length = windows[-1][1]
                arr = np.zeros([length, num_sensors])
                if coef_noised is None:
                    weight = np.zeros([length, num_sensors])
                else:  # if coef_noised is not None (for denoising)
                    weight = coef_noised * np.ones([length, num_sensors])
                for w in windows_index:
                    denormalized = Z[w, :, :] * stds[w] + means[w]
                    window = windows[w]
                    if time_scale == 1:
                        interpolated = denormalized
                    else:
                        interpolated = []
                        for s in range(num_sensors):
                            interp_x = range(window[0], window[1], time_scale)
                            interp_y = denormalized[:, s]
                            interp = interpolate.interp1d(
                                    interp_x, interp_y, kind=interp1d_kind)
                            interpolated.append(interp(range(
                                    window[0], window[1])))
                        interpolated = np.array(interpolated).T
                    arr[window[0]:window[1], :] += interpolated
                    weight[window[0]:window[1], :] += np.ones(
                            [window[1]-window[0], num_sensors])
                arr /= weight
                arr[np.isnan(arr)] = 0.0
                reconstructed[i] = arr
        self.reconstructed = reconstructed
        if self.plot_params['reconstruction']:
            self.plot_reconstruction()
        RMSEs = []
        for (arr_norm, arr_recon) in zip(self.normalized, self.reconstructed):
            if arr_recon is not None:
                RMSE = util.get_rmse(arr_norm, arr_recon)
                RMSEs.append(RMSE)
        RMSE = np.array(RMSEs).mean()
        return RMSE

    def plot_reconstruction(self):
        print()
        print('plot reconstruction')
        i = 0
        for (arr_norm, arr_recon) in zip(self.normalized, self.reconstructed):
            if arr_recon is not None:
                length = min(arr_norm.shape[0], arr_recon.shape[0])
                plt.figure(figsize=(length / util.plot_time_unit,
                                    util.plot_unit_height * 3))
                ax = plt.subplot(3, 1, 1)
                plt.title('normalized')
                plt.plot(arr_norm)
                plt.subplot(3, 1, 2, sharex=ax, sharey=ax)
                plt.title('reconstructed')
                plt.plot(arr_recon)
                plt.subplot(3, 1, 3, sharex=ax)
                plt.title('RMSE = ' + str(util.get_rmse(arr_norm, arr_recon)))
                plt.plot(arr_norm - arr_recon)
#                plt.tight_layout()
                plt.subplots_adjust(bottom=0.05, top=0.95,
                                    right=0.95, left=0.05,
                                    wspace=0.25, hspace=0.25)
                plt.show()
                i += 1
                if i >= self.plot_params['num_reconstructions']:
                    break

    def denormalize(self):
        denormalized = []
        for (arr, mean, std) in zip(self.reconstructed, self.means, self.stds):
            denormalized.append(arr * std + mean)
        self.denormalized = denormalized

    def divide(self, divide_params):  # assign -1 for test data & 0, 1, ..., num_divisions-1 otherwise 
        self.divide_params = divide_params
        num_divisions = self.divide_params['num_divisions']
        divide_mode = self.divide_params['mode']
        seed = self.divide_params['seed']
        divisions = -np.ones(self.num_data)
        random.seed(seed)
        if divide_mode == 'label_balanced_except_test':
            modes = np.array(self.modes)
            labels = np.array(self.labels)
            label_numbers = list(set(labels))
            for label_number in label_numbers:
                indices = (modes != test) & (labels == label_number)
                length = len(divisions[indices])
                divisions[indices] = self.rand_arbitrary(num_divisions, length)
        if divide_mode == 'label_balanced':
            labels = np.array(self.labels)
            label_numbers = list(set(labels))
            for label_number in label_numbers:
                indices = (labels == label_number)
                length = len(divisions[indices])
                divisions[indices] = self.rand_arbitrary(num_divisions, length)
        elif divide_mode == 'balanced_except_test':
            modes = np.array(self.modes)
            indices = (modes != test)
            length = len(divisions[indices])
            divisions[indices] = self.rand_arbitrary(num_divisions, length)
        elif divide_mode == 'balanced':
            divisions = self.rand_arbitrary(num_divisions, self.num_data)
        self.divisions = list(map(int, list(divisions)))

    def rand_arbitrary(self, num_divisions, length):
        rand = list(range(num_divisions))
        random.shuffle(rand)
        rand *= length // num_divisions + 1
        rand = rand[:length]
        random.shuffle(rand)
        return rand

    def assign_training_validation_data(self, divisions):
        modes = self.modes
        for (i, division) in enumerate(self.divisions):
            if division >= 0:
                if division in divisions:
                    modes[i] = validation
                else:
                    modes[i] = training
            else:
                modes[i] = test
        self.modes = modes
        self.training_data_list = list(
                np.arange(len(modes))[np.array(modes) == training])
        self.validation_data_list = list(
                np.arange(len(modes))[np.array(modes) == validation])
        self.test_data_list = list(
                np.arange(len(modes))[np.array(modes) == test])

    def extract_training_data(self, extract_params):
        return self.extract(extract_params, extract_modes=[training])

    def extract_validation_data(self, extract_params):
        return self.extract(extract_params, extract_modes=[validation])

    def extract_test_data(self, extract_params):
        return self.extract(extract_params, extract_modes=[test])

    def extract_training_validation_data(self, extract_params):
        return self.extract(extract_params,
                            extract_modes=[training, validation])

    def extract(self, extract_params, extract_modes=None, extract_labels=None):
        self.extract_params = extract_params
        arrs = extract_params['target']
        windowses = []
        Xs = []
        labels = []
        extract_index = []
        if self.normalize_params['mode'] != 'window':
            for (i, arr) in enumerate(arrs):
                if extract_modes is not None:
                    if self.modes[i] not in extract_modes:
                        continue
                if extract_labels is not None:
                    if self.labels[i] not in extract_labels:
                        continue
                windows = self.generate_windows(arr)
                X = self.extract_windows(arr, windows)
                windowses.append(windows)
                Xs.append(X)
                labels.append(self.labels[i])
                extract_index.append(i)
        else:  # self.normalize_params['mode'] == 'window':
            meanss = []
            stdss = []
            for (i, arr) in enumerate(arrs):
                if extract_modes is not None:
                    if self.modes[i] not in extract_modes:
                        continue
                if extract_labels is not None:
                    if self.labels[i] not in extract_labels:
                        continue
                windows = self.generate_windows(arr)
                (X, means, stds) = self.normalize_extract_windows(arr, windows)
                windowses.append(windows)
                Xs.append(X)
                labels.append(self.labels[i])
                extract_index.append(i)
                meanss.append(means)
                stdss.append(stds)
            self.meanss = meanss
            self.stdss = stdss
        self.extract_index = extract_index
        return (Xs, windowses, labels)

    def generate_windows(self, arr):
        time_scale = self.extract_params['time_scale']
        window_size = self.extract_params['window_size']
        window_stride = self.extract_params['window_stride']
        length = arr.shape[0]
        begins = range(0, length-(window_size-1)*time_scale, window_stride)
        ends = range((window_size-1)*time_scale+1, length+1, window_stride)
        return list(zip(begins, ends))

    def extract_windows(self, arr, windows):
        time_scale = self.extract_params['time_scale']
        window_size = self.extract_params['window_size']
        num_sensors = arr.shape[1]
        num_windows = len(windows)
        X = np.zeros([num_windows, window_size, num_sensors])
        for (w, window) in enumerate(windows):
            X[w, :, :] = arr[window[0]:window[1]:time_scale, :]
        return X

    def normalize_extract_windows(self, arr, windows):
        time_scale = self.extract_params['time_scale']
        window_size = self.extract_params['window_size']
        num_sensors = arr.shape[1]
        num_windows = len(windows)
        X = np.zeros([num_windows, window_size, num_sensors])
        means = []
        stds = []
        for (w, window) in enumerate(windows):
            extracted = arr[window[0]:window[1]:time_scale, :]
            mean = extracted.mean(axis=0)
            means.append(mean)
            std = extracted.std(axis=0)
            stds.append(std)
            norm = stats.zscore(extracted, axis=0)
            norm[np.isnan(norm)] = 0.0  # norm[i] = nan if std[i] = 0
            X[w, :, :] = norm
        return (X, means, stds)

    def plot_arrays(self, arrs):
        for (arr, label) in zip(arrs, self.labels):
            if arr is not None:
                length = arr.shape[0]
                plt.figure(figsize=(length / util.plot_time_unit,
                                    util.plot_unit_height))
                plt.title(label)
                plt.plot(arr)
                plt.show()
