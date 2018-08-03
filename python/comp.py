# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
#sys.path.append('/home/genta/work/spams-python/install/lib/python2.7/site-packages')
import spams
import util


class Compressor:
    def print_setting(self):
        print()
        print(self.__class__.__name__)


class SparseDecomposition(Compressor):
    def __init__(self, plot_params=None):
        if plot_params is None:
            self.plot_params = {
                'cost': True,
                'dendrogram': False,
                'dictionary': False,
                'row_atoms': 10,
                'decomposition': False,
                'num_decompositions': 1,
                'alpha': False,
                'alpha_thinned': False,
            }
        else:
            self.plot_params = plot_params

    '''
    def fit(self, Xs, fit_params):
        self.fit_params = fit_params
        lambda1 = fit_params['lambda1']
        lambda2 = 0.0
        num_atoms = fit_params['num_atoms']
        batch_size = 512 # default
        iter_min = fit_params['iter_min']
        params_dl = {
            'mode': spams.PENALTY,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'K': num_atoms,
            'posAlpha': True,
            'posD': False,
            'numThreads': -1,
            'batchsize': batch_size,
            'iter': iter_min,
            'return_model': True,
            'verbose': False,
        }
        params_sd = {
            'mode': spams.PENALTY,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'pos': True,
            'numThreads': -1,
        }
        X = self.stack(Xs)
        self.train_dictionary(X, params_dl, params_sd)
        if self.plot_params['dictionary']:
            self.plot_dictionary()
    '''

    def fit(self, Xs, fit_params):
        self.fit_params = fit_params
        lambda1 = fit_params['lambda1']
        lambda2 = 0.0
        num_atoms = fit_params['num_atoms']
        batch_size = 512  # default
        iter_min = fit_params['iter_min']
        params_dl = {
            'mode': spams.PENALTY,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'K': num_atoms,
            'posAlpha': True,
            'posD': False,
            'numThreads': -1,
            'batchsize': batch_size,
            'iter': iter_min,
            'return_model': False,
            'verbose': False,
        }
        params_sd = {
            'mode': spams.PENALTY,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'pos': True,
            'numThreads': -1,
        }
        X = self.stack(Xs)
        self.train_dictionary(X, params_dl, params_sd)
        if self.plot_params['dictionary']:
            self.plot_dictionary()

    def stack(self, Xs):
        for (i, X) in enumerate(Xs):
            window_size = X.shape[1]
            num_sensors = X.shape[2]
            if i == 0:
                stacked = X.reshape([-1, window_size*num_sensors]).T
            else:
                stacked = np.c_[stacked, X.reshape([-1, window_size*num_sensors]).T]
        self.window_size = window_size
        self.num_sensors = num_sensors
        return stacked

    '''
    def train_dictionary(self, X, params_dl, params_sd):
        iter_max = self.fit_params['iter_max']
        X = np.asfortranarray(X)
        #tic = time.time()
        (D, model) = spams.trainDL(X, **params_dl)
        #tac = time.time()
        #print '{0} sec / {1} iter'.format(tac - tic, model['iter'])
        alpha = spams.lasso(X, D, **params_sd)
        cost = [self.get_cost(X, D, alpha, params_dl)]
        while model['iter'] < iter_max:
            (D, model) = spams.trainDL(X, model=model, D=D, **params_dl)
            alpha = spams.lasso(X, D, **params_sd)
            cost.append(self.get_cost(X, D, alpha, params_dl))
        if self.plot_params['cost'] and len(cost) > 1:
            plt.title('cost = {0:g}'.format(cost[-1]))
            plt.plot(cost)
            plt.show()
        self.D = D
    '''

    def train_dictionary(self, X, params_dl, params_sd):
        X = np.asfortranarray(X)
#        tic = time.time()
        D = spams.trainDL(X, **params_dl)
#        tac = time.time()
#        print('{0} sec / {1} iter'.format(tac - tic, model['iter']))
        self.D = D

    def get_cost(self, X, D, alpha, params_dl):
        mode = params_dl['mode']
        lambda1 = params_dl['lambda1']
        lambda2 = params_dl['lambda2']
        A = alpha.toarray()
        if mode == 0:
            cost = 0.5 * np.power(np.linalg.norm(X - D * alpha, 2), 2)
        elif mode == 1:
            cost = np.linalg.norm(A, 1)
        elif mode == 2:
            cost = 0.5 * np.power(
                    np.linalg.norm(X - D * alpha, 2), 2
                    ) + lambda1 * np.linalg.norm(A, 1
                    ) + 0.5 * lambda2 * np.power(np.linalg.norm(A, 2), 2)
        elif mode == 3:
            cost = 0.5 * np.power(np.linalg.norm(X - D * alpha, 2), 2)
        elif mode == 4:
            cost = np.linalg.norm(A, 0)
        elif mode == 5:
            cost = 0.5 * np.power(
                    np.linalg.norm(X - D * alpha, 2), 2
                    ) + lambda1 * np.linalg.norm(A, 0)
        else:
            cost = 0.0
        return cost

    def plot_dictionary(self):
        D = self.D
        window_size = self.window_size
        num_sensors = self.num_sensors
        num_atoms = D.shape[1]
        row_atoms = self.plot_params['row_atoms']
        plt.figure(figsize=(util.plot_unit_width * row_atoms,
                            util.plot_unit_height * (
                                    (num_atoms - 1) / row_atoms + 1)))
        for a in range(num_atoms):
            plt.subplot((num_atoms-1)/row_atoms+1, row_atoms, a+1)
            plt.title('[' + str(a) + ']')
            plt.ylim([np.min(D), np.max(D)])
            plt.plot(D[:, a].reshape([window_size, num_sensors]))
        plt.tight_layout()
        plt.show()

    def compress(self, Xs, windowses, labels, compress_params):
        self.compress_params = compress_params
        window_size = self.window_size
        num_sensors = self.num_sensors
        lambda1 = compress_params['lambda1']
        lambda2 = 0.0
        params_sd = {
            'mode': spams.PENALTY,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'pos': True,
            'numThreads': -1,
        }
        alphas = []
        windows_indices = []
        for (X, windows) in zip(Xs, windowses):
            alpha = self.decompose(X.reshape(
                    [-1, window_size * num_sensors]).T, params_sd)
            windows_index = self.thin_windows(alpha, windows)
            alphas.append(alpha)
            windows_indices.append(windows_index)
            if self.plot_params['decomposition']:
                self.plot_decomposition(X, alpha)
            if self.plot_params['alpha']:
                self.plot_alpha(alpha)
            if self.plot_params['alpha_thinned']:
                self.plot_alpha(alpha, windows_index)
        self.alphas = alphas
        self.windows_indices = windows_indices
        self.labels = labels
        num_nonzeros = 0
        for (alpha, window_index) in zip(alphas, windows_indices):
            A = alpha.toarray()
            num_nonzeros += ((A[:, windows_index] != 0.0) * 1).sum()
        return num_nonzeros

    def decompress(self):
        D = self.D
        window_size = self.window_size
        num_sensors = self.num_sensors
        Rs = []
        for alpha in self.alphas:
            A = alpha.toarray()
            R = D.dot(A)
            R = R.T.reshape([-1, window_size, num_sensors])
            Rs.append(R)
        return (Rs, self.windows_indices)

    def decompose(self, X, params_sd):
        X = np.asfortranarray(X)
        alpha = spams.lasso(X, self.D, **params_sd)
        return alpha

    def thin_windows(self, alpha, windows):
        num_windows = len(windows)
        mode = self.compress_params['mode']
        if mode == 'fixed':
            interval = self.compress_params['interval']
            windows_index = list(range(0, num_windows, interval))
            if windows_index[-1] != num_windows-1:
                windows_index.append(num_windows-1)
        elif mode == 'variable':
            A = alpha.toarray()
            windows_index = [0] + self.segment(A) + [num_windows-1]
        else:
            windows_index = range(0, num_windows)
        return windows_index

    def segment(self, A, windows=None):
        interval_max = self.compress_params['interval_max']
        interval_min = self.compress_params['interval_min']
        if windows is None:
            if A.shape[1] > interval_max:
                a = A[:, interval_min: -interval_min].max(axis=1).argmax()
                w = A[a, interval_min: -interval_min].argmax() + interval_min
                return self.segment(A[:, :w+1]) + [w] + map(
                        lambda x: x + w, self.segment(A[:, w:]))
            else:
                return []
        else:  # use windows information
            return []

    def plot_decomposition(self, X, alpha):
        D = self.D
        window_size = self.window_size
        num_sensors = self.num_sensors
        R = D.dot(alpha.toarray())
        R = R.T.reshape([-1, window_size, num_sensors])
        num_windows = X.shape[0]
        row_atoms = self.plot_params['row_atoms']
        for w in range(num_windows):
            if w >= self.plot_params['num_decompositions']:
                break
            plt.figure(figsize=(util.plot_unit_width * (row_atoms + 1),
                                util.plot_unit_height))
            # plot X & R
            plt.subplot(1, row_atoms + 1, 1)
            plt.plot(X[w], 'o-')
            plt.plot(R[w])
            # plot alpha
            col = 2
            for indptr in range(alpha.indptr[w], alpha.indptr[w+1]):
                plt.subplot(1, row_atoms+1, col)
                plt.title('[{0}] {1:.3f}'.format(
                        alpha.indices[indptr], alpha.data[indptr]))
                plt.ylim([D.min(), D.max()])
                plt.yticks([])
                plt.plot(D[:, alpha.indices[indptr]].reshape(
                        [window_size, num_sensors]))
                if col > row_atoms:
                    break
                col += 1
            plt.show()

    def plot_alpha(self, alpha, windows_index=None):
        A = alpha.toarray()
        if windows_index is not None:
            mask = np.zeros(A.shape[1])
            mask[windows_index] = 1
            A *= mask
        compression_rate = 1.0 * A.nonzero()[0].size / A.size
        plt.figure(figsize=(A.shape[1] * util.plot_cell_size,
                            A.shape[0] * util.plot_cell_size))
        plt.title('compression rate = {0:g}'.format(compression_rate))
        sns.heatmap(A, xticklabels=False, yticklabels=False,
                    cbar=False, square=True)
        plt.show()

    def aggregate(self, alpha, windows_index):
        mode = self.extract_params['mode']
        A = alpha.toarray()
        if mode == 'max':
            aggregated = A[:, windows_index].max(axis=1)
        elif mode == 'sum':
            aggregated = A[:, windows_index].sum(axis=1)
        elif mode == 'mean':
            aggregated = A[:, windows_index].mean(axis=1)
        elif mode == 'hist':
            A_binary = A[:, windows_index]
            A_binary[A_binary > 0.0] = 1
            aggregated = A_binary.sum(axis=1)
        elif mode == 'hist_max':
            num_atoms = A.shape[0]
            aggregated = np.histogram(A[:, windows_index].argmax(axis=0),
                                      bins=num_atoms, range=(0, num_atoms))[0]
        feature = aggregated
        return feature

    def extract_augmented(self, extract_params):
        return self.extract(extract_params, augmented=True)

    def extract(self, extract_params, augmented=False):
        self.extract_params = extract_params
        if augmented:
            mode = self.compress_params['mode']
            interval = self.compress_params['interval']
            features = []
            labels = []
            if mode == 'fixed':
                for (alpha, label) in zip(self.alphas, self.labels):
                    num_windows = alpha.shape[1]
                    for i in range(interval):
                        windows_index = range(i, num_windows, interval)
                        if len(windows_index) == 0:
                            break
                        feature = self.aggregate(alpha, windows_index)
                        features.append(feature)
                        labels.append(label)
            elif mode == 'variable':  # XXX 未定義
                pass
            return (np.array(features), np.array(labels))
        else:
            features = []
            for (alpha, windows_index) in zip(
                    self.alphas, self.windows_indices):
                feature = self.aggregate(alpha, windows_index)
                features.append(feature)
            return (np.array(features), np.array(self.labels))


class SingularValueDecomposition(Compressor):  # SVD
    def __init__(self, plot_params=None):
        if plot_params is None:
            self.plot_params = {
                'basis': True,
                'row_basis': 10,
                'coefficient': True,
                'coefficient_truncated': False,
                'coefficient_thinned': True,
            }
        else:
            self.plot_params = plot_params

    def fit(self, Xs, fit_params):
        self.fit_params = fit_params
        mode = fit_params['mode']
        if mode == 'window_sensor_vs_time' or mode == 'window_vs_sensor_time':
            X = self.stack(Xs)
            (U, s, V) = np.linalg.svd(X)
#            self.U = U
#            self.S = np.diag(s)
            self.V = V
            if self.plot_params['basis']:
                self.plot_basis()

    def stack(self, Xs):
        mode = self.fit_params['mode']
        if mode == 'window_sensor_vs_time':
            for (i, X) in enumerate(Xs):
                window_size = X.shape[1]
                num_sensors = X.shape[2]
                if i == 0:
                    stacked = X.transpose((0, 2, 1)).reshape([-1, window_size])
                else:
                    stacked = np.r_[stacked, X.transpose((0, 2, 1)).reshape(
                            [-1, window_size])]
            self.window_size = window_size
            self.num_sensors = num_sensors
            return stacked
        elif mode == 'window_vs_sensor_time':
            for (i, X) in enumerate(Xs):
                window_size = X.shape[1]
                num_sensors = X.shape[2]
                if i == 0:
                    stacked = X.reshape([-1, window_size * num_sensors])
                else:
                    stacked = np.r_[stacked, X.reshape(
                            [-1, window_size * num_sensors])]
            self.window_size = window_size
            self.num_sensors = num_sensors
            return stacked

    def plot_basis(self):
        V = self.V
        V_max = V.max()
        V_min = V.min()
        num_basis = V.shape[0]
        mode = self.fit_params['mode']
        row_basis = self.plot_params['row_basis']
        if mode == 'window_sensor_vs_time':
            plt.figure(figsize=(util.plot_unit_width * row_basis,
                                util.plot_unit_height * (
                                        (num_basis - 1) / row_basis + 1)))
            for b in range(num_basis):
                plt.subplot((num_basis - 1) / row_basis + 1, row_basis, b + 1)
                plt.title('[' + str(b) + ']')
                plt.ylim([V_min, V_max])
                plt.plot(V[b])
            plt.tight_layout()
            plt.show()
        elif mode == 'window_vs_sensor_time':
            window_size = self.window_size
            num_sensors = self.num_sensors
            plt.figure(figsize=(util.plot_unit_width * row_basis,
                                util.plot_unit_height * (
                                        (num_basis - 1) / row_basis + 1)))
            for b in range(num_basis):
                plt.subplot((num_basis-1)/row_basis+1, row_basis, b+1)
                plt.title('[' + str(b) + ']')
                plt.ylim([V_min, V_max])
                plt.plot(V[b].reshape([window_size, num_sensors]))
            plt.tight_layout()
            plt.show()

    def compress(self, Xs, windowses, compress_params):
        self.compress_params = compress_params
        mode = self.fit_params['mode']
        if mode == 'sensor_vs_time':
            Css = []
            Vss = []
            windows_indices = []
            for (X, windows) in zip(Xs, windowses):
                num_windows = X.shape[0]
                Cs = []
                Vs = []
                for w in range(num_windows):
                    (U, s, V) = np.linalg.svd(X[w].T, full_matrices=False)
                    US = U.dot(np.diag(s))
                    C = self.truncate(US)
                    Cs.append(C)
                    Vs.append(V)
                Css.append(Cs)
                Vss.append(Vs)
                windows_index = self.thin_windows(windows)
                windows_indices.append(windows_index)
            self.window_size = Xs[0].shape[1]
            self.num_sensors = Xs[0].shape[2]
            self.Css = Css
            self.Vss = Vss
            self.windows_indices = windows_indices
        elif mode == 'window_sensor_vs_time':
            V_inv = self.V.T  # np.linalg.inv(self.V)
            window_size = self.window_size
            num_sensors = self.num_sensors
            Cs = []
            windows_indices = []
            for (X, windows) in zip(Xs, windowses):
                US = X.transpose((0, 2, 1)).reshape(
                        [-1, window_size]).dot(V_inv)
                C = self.truncate(US)
                windows_index = self.thin_windows(windows)
                Cs.append(C)
                windows_indices.append(windows_index)
                if self.plot_params['coefficient']:
                    self.plot_coefficient(US)
                if self.plot_params['coefficient_truncate']:
                    self.plot_coefficient(C)
                if self.plot_params['coefficient_thin']:
                    self.plot_coefficient(C, windows_index)
            self.Cs = Cs
            self.windows_indices = windows_indices
        elif mode == 'window_vs_sensor_time':
            V_inv = self.V.T  # np.linalg.inv(self.V)
            window_size = self.window_size
            num_sensors = self.num_sensors
            Cs = []
            windows_indices = []
            for (X, windows) in zip(Xs, windowses):
                US = X.reshape([-1, window_size*num_sensors]).dot(V_inv)
                C = self.truncate(US)
                windows_index = self.thin_windows(windows)
                Cs.append(C)
                windows_indices.append(windows_index)
                if self.plot_params['coefficient']:
                    self.plot_coefficient(US)
                if self.plot_params['coefficient_truncated']:
                    self.plot_coefficient(C)
                if self.plot_params['coefficient_thinned']:
                    self.plot_coefficient(C, windows_index)
            self.Cs = Cs
            self.windows_indices = windows_indices

    def truncate(self, US):
        k = self.compress_params['k']
        mode = self.compress_params['mode']
        if mode == 'shared':
            mask = np.zeros(US.shape)
            mask[:, :k] = 1
            C = US * mask
        elif mode == 'unshared':
            num_windows = US.shape[0]
            argsort = np.absolute(US).argsort(axis=1)[:, ::-1][:, :k]
            mask = np.zeros(US.shape)
            for w in range(num_windows):
                mask[w, argsort[w]] = 1
            C = US * mask
        return C

    def thin_windows(self, windows):
        num_windows = len(windows)
        interval = self.compress_params['interval']
        windows_index = range(0, num_windows, interval)
        if windows_index[-1] != num_windows-1:
            windows_index.append(num_windows-1)
        return windows_index

    def plot_coefficient(self, C, windows_index=None):
        mode = self.fit_params['mode']
        if mode == 'sensor_vs_time':
            pass
        elif mode == 'window_sensor_vs_time':
            num_sensors = self.num_sensors
            if windows_index is not None:
                mask = np.zeros(C.shape)
                for s in range(num_sensors):
                    mask[np.array(windows_index) * num_sensors+s] = 1
                C *= mask  # XXX updating C is not good
        elif mode == 'window_vs_sensor_time':
            if windows_index is not None:
                mask = np.zeros(C.shape)
                mask[windows_index] = 1
                C *= mask  # XXX updating C is not good
        compression_rate = 1.0 * C.nonzero()[0].size / C.size
        plt.figure(figsize=(C.shape[0] * util.plot_cell_size,
                            C.shape[1] * util.plot_cell_size))
        plt.title('compression rate = {0:g}'.format(compression_rate))
        sns.heatmap(C.T, xticklabels=False, yticklabels=False,
                    cbar=False, square=True)
        plt.show()

    def decompress(self):
        mode = self.fit_params['mode']
        if mode == 'sensor_vs_time':
            Rss = []
            for (Cs, Vs) in zip(self.Css, self.Vss):
                Rs = []
                for (C, V) in zip(Cs, Vs):
                    R = C.dot(V).T
                    Rs.append(R)
                Rss.append(np.array(Rs))
            return (Rss, self.windows_indices)
        elif mode == 'window_sensor_vs_time':
            V = self.V
            window_size = self.window_size
            num_sensors = self.num_sensors
            Rs = []
            for C in self.Cs:
                R = C.dot(V)
                R = R.reshape([-1, num_sensors, window_size]).transpose(
                        (0, 2, 1))
                Rs.append(R)
            return (Rs, self.windows_indices)
        elif mode == 'window_vs_sensor_time':
            V = self.V
            window_size = self.window_size
            num_sensors = self.num_sensors
            Rs = []
            for C in self.Cs:
                R = C.dot(V)
                R = R.reshape([-1, window_size, num_sensors])
                Rs.append(R)
            return (Rs, self.windows_indices)


class PiecewiseAggregateApproximation(Compressor):  # PAA
    def __init__(self):
        pass

    def compress(self, Xs, windowses, compress_params):
        self.compress_params = compress_params
        As = []
        windows_indices = []
        for (X, windows) in zip(Xs, windowses):
            A = X.mean(axis=1)
            windows_index = self.thin_windows(windows)
            As.append(A)
            windows_indices.append(windows_index)
        self.As = As
        self.windows_indices = windows_indices

    def thin_windows(self, windows):
        num_windows = len(windows)
        interval = self.compress_params['interval']
        windows_index = range(0, num_windows, interval)
        if windows_index[-1] != num_windows-1:
            windows_index.append(num_windows-1)
        return windows_index

    def decompress(self):
        interval = self.compress_params['interval']
        Rs = []
        for (A, windows_index) in zip(self.As, self.windows_indices):
            num_sensors = A.shape[1]
            R = A.repeat(interval, axis=0).reshape([-1, interval, num_sensors])
            Rs.append(R)
        return (Rs, self.windows_indices)

#class SymbolicAggregateApproximation(Compressor):  # SAX
#class PiecewiseLinearApproximation(Compressor):  # PLA
#class AdaptivePiecewiseConstantApproximation(Compressor):  # APCA
#class DiscreteFourierTransform(Compressor):  # DFT
#class DiscreteWaveletTransform(Compressor):  # DWT
