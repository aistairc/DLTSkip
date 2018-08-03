# -*- coding: utf-8 -*-

import tsc
from sklearn import neighbors, svm, tree, ensemble, naive_bayes, linear_model, discriminant_analysis

'''
names = [
    'Adiac',
    'Beef',
    'Car',
    'CBF',
    'ChlorineConcentration',
    'CinC_ECG_torso',
    'Coffee',
    'Cricket_X',
    'Cricket_Y',
    'Cricket_Z',
    'DiatomSizeReduction',
    'ECGFiveDays',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    '50words',
    'FISH',
    'Gun_Point',
    'Haptics',
    'InlineSkate',
    'ItalyPowerDemand',
    'Lighting2',
    'Lighting7',
    'MALLAT',
    'MedicalImages',
    'MoteStrain',
    'NonInvasiveFatalECG_Thorax1',
    'NonInvasiveFatalECG_Thorax2',
    'OliveOil',
    'OSULeaf',
    'Plane',
    'SonyAIBORobotSurface',
    'SonyAIBORobotSurfaceII',
    'StarLightCurves',
    'SwedishLeaf',
    'Symbols',
    'synthetic_control',
    'Trace',
    'TwoLeadECG',
    'Two_Patterns',
    'uWaveGestureLibrary_X',
    'uWaveGestureLibrary_Y',
    'uWaveGestureLibrary_Z',
    'wafer',
    'WordsSynonyms',
    'yoga',
]
'''
names = ['Gun_Point']

num_divisions = 3

classifiers = [
    neighbors.KNeighborsClassifier(n_neighbors=1),
    linear_model.LogisticRegression(),
]

classifier_names = [
    'k-Nearest Neighbors (k=1)',
    'Logistic Regression',
]

grid_params = {
    'time_scale': [1],
    'window_size': [80],  # [8, 16, 24, 32, 40],
    'lambda1': [0.1],  # [0.1, 0.2, 0.5, 1.0],
    'num_atoms_ratio': [2],  # num_atoms / window_size
    'interval_ratio': [1, 1/2, 1/4, 1/8],  # interval / window_size
    'extract_mode': ['max', 'sum', 'mean', 'hist'],
    'classifier': classifiers,
    'classifier_name': classifier_names,
}

for name in names:
    # grid search
    gs = tsc.GridSearchUCR(name, num_divisions)
    best_param = gs.fit(grid_params)
    gs.predict([best_param])
    '''
    # plot dependency of interval
    target = 'interval'
    grid = [1, 2, 4, 8, 16, 32, 64]
    params = tsc.generate_params(target, grid, best_param)
    compression_rates, RMSEs, error_rates = gs.predict(
            params, print_predict=False)
    tsc.plot_dependency(
            target, grid, compression_rates, RMSEs, error_rates, logx=True)
    # plot dependency of lambda1
    target = 'lambda1'
    grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    params = tsc.generate_params(target, grid, best_param)
    compression_rates, RMSEs, error_rates = gs.predict(
            params, print_predict=False)
    tsc.plot_dependency(target, grid, compression_rates, RMSEs, error_rates)
    '''


# plot dependencies around best parameters
# (1) select by majority vote of window_size
# (2) select smallest lambda1
# (3) select smallest interval
# (4) select extract_mode in order of ('max', 'sum', 'hist')
for name in names:
    # set fixed best_param
    best_param = {
        'classifier': linear_model.LogisticRegression(),
        'classifier_name': 'Logistic Regression',
        'time_scale': 1,
        'window_size': 24,
        'extract_mode': 'max',
        'interval': 24,
        'lambda1': 0.1,
        'num_atoms': 48,
    }
    # set best_param for Logistic Regression
    '''
    best_param = {
        'classifier': linear_model.LogisticRegression(),
        'classifier_name': 'Logistic Regression',
        'time_scale': 1,
    }
    if name is None:
        continue
    elif name == 'Adiac':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 3
        best_param['lambda1'] = 0.1
    elif name == 'Beef':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.1
    elif name == 'Car':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.2
    elif name == 'CBF':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 5
        best_param['lambda1'] = 0.2
    elif name == 'ChlorineConcentration':
        best_param['window_size'] = 8
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 1
        best_param['lambda1'] = 1.0
    elif name == 'CinC_ECG_torso':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 5
        best_param['lambda1'] = 0.1
    elif name == 'Coffee':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.1
    elif name == 'Cricket_X':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 1.0
    elif name == 'Cricket_Y':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 1.0
    elif name == 'Cricket_Z':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 10
        best_param['lambda1'] = 1.0
    elif name == 'DiatomSizeReduction':
        best_param['window_size'] = 16
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 2
        best_param['lambda1'] = 1.0
    elif name == 'ECGFiveDays':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.1
    elif name == 'FaceAll':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 1.0
    elif name == 'FaceFour':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 3
        best_param['lambda1'] = 0.5
    elif name == 'FacesUCR':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 1.0
    elif name == '50words':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.5
    elif name == 'FISH':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.1
    elif name == 'Gun_Point':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 3
        best_param['lambda1'] = 0.1
    elif name == 'Haptics':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.5
    elif name == 'InlineSkate':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.2
    elif name == 'ItalyPowerDemand':
        best_param['window_size'] = 8
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 1
        best_param['lambda1'] = 0.1
    elif name == 'Lighting2':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 40
        best_param['lambda1'] = 1.0
    elif name == 'Lighting7':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 4
        best_param['lambda1'] = 1.0
    elif name == 'MALLAT':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.1
    elif name == 'MedicalImages':
        best_param['window_size'] = 16
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 2
        best_param['lambda1'] = 1.0
    elif name == 'MoteStrain':
        best_param['window_size'] = 16
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 2
        best_param['lambda1'] = 0.2
    elif name == 'NonInvasiveFatalECG_Thorax1':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.5
    elif name == 'NonInvasiveFatalECG_Thorax2':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 4
        best_param['lambda1'] = 0.5
    elif name == 'OliveOil':
        best_param['window_size'] = 16
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 2
        best_param['lambda1'] = 0.2
    elif name == 'OSULeaf':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 3
        best_param['lambda1'] = 0.1
    elif name == 'Plane':
        best_param['window_size'] = 16
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 2
        best_param['lambda1'] = 0.1
    elif name == 'SonyAIBORobotSurface':
        best_param['window_size'] = 8
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 1
        best_param['lambda1'] = 0.1
    elif name == 'SonyAIBORobotSurfaceII':
        best_param['window_size'] = 16
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 2
        best_param['lambda1'] = 0.1
    elif name == 'StarLightCurves':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.1
    elif name == 'SwedishLeaf':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 5
        best_param['lambda1'] = 0.2
    elif name == 'Symbols':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.1
    elif name == 'synthetic_control':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 8
        best_param['lambda1'] = 1.0
    elif name == 'Trace':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 4
        best_param['lambda1'] = 0.1
    elif name == 'Two_Patterns':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 4
        best_param['lambda1'] = 1.0
    elif name == 'TwoLeadECG':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 4
        best_param['lambda1'] = 0.1
    elif name == 'uWaveGestureLibrary_X':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.1
    elif name == 'uWaveGestureLibrary_Y':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 1.0
    elif name == 'uWaveGestureLibrary_Z':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.2
    elif name == 'wafer':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 1.0
    elif name == 'WordsSynonyms':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 10
        best_param['lambda1'] = 1.0
    elif name == 'yoga':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.2
    else:
        continue
    best_param['num_atoms'] = 2*best_param['window_size']
    '''
    # set best_param for 1-Nearest Neighbors
    '''
    best_param = {
        'classifier': neighbors.KNeighborsClassifier(n_neighbors=1),
        'classifier_name': 'k-Nearest Neighbors (k=1)',
        'time_scale': 1,
    }
    if name is None:
        continue
    elif name == 'Adiac':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.1
    elif name == 'Beef':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 3
        best_param['lambda1'] = 1.0
    elif name == 'Car':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.1
    elif name == 'CBF':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 5
        best_param['lambda1'] = 0.5
    elif name == 'ChlorineConcentration':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.1
    elif name == 'CinC_ECG_torso':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 3
        best_param['lambda1'] = 1.0
    elif name == 'Coffee':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.1
    elif name == 'Cricket_X':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.5
    elif name == 'Cricket_Y':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 5
        best_param['lambda1'] = 1.0
    elif name == 'Cricket_Z':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 1.0
    elif name == 'DiatomSizeReduction':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 6
        best_param['lambda1'] = 0.2
    elif name == 'ECGFiveDays':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 16
        best_param['lambda1'] = 0.1
    elif name == 'FaceAll':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.5
    elif name == 'FaceFour':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 5
        best_param['lambda1'] = 0.2
    elif name == 'FacesUCR':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.5
    elif name == '50words':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.2
    elif name == 'FISH':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.1
    elif name == 'Gun_Point':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 5
        best_param['lambda1'] = 0.2
    elif name == 'Haptics':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 40
        best_param['lambda1'] = 0.2
    elif name == 'InlineSkate':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 4
        best_param['lambda1'] = 0.1
    elif name == 'ItalyPowerDemand':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 3
        best_param['lambda1'] = 0.5
    elif name == 'Lighting2':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 24
        best_param['lambda1'] = 0.5
    elif name == 'Lighting7':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.2
    elif name == 'MALLAT':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 5
        best_param['lambda1'] = 1.0
    elif name == 'MedicalImages':
        best_param['window_size'] = 16
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 2
        best_param['lambda1'] = 0.1
    elif name == 'MoteStrain':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 12
        best_param['lambda1'] = 0.1
    elif name == 'NonInvasiveFatalECG_Thorax1':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 5
        best_param['lambda1'] = 0.2
    elif name == 'NonInvasiveFatalECG_Thorax2':
        best_param['window_size'] = 16
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 2
        best_param['lambda1'] = 0.5
    elif name == 'OliveOil':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 3
        best_param['lambda1'] = 0.2
    elif name == 'OSULeaf':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 3
        best_param['lambda1'] = 0.5
    elif name == 'Plane':
        best_param['window_size'] = 16
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 2
        best_param['lambda1'] = 0.1
    elif name == 'SonyAIBORobotSurface':
        best_param['window_size'] = 8
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 2
        best_param['lambda1'] = 0.1
    elif name == 'SonyAIBORobotSurfaceII':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 6
        best_param['lambda1'] = 1.0
    elif name == 'StarLightCurves':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.5
    elif name == 'SwedishLeaf':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 3
        best_param['lambda1'] = 0.1
    elif name == 'Symbols':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 4
        best_param['lambda1'] = 0.1
    elif name == 'synthetic_control':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 3
        best_param['lambda1'] = 1.0
    elif name == 'Trace':
        best_param['window_size'] = 24
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 3
        best_param['lambda1'] = 0.1
    elif name == 'Two_Patterns':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 5
        best_param['lambda1'] = 1.0
    elif name == 'TwoLeadECG':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'max'
        best_param['interval'] = 5
        best_param['lambda1'] = 0.1
    elif name == 'uWaveGestureLibrary_X':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.1
    elif name == 'uWaveGestureLibrary_Y':
        best_param['window_size'] = 160
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.1
    elif name == 'uWaveGestureLibrary_Z':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.5
    elif name == 'wafer':
        best_param['window_size'] = 40
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 20
        best_param['lambda1'] = 0.2
    elif name == 'WordsSynonyms':
        best_param['window_size'] = 80
        best_param['extract_mode'] = 'hist'
        best_param['interval'] = 10
        best_param['lambda1'] = 0.5
    elif name == 'yoga':
        best_param['window_size'] = 32
        best_param['extract_mode'] = 'sum'
        best_param['interval'] = 4
        best_param['lambda1'] = 1.0
    else:
        continue
    best_param['num_atoms'] = 2*best_param['window_size']
    '''
    # read time series data
    '''
    gs = tsc.GridSearchUCR(name)
    '''
    # predict
    '''
    target = 'extract_mode'
#    grid = ['max', 'sum', 'hist']
#    params = tsc.generate_params(target, grid, best_param)
#    compression_rates, RMSEs, error_rates = gs.predict(
#            params, file_name='0LR_result.csv')
    compression_rates, RMSEs, error_rates = gs.predict(
            [best_param], file_name='aaa.csv')
    '''
