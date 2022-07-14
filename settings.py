import numpy as np
import os
import time
from mypaths import ROOT, CHECKPT_ROOT

FEATURE_PATHS = {'INDU': 'SAT/SAT_Competition_RACE_INDU-feat.csv',
                 'HAND': 'SAT/SAT_Competition_RACE_HAND-feat.csv',
                 'RAND': 'SAT/SAT_Competition_RACE_RAND-feat.csv',
                 'IBM': 'SAT/IBM-ALL-feat.csv',
                 'SWV': 'SAT/SWV-feat.csv',
                 'GCP': 'SAT/SWGCP-ALL-feat.csv',
                 'QCP': 'SAT/QCP-ALL-feat.csv',
                 'RUE': 'TSP/PORTGEN-feat.csv',
                 'RCE': 'TSP/PORTCGEN-feat.csv',
                 'TSPLIB': 'TSP/TSPLIB_feat_all.csv',
                 'BIGMIX': 'MIP/BIGMIX-train_test-features-withfilename.csv',
                 'CORLAT': 'MIP/CORLAT-train_test-features-withfilename.csv',
                 'CORLAT-normalized': 'MIP/CORLAT-train_test-features-withfilename.csv',
                 'CRR': 'MIP/CORLAT-REG-RCW-features.csv',
                 'CR': 'MIP/CORLAT-REG-features.csv',
                 'RCW': 'MIP/RCW-train_test-features-withfilename.csv',
                 'REG': 'MIP/REG-train_test-features-withfilename.csv',
                 'GCP-graph': 'GCP/GCP_graph_features.csv',
                 'GCP-CNF': 'GCP/GCP_cnf_features.csv'}

class OptimizationSettings():
    def __init__(self, learning_rate=0.0001, dropout=0.5, sampling_method=None,
                 batch_size=32, num_epochs=100000, batches_per_epoch=5, size_noise=0.,
                 filter_features=True):
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.sampling_method = sampling_method
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.batches_per_epoch = batches_per_epoch
        self.size_noise = size_noise
        self.filter_features = filter_features

        def __str__(self):
            return '_'.join(['lr_' + self.learning_rate,
                             'dropout_' + self.dropout,
                             'sampling_' + self.sampling_method,
                             'bs_' + self.batch_size,
                             'sizenoise_' + self.size_noise])

class ArchitectureSettings():
    def __init__(self, use_resnet=False, append_handcrafted=False,
                 exchangeable_layers=5, feedforward_layers=1, units=100,
                 use_batch_norm=False, output_dimension=False, zero_exch=False,
                 poly_pool=False, deep_set_pooling=False, size_field_length=18,
                 size_base=2, bins=1, predict_feat=False, permute_dense=True):
        self.use_resnet = use_resnet
        self.append_handcrafted = append_handcrafted
        self.exchangeable_layers = exchangeable_layers
        self.feedforward_layers = feedforward_layers
        self.units = units
        self.use_batch_norm = use_batch_norm
        self.output_dimension = output_dimension
        self.zero_exch = zero_exch
        self.append_size_info = size_base is not None
        self.poly_pool = poly_pool
        self.deep_set_pooling = deep_set_pooling
        self.size_field_length = size_field_length
        self.size_base = size_base
        self.bins = bins
        self.predict_feat = predict_feat
        self.permute_dense = permute_dense

    def __str__(self):
        return '_'.join(['exch_l_' + str(self.exchangeable_layers),
                         'ff_l_' + str(self.feedforward_layers),
                         'units_' + str(self.units),
                         'size_' + str(self.append_size_info)])

class Settings():
    def __init__(self, args):
        # Main parameters
        self.experiments_root = ROOT
        self.dataset = args.dataset
        self.dataset_test = args.dataset_test

        if not args.project_name:
            self.project_name = 'Dataset:%s' % (self.dataset)
        else:
            self.project_name = args.project_name

        self.add_random_forest = args.add_random_forest
        self.mode = args.mode
        self.experiment_name = args.experiment_name
        if args.max_nonzeros == -1:
            self.max_nonzeros = np.inf
        else:
            self.max_nonzeros = args.max_nonzeros

        # Output
        self.time = str(float(time.time()))
        self.save_checkpts_logs = args.save_checkpts_logs
        self.save_predictions = args.save_predictions
        self.verbose = args.verbose
        self.timestamp = args.timestamp
        self.delegate = args.delegate
        self.load_path_just_features = args.load_path_just_features
        self.validation_frequency = args.validation_frequency
        self.no_comet = args.no_comet
        self.comet_username = args.comet_username
        self.comet_api_key = args.comet_api_key
        self.update_frequency = args.update_frequency

        if self.verbose:
            print('Root directory:', self.experiments_root)

        if not self.dataset:
            data_directory = os.path.join(self.experiments_root, 'data')
            available_datasets = [os.path.basename(
                x[0]) for x in os.walk(data_directory)][1:]
            raise Exception('Dataset not specified! Choose a dataset to use: %s\n'
                            'Available datasets: %s' % (data_directory, available_datasets))
        else:
            self.data_directory = os.path.join(self.experiments_root, 'data', self.dataset)
            if not os.path.exists(self.data_directory):
                raise Exception('Invalid dataset %s specified' % self.data_directory)

        if self.dataset_test:
            self.test_data_directory = os.path.join(self.experiments_root, 'data', self.dataset_test)
            if not os.path.exists(self.test_data_directory):
                raise Exception('Invalid test dataset %s specified' % self.test_data_directory)
        else:
            self.test_data_directory = None

        self.feature_runtime_directory = os.path.join(self.experiments_root,
                'aij_feature_runtime_data/')
        self.checkpt_root = os.path.join(self.experiments_root,
                'checkpts') if CHECKPT_ROOT is None else CHECKPT_ROOT
        self.logging_directory = os.path.join(self.experiments_root,'logs', self.dataset)
        self.checkpt_directory = os.path.join(self.checkpt_root, self.dataset)
        self.predictions_directory = os.path.join(self.experiments_root,
                'predictions', self.dataset)

        if not args.load_path or os.path.exists(args.load_path):
            self.load_path = args.load_path
        else:
            self.load_path = os.path.join(self.checkpt_directory, args.load_path)
            if not os.path.exists(self.load_path):
                raise Exception('Cannot load model from checkpoint path %s or %s' % (
                    args.load_path, self.load_path))

        if not args.load_path_just_features or os.path.exists(args.load_path_just_features):
            self.load_path_just_features = args.load_path_just_features
        else:
            self.load_path_just_features = os.path.join(self.checkpt_directory,
                    args.load_path_just_features)
            if not os.path.exists(self.load_path_just_features):
                raise Exception('Cannot load model from checkpoint path %s or %s' % (
                    args.load_path_just_features, self.load_path_just_features))

        # Check directory structure
        if self.save_checkpts_logs:
            if not os.path.exists(self.checkpt_directory):
                os.mkdir(self.checkpt_directory)
            if not os.path.exists(self.logging_directory):
                os.mkdir(self.logging_directory)
        if self.save_predictions:
            if not os.path.exists(self.predictions_directory):
                os.mkdir(self.predictions_directory )

        # Hardware
        self.device = 'cpu' if args.no_cuda else 'cuda'
        if self.verbose:
            print('Device: %s' % self.device)

        # Architecture
        self.arch = ArchitectureSettings(use_resnet=args.resnet,
                append_handcrafted=args.append_handcrafted,
                exchangeable_layers=args.exchangeable_layers,
                feedforward_layers=args.feedforward_layers,
                units=args.units, use_batch_norm=args.batch_norm,
                output_dimension=args.output_dimension, zero_exch=args.zero_exch,
                poly_pool=args.poly_pool, deep_set_pooling=args.deep_set_pooling,
                size_field_length=args.size_field_length,
                size_base=args.size_base, bins=args.bins, predict_feat=args.predict_feat)

        # Optimization
        self.opt = OptimizationSettings(learning_rate=args.lrate,
                dropout=args.dropout, sampling_method=args.sampling_method,
                batch_size=args.batch_size, num_epochs=args.num_epochs,
                batches_per_epoch=args.batches_per_epoch,
                size_noise=args.size_noise, filter_features=args.filter_features)

        # Target
        self.predict_runtime = args.predict_runtime
        self.predict_sat = args.predict_sat
        self.predict_assigns = args.predict_ass

        if not (self.predict_runtime or self.predict_sat or self.predict_assigns):
            raise Exception('Must predict runtime, SAT, or assignments')
        elif self.predict_runtime and self.predict_sat:
            raise Exception('Can only predict one of SAT or runtime')

        self.predict_residual = args.predict_residual
        self.log_transform = args.log_transform
        self.discretize_target = (args.bins != 1)

        #if args.mode == 'nn_handcrafted':
        #    self.single_instance_batching = False
        #else:
        self.single_instance_batching = True

        self.experiment_id = self.mode + '_' + str(self.arch) + '_' + self.experiment_name

        self.predictions_path = os.path.join(self.predictions_directory,
                self.experiment_id + (self.time if self.timestamp else '') + '.csv')

        self.performance_logger_path = os.path.join(self.logging_directory,
                self.experiment_id + (self.time if self.timestamp else '') + 'perf.csv')

        # Write experiment name to file
        if self.save_checkpts_logs and self.verbose:
            print('Begin logging to %s' % (self.performance_logger_path))
            print(self.experiment_id, file=open(self.performance_logger_path, 'a'))
