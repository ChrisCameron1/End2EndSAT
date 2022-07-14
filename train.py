import argparse
import glob
import numpy as np
import os
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Library for monitoring online; for monitoring, go to https://www.comet.ml/
from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils import data
import sklearn.ensemble as ske

from exchangable_tensor.sp_layers import (SparsePool, SparseExchangeable,
    SparseSequential, prepare_global_index)
from settings import Settings
from mypaths import COMET
from nn_utils import (CustomBatchNorm, HybridNetwork, ResidualNetwork, SimpleCNN, SparsePoolBoth)
from datasets import CombinatorialOptimizationDataset
from utils import (subsample, weights_init, detach, apply_discr, collate_fn,
    get_feature_normalization_params, get_size_features, bin_setup, 
    rmse_error, get_partition_file)

def get_args():
    parser = argparse.ArgumentParser()

    # High-level parameters
    parser.add_argument('-rf', dest='add_random_forest', action='store_true',
            help='Add random forest evaluation')
    parser.add_argument('-m', dest='mode', help='Training mode to use', default='nn_raw',
            choices=['rf', 'nn_handcrafted', 'nn_raw', 'nn_residual'])
    parser.add_argument('-s', dest='seed', type=int, default=123456,
            help='Random seed for reproducibility')
    parser.add_argument('-l', dest='load_path', default='',
            help='File where Torch model should be loaded from')
    parser.add_argument('-lf', dest='load_path_just_features', type=str, default=None,
            help='File where Torch model for hand-crafted features should '
                 'be loaded from; if set, network will predict residuals')
    parser.add_argument('-d', dest='delegate', action='store_true',
            help='Read instance partitions from file')
    parser.add_argument('-ne', dest='experiment_name', default='None',
            help='Name of experiment for local files')
    parser.add_argument('-nz', dest='max_nonzeros', default=-1, type=int,
            help='Maximum number of nonzeroes (clause-variable pairs); no limit by default')
    parser.add_argument('-cn', dest='no_comet', action='store_true', 
            help='Turn off job monitoring')
    parser.add_argument('-cu', dest='comet_username', type=str, default=COMET['un'],
            help='Username for Comet.ml job monitoring')
    parser.add_argument('-ck', dest='comet_api_key', type=str, default=COMET['key'],
            help='API key for Comet.ml')
    parser.add_argument('-cp', dest='project_name', type=str, default='runtime-prediction',
            help='Project name for grouping experiments in Comet.ml')

    # Prediction target settings
    parser.add_argument('-pt', dest='predict_runtime', action='store_true', 
            help='Predict runtime')
    parser.add_argument('-ps', dest='predict_sat', action='store_true', 
            help='Predict SAT/UNSAT')
    parser.add_argument('-pa', dest='predict_ass', action='store_true',
            help='Predict satisfying assignments')
    parser.add_argument('-lg', dest='log_transform', action='store_true',
            help='Log-transform prediction target')
    parser.add_argument('-pr', dest='predict_residual', action='store_true',
            help='Predict residual between target and model from just hand-crafted features')
    parser.add_argument('-pf', dest='predict_feat', action='store_true',
            help='Whether to predict features')

    # Output
    parser.add_argument('-vf', dest='validation_frequency', type=int, default=10,
            help='Number of epochs between validation/testing (default 10)')
    parser.add_argument('-i', dest='save_checkpts_logs', action='store_false',
            help='Turn off checkpoints and log files')
    parser.add_argument('-v', dest='verbose', action='store_true',
            help='Turn on console output')
    parser.add_argument('-p', dest='save_predictions', action='store_true',
            help='Turn on saving of prediction/target idatapoints for plotting')
    parser.add_argument('-t', dest='timestamp', action='store_true',
            help='Append execution datetime to files (useful for batch runs)')
    parser.add_argument('-oe', dest='offline_eval', action='store_true',
            help='Do not evaluate - just store checkpoints')

    # Optimization parameters
    parser.add_argument('-r', dest='lrate', default=0.0001, type=float,
            help='Learning rate for network training (default 0.0001)')
    parser.add_argument('-dp', dest='dropout', type=float, default=0,
            help='Amount of dropout to add at each layer (default 0)')
    parser.add_argument('-sm', dest='sampling_method', default='uniform',
            choices=['uniform', 'neighbour'],
            help='Subsampling method for oversized instances (default [uniform] at random, '
                 'nearest [neighbour]s not supported)')
    parser.add_argument('-bs', dest='batch_size', type=int, default=32,
            help='Batch size (default 32)')
    parser.add_argument('-ep', dest='num_epochs', type=int, default=10000000,
            help='Number of epochs to run for (default 10000000)')
    parser.add_argument('-bep', dest='batches_per_epoch', type=int, default=5,
            help='Number of batches to process per epoch (default 5)')
    parser.add_argument('-sn', dest='size_noise', type=float, default=0.,
            help='Amount of multiplicitive noise to add to size features (default none)')
    parser.add_argument('-ff', dest='filter_features', action='store_true',
                        help='Filter out probing and timing features')

    # Hardware
    parser.add_argument('-nc', dest='no_cuda', action='store_true', help='Run on CPU only')

    # Architecture parameters
    parser.add_argument('-rn', dest='resnet', action='store_true',
            help='Use resnet architecture')
    parser.add_argument('-fh', dest='append_handcrafted', action='store_true',
            help='Append hand-crafted features to input')
    parser.add_argument('-le', dest='exchangeable_layers', type=int, default=5,
            help='Number of exchangeable layers to append (default 5)')
    parser.add_argument('-lff', dest='feedforward_layers', type=int, default=3,
            help='Number of feedforward layers to append after '
                 'exchangeable model (default 3)')
    parser.add_argument('-u', dest='units', type=int, default=100,
            help='Number of units (channels) per exchangeable layer (default 100)')
    parser.add_argument('-od', dest='output_dimension', type=int, default=64,
            help='Output dimension at last layer of exchangeable model (default 64)')
    parser.add_argument('-ez', dest='zero_exch', action='store_true',
            help='Replace exchangeable output with 0s (for debugging purposes)')
    parser.add_argument('-pp', dest='poly_pool', type=int, default=1,
            help='Pool with higher order moments (default 1; no higher order polynomials)')
    parser.add_argument('-pm', dest='deep_set_pooling', action='store_true',
            help='Use max-pooling instead of mean-pooling')
    parser.add_argument('-bn', dest='batch_norm', action='store_true',
            help='Perform batch normalization at every exchangeable layer')
    parser.add_argument('-uf', dest='update_frequency', type=int, default=2,
            help='Frequency at which to update batch norm parameters '
                 '(default every [2] epochs)')
    parser.add_argument('-sl', dest='size_field_length', type=int, default=None,
            help='Length of representation for encoding size features')
    parser.add_argument('-sb', dest='size_base', type=int, default=None,
            help='Length of representation for encoding size features')
    parser.add_argument('-b', dest='bins', default=1, type=int,
            help='Histogram bins for discretization of runtime (default 1; none)')
    parser.add_argument('-pd', dest='permute_dense', action='store_true',
            help='Permute matrix for dense representation')

    parser.add_argument('-td', dest='dataset_test', type=str, default=None,
            help='Relative path to subdirectory for .npz files to test on')
    parser.add_argument('dataset', type=str,
            help='Relative path to subdirectory for .npz files to train on')

    return parser.parse_args()

def setup_parameterized_model(settings, sample_data_point, feature_dim=None):
    pool_both = None

    if settings.mode == 'nn_handcrafted':
        if settings.predict_assigns:
            raise Exception('Assignment prediction not supported for handcrafted-only')

        units = settings.arch.units
        num_features = len(sample_data_point['features'])

        if settings.opt.filter_features:
            deleted_length = len(
                    CombinatorialOptimizationDataset.PROBING_FEATURES) + len(
                    CombinatorialOptimizationDataset.TIMING_FEATURES)
            num_features -= deleted_length

        if settings.verbose:
            print('Number of input features: %d' % num_features)

        if settings.arch.use_batch_norm:
            raise Exception('Batch norm not implemented for handcrafted-only network')

        inputs = []
        inputs.append(nn.Linear(in_features=num_features, out_features=units))
        inputs.append(nn.LeakyReLU())

        for _ in range(settings.arch.feedforward_layers):
            inputs.append(nn.Linear(in_features=units, out_features=units))
            inputs.append(nn.LeakyReLU())
            if settings.opt.dropout > 0:
                inputs.append(nn.Dropout(p=settings.opt.dropout))

        inputs.append(nn.Linear(in_features=units, out_features=1))

        if settings.predict_sat:
            inputs.append(nn.Sigmoid())

        mod = nn.Sequential(*inputs)
        mod.apply(weights_init)
        return mod, None, None, None, None

    elif settings.mode == 'cnn':
        if settings.predict_assigns:
            raise Exception('Assignment prediction not supported for CNN')

        mod = SimpleCNN(sample_data_point, settings)
        mod.apply(weights_init)

        return mod, None, None, None, None

    if sample_data_point['size'] > settings.max_nonzeros:
        idx = subsample(sample_data_point.index, size=settings.max_nonzeros,
                method=settings.sampling_method)
    else:
        idx = np.arange(sample_data_point['size'])

    index = prepare_global_index(sample_data_point['indices'][idx, ...])
    index = torch.from_numpy(np.array(index, dtype='int')).to(settings.device)

    units = settings.arch.units
    layers = settings.arch.exchangeable_layers
    norm = settings.arch.use_batch_norm
    batch_norm = None

    if settings.arch.size_base is None:
        input_width = sample_data_point['values'].shape[1]
    else:
        input_width = sample_data_point['values'].shape[1] + (
                index.shape[1]) * settings.arch.size_field_length
    if settings.verbose:
        print('Input width: %d' % input_width)

    if settings.arch.use_resnet:
        if settings.verbose:
            print('Using resnet architecture with %d layers' % (layers))
        inputs = [index, SparseExchangeable(input_width, units, index)] + [
                  ResidualNetwork(units, index, norm=norm)] * (layers // 2) + [
                  SparseExchangeable(units, settings.arch.output_dimension, index),
                  SparsePool(index, settings.arch.output_dimension,
                        axis=1, keep_dims=False)]
    else:
        if settings.verbose:
            print('Using regular architecture')
        inputs = [index, SparseExchangeable(input_width, units, index), nn.LeakyReLU()]
        for _ in range(layers):
            if settings.arch.deep_set_pooling:
                pool_fns = [nn.Sequential(nn.Linear(units, 128), nn.ReLU(),
                    nn.Linear(128, units)) for _ in range(2)]
            else:
                pool_fns = None

            inputs.append(SparseExchangeable(units, units, index,
                poly_pool=settings.arch.poly_pool, deepset=pool_fns))

        inputs.append(SparseExchangeable(units, settings.arch.output_dimension, index))
        inputs.append(SparsePool(index, settings.arch.output_dimension,
            axis=1, keep_dims=False))

    exchangeable_model = SparseSequential(*inputs)

    batch_norm = None
    if settings.arch.append_handcrafted:
        if settings.arch.use_batch_norm:
            raise Exception('Batch norm not implemented for appended handcrafted features')

        model = None
        if settings.predict_runtime or settings.predict_sat:
            model = HybridNetwork(exchangeable_model=exchangeable_model,
                                  exchangeable_model_dim=settings.arch.output_dimension,
                                  feature_dim=feature_dim, units=settings.arch.units, 
                                  out_dim=settings.arch.bins,
                                  feedforward_layers=settings.arch.feedforward_layers,
                                  dropout=settings.opt.dropout,
                                  append_size_info=settings.arch.append_size_info,
                                  predict_features=settings.arch.predict_feat,
                                  zero_exch=settings.arch.zero_exch,
                                  predict_sat=settings.predict_sat, predict_assigns=False, 
                                  k=settings.arch.size_field_length,
                                  base=settings.arch.size_base, noise=settings.opt.size_noise)
            model.apply(weights_init)

        assigns_model = None
        if settings.predict_assigns:
            assigns_model = HybridNetwork(exchangeable_model=exchangeable_model,
                                  exchangeable_model_dim=settings.arch.output_dimension,
                                  feature_dim=feature_dim, out_dim=settings.arch.bins,
                                  feedforward_layers=settings.arch.feedforward_layers,
                                  dropout=settings.opt.dropout,
                                  append_size_info=settings.arch.append_size_info,
                                  predict_features=settings.arch.predict_feat,
                                  zero_exch=settings.arch.zero_exch,
                                  predict_sat=False, predict_assigns=True, 
                                  k=settings.arch.size_field_length,
                                  base=settings.arch.size_base, noise=settings.opt.size_noise)
            assigns_model.apply(weights_init)
    else:
        pool_both = SparsePoolBoth(
                index=None if not settings.arch.append_size_info else index,
                base=settings.arch.size_base,
                k=settings.arch.size_field_length,
                noise=settings.opt.size_noise)
        if settings.predict_runtime or settings.predict_sat:
            inputs = [exchangeable_model, pool_both]
        if settings.predict_assigns:
            assigns_inputs = [exchangeable_model]

        out_dim_imp = settings.arch.output_dimension
        if settings.arch.append_size_info:
            out_dim_imp += settings.arch.size_field_length * 2

        if settings.predict_runtime or settings.predict_sat:
            inputs.append(nn.LayerNorm(out_dim_imp))
            inputs.append(nn.Linear(in_features=out_dim_imp,
                out_features=settings.arch.output_dimension))
        if settings.predict_assigns:
            assigns_inputs.append(nn.LayerNorm(out_dim_imp))
            assigns_inputs.append(nn.Linear(in_features=out_dim_imp,
                out_features=settings.arch.output_dimension))

        if settings.arch.use_batch_norm:
            batch_norm = CustomBatchNorm(settings.arch.output_dimension,
                    device=settings.device)
            if settings.predict_runtime or settings.predict_sat:
                inputs.append(batch_norm)
            if settings.predict_assigns:
                assigns_inputs.append(batch_norm)
        else:
            batch_norm = None

        for _ in range(settings.arch.feedforward_layers):
            if settings.predict_runtime or settings.predict_sat:
                inputs.append(nn.Linear(in_features=settings.arch.output_dimension,
                    out_features=settings.arch.output_dimension))
                inputs.append(nn.LeakyReLU())
                if settings.opt.dropout > 0: 
                    inputs.append(nn.Dropout(p=settings.opt.dropout))

            if settings.predict_assigns:
                assigns_inputs.append(nn.Linear(in_features=settings.arch.output_dimension,
                    out_features=settings.arch.output_dimension))
                assigns_inputs.append(nn.LeakyReLU())
                if settings.opt.dropout > 0: 
                    assigns_inputs.append(nn.Dropout(p=settings.opt.dropout))

        if settings.predict_runtime or settings.predict_sat:
            inputs.append(nn.Linear(in_features=settings.arch.output_dimension,
                out_features=settings.arch.bins))

        if settings.predict_sat:
            inputs.append(nn.Sigmoid())
        if settings.predict_assigns:
            assigns_inputs.append(nn.Sigmoid())

        model = None
        if settings.predict_runtime or settings.predict_sat:
            model = nn.Sequential(*inputs)
            model.apply(weights_init)

        assigns_model = None
        if settings.predict_assigns:
            assigns_model = nn.Sequential(*assigns_inputs)
            assigns_model.apply(weights_init)

    return model, assigns_model, exchangeable_model, pool_both, batch_norm

def get_instance_partitions(files, test_files=[], partition_from_file=True,
        data_path='./', test_data_path=None):
    train_file = get_partition_file(os.path.join(data_path,
         'train*.txt'))
    valid_file = get_partition_file(os.path.join(data_path,
        'valid*.txt'))
    if not test_data_path:
        test_file = get_partition_file(os.path.join(data_path,
            'test*.txt'))
    else:
        test_file = get_partition_file(os.path.join(test_data_path,
            'test*.txt'))

    if not partition_from_file or (not os.path.exists(train_file)
        ) or (not os.path.exists(valid_file) or (not os.path.exists(test_file))):
        if test_data_path:
            np.random.shuffle(files)
            train_bound = int(0.9 * len(files))
            train_files = files[:train_bound]
            validation_files = files[train_bound:]
            test_files = test_files
        else:
            np.random.shuffle(files)
            train_bound = int(0.7 * len(files))
            validation_bound = int(0.8 * len(files))
            train_files = files[:train_bound]
            validation_files = files[train_bound:validation_bound]
            test_files = files[validation_bound:]
        return train_files, validation_files, test_files

    else:
        print('Data path: %s' % data_path)
        with open(train_file, 'r') as f:
            train_files = [os.path.join(data_path,
                os.path.basename(l.strip())) for l in f.readlines()]
        with open(valid_file, 'r') as f:
            validation_files = [os.path.join(data_path,
                os.path.basename(l.strip())) for l in f.readlines()]
        with open(test_file, 'r') as f:
            if test_data_path:
                test_files = [os.path.join(test_data_path,
                    os.path.basename(l.strip())) for l in f.readlines()]
            else:
                test_files = [os.path.join(data_path,
                    os.path.basename(l.strip())) for l in f.readlines()]

        if len(train_files) == 0:
            raise Exception('Empty training set for ' + data_path)
        if len(validation_files) == 0:
            raise Exception('Empty validation set for ' + data_path)
        if len(test_files) == 0:
            raise Exception('Empty testing set for ' + data_path)

        all_files = train_files + validation_files + test_files
        if not set(all_files) == set(files):
            print('WARNING: splits (num: %d) do not contain all files (num: %d)' % (
                len(set(all_files)), len(files)))

        return train_files, validation_files, test_files

def write_output(epoch=None, train=None, validation=None, test=None,
        predict_assigns=False, verbose=True, write=True, log_path=None):
    if verbose:
        if predict_assigns:
            print('VALIDATION. Epoch: {ep}. Batch loss: {v_loss}. '
                  'Reconstructed: {h_loss_s}. Assigns accuracy: {av_loss}.'.format(
                    ep=epoch, v_loss=validation[0], h_loss_s=validation[1], 
                    av_loss=validation[2]))
            print('TESTING. Epoch: {ep}. Batch loss: {t_loss}. '
                  'Reconstructed: {e_loss_s}. Assigns accuracy: {at_loss}.'.format(
                    ep=epoch, t_loss=test[0], e_loss_s=test[1], 
                    at_loss=test[2]))
        else:
            print('VALIDATION. Epoch: {ep}. Batch loss: {v_loss}. '
                  'Reconstructed: {h_loss_s}.'.format(
                    ep=epoch, v_loss=validation[0], h_loss_s=validation[1]))
            print('TESTING. Epoch: {ep}. Batch loss: {t_loss}. '
                  'Reconstructed: {e_loss_s}.'.format(
                    ep=epoch, t_loss=test[0], e_loss_s=test[1]))

    if write:
        write_string = '{ep},{v_loss},{av_loss},{t_loss},{at_loss},{p_loss}'.format(ep=epoch,
                v_loss=validation[0], av_loss=validation[2],
                t_loss=test[0], at_loss=test[2], p_loss=train[0])

        print(write_string, file=open(log_path, 'a'))

def set_generator_descriptor(generator, mode='train', batch_size=None, size=None, i=None):
    if mode == 'train':
        generator.set_description('TRAINING. Batch sample: %d/%d (instance size: %d): ' % (
            i, batch_size, size))
        generator.refresh()
    elif mode == 'validate':
        generator.set_description('VALIDATION. Batch sample: %d/%d (instance size: %d): ' % (
            i, batch_size, size))
        generator.refresh()
    elif mode == 'test':
        generator.set_description('TESTING. Batch sample: %d/%d (instance size: %d): ' % (
            i, batch_size, size))
        generator.refresh()

def get_instance_index_input(dat, max_nonzeros, sampling_method,
        device, base=None, size_field_length=None):
    if dat['size'] > max_nonzeros:
        idx = subsample(dat['indices'], size=max_nonzeros, method=sampling_method)
    else:
        idx = np.arange(dat['size'])

    if dat['size'] <= max_nonzeros * 10:
        device = device
    else:
        device = 'cpu'

    append_size_info = base is not None
    index = torch.from_numpy(prepare_global_index(dat['indices'][idx, ...])).to(device)
    size_features = get_size_features(dat['indices'], base=base,
           field_length=size_field_length) if append_size_info else np.zeros((
           idx.shape[0], 0))
    input = torch.from_numpy(np.concatenate([dat['values'][idx, ...],
        size_features[idx, ...]], axis=1)).float().to(device)

    return index, input

def write_prediction_to_file(epoch, points, assigns_points, mode='test', predictions_file=None):
    print(epoch, file=open(predictions_file, 'w'))
    for filename, target, prediction in points:
        print('{filename},{target},{prediction},{mode},0'.format(filename=filename,
            target=target, prediction=prediction, mode=mode),
            file=open(predictions_file, 'a'))
    for filename, target, prediction in assigns_points:
        print('{filename},{target},{prediction},{mode},1'.format(filename=filename,
            target=target, prediction=prediction, mode=mode),
            file=open(predictions_file, 'a'))

def evaluate(model=None, assigns_model=None, exchangeable_model=None, pool_both=None,
             optimizer=None, generator=None, settings=None, mode='train',
             loss_function=nn.MSELoss(), assigns_loss_function=nn.BCELoss(),
             surrogate_loss_function=nn.MSELoss(), batches_per_epoch=5):
    if not mode in ['train', 'validate', 'test', 'update']:
        raise Exception('Evaluation mode %s not supported' % (mode))

    cumulative_batch_loss = 0.
    cumulative_surrogate_batch_loss = 0.
    assigns_surrogate_batch_loss = 0.
    evals = 0
    points = []
    assigns_points = []

    generator = tqdm(generator, leave=True, total=batches_per_epoch)
    batch_count = 0

    for local_batch, local_labels in generator:
        batch_count += 1
        if batches_per_epoch:
            if batch_count > batches_per_epoch:
                break

        optimizer.zero_grad()
        batch_size = len(local_batch)
        batch_evals = 0

        if settings.mode == 'cnn':
            for dat in local_batch:
                y_train = []
                i = torch.LongTensor(dat['indices'])
                v = torch.FloatTensor(dat['values'])

                input = torch.sparse.FloatTensor(i.t(), v).to_dense()
                if settings.architecture.permute_dense:
                    input = input[torch.randperm(input.size()[0])]
                    input = input[:, torch.randperm(input.size()[1])]
                input = input.unsqueeze(0)
                input = input.permute(0, 3, 1, 2)

                y_train.append(dat['target'])
                y_train = np.array(y_train)
                features = input.float().to(settings.device)
                target = torch.from_numpy(y_train).float().to(settings.device)

                if mode == 'train':
                    y_pred = model(features)
                    loss = loss_function(y_pred.flatten(), target.flatten())
                    cumulative_batch_loss += detach(loss)
                else:
                    y_pred = model(features)
                    loss = loss_function(y_pred.flatten(), target.flatten())
                    cumulative_batch_loss += detach(loss)

                if settings.predict_sat:
                    surrogate_loss = 1. - np.mean(np.abs(np.round(detach(
                        y_pred)).flatten() - detach(target).flatten()))
                    cumulative_surrogate_batch_loss += surrogate_loss

                if mode == 'train':
                    loss.backward()

            if mode == 'train':
                optimizer.step()

            evals += batch_size

        elif settings.single_instance_batching:
            for dat in local_batch:
                batch_evals += 1
                evals += 1
                set_generator_descriptor(generator, mode=mode,
                        batch_size=batch_size, size=dat['size'], i=batch_evals)

                index, input = get_instance_index_input(dat, settings.max_nonzeros,
                        settings.opt.sampling_method, settings.device,
                        base=settings.arch.size_base,
                        size_field_length=settings.arch.size_field_length)

                if model != 'nn_handcrafted':
                    model._index = index
                    exchangeable_model.index = index
                    if settings.predict_assigns:
                        assigns_model._index = index
                    if pool_both:
                        pool_both.index = index

                target = torch.from_numpy(np.array([dat['target']]))[:, None]
                target = target.float().to(settings.device)

                assigns_target = torch.from_numpy(np.array(dat['assigns']))
                assigns_target = target.float().to(settings.device)

                if settings.arch.bins != 1:
                    orig_target = target
                    target = np.array([np.argmax(apply_discr(target, settings.bin_edges,
                        settings.binwidth), 0)])
                    target = torch.from_numpy(target).to(settings.device)

                features_append = torch.from_numpy(dat['features_normalized']).float(
                        ).to(settings.device)
                if settings.arch.append_handcrafted:
                    prediction, predicted_features = model(input, features_append)
                else:
                    if settings.mode == 'nn_handcrafted':
                        input = features_append

                    prediction = model(input)[None, :]
                    if settings.predict_assigns and len(assigns_target) > 0:
                        assigns_prediction = assigns_model(input).squeeze()

                if settings.arch.append_size_info:
                    if mode == 'train' or settings.predict_sat:
                        loss = loss_function(prediction, target)
                    else:
                        loss = loss_function(torch.clamp(prediction,
                            float(dat['lower_bound']), float(dat['upper_bound'])),
                            target)
                else:
                    loss = loss_function(prediction.flatten(), target.flatten())
                loss /= batch_size

                assigns_loss = 0.
                if settings.predict_assigns and len(assigns_target) > 1:
                    assigns_loss = assigns_loss_function(assigns_prediction, assigns_target)
                    assigns_loss /= batch_size

                if mode == 'train':
                    loss.backward()
                    if settings.predict_assigns and len(assigns_target) > 1:
                        assigns_loss.backward()

                surrogate_loss = 0
                if settings.arch.bins != 1:
                    max_out = torch.argmax(prediction)
                    dediscr_out = settings.bin_edges[max_out] + (settings.binwidth / 2)
                    surrogate_loss = surrogate_loss_function(dediscr_out, dat.target)
                    surrogate_loss = detach(surrogate_loss)
                elif settings.predict_sat:
                    surrogate_loss = 1. - np.mean(np.abs(np.round(detach(
                        prediction)).flatten() - detach(target).flatten()))
                else:
                    surrogate_loss = loss
                    surrogate_loss = detach(surrogate_loss)

                if settings.predict_assigns and len(assigns_target) > 1:
                    assigns_surrogate_loss = 1. - np.mean(np.abs(np.round(detach(
                                                assigns_prediction)).flatten() - detach(
                                                assigns_target).flatten()))

                if np.isnan(detach(loss)):
                    raise Exception('NAN loss. File: %f, Prediction: %f, Target: %f' % (
                        dat.filename, prediction, target))

                points.append((dat['filename'], target, prediction.detach()))
                cumulative_batch_loss += detach(loss) * batch_size
                cumulative_surrogate_batch_loss += surrogate_loss

                if settings.predict_assigns and len(assigns_target) > 1:
                    assigns_points.append((dat['filename'], assigns_target,
                        assigns_prediction.detach()))
                    cumulative_batch_loss += detach(assigns_loss) * batch_size
                    assigns_surrogate_batch_loss += assigns_surrogate_loss

            if mode == 'train':
                optimizer.step()
        else:
            features = []
            y_train = []
            for dat in local_batch:
                lower_bound = float(dat['lower_bound'])
                upper_bound = float(dat['upper_bound'])
                features.append(dat['features_normalized'])
                y_train.append(dat['target'])

            features = np.array(features)
            y_train = np.array(y_train)
            features = torch.from_numpy(features).float().to(settings.device)
            target = torch.from_numpy(y_train).float().to(settings.device)

            if mode == 'train':
                y_pred = model(features)
                loss = loss_function(y_pred.flatten(), target.flatten())
                
                if settings.predict_sat:
                    cumulative_batch_loss += detach(loss) * batch_size
                else:
                    y_pred_clamped = torch.clamp(model(features), lower_bound, upper_bound)
                    loss_clamped = loss_function(y_pred_clamped.flatten(), target.flatten())
                    cumulative_batch_loss += detach(loss_clamped) * batch_size
            else:
                if settings.predict_sat:
                    y_pred = model(features)
                    loss = loss_function(y_pred.flatten(), target.flatten())
                    cumulative_batch_loss += detach(loss) * batch_size
                else:
                    y_pred = torch.clamp(model(features), lower_bound, upper_bound)
                    loss = loss_function(y_pred.flatten(), target.flatten())
                    cumulative_batch_loss += detach(loss) * batch_size

            if settings.predict_sat:
                surrogate_loss = 1. - np.mean(np.abs(np.round(detach(
                    y_pred)).flatten() - detach(target).flatten()))
                cumulative_surrogate_batch_loss += surrogate_loss * batch_size

            if mode == 'train':
                loss.backward()
                optimizer.step()

            evals += batch_size

    mean_batch_loss = cumulative_batch_loss / (1. * evals)
    root_mean_batch_loss = np.sqrt(mean_batch_loss)
    mean_surrogate_batch_loss = cumulative_surrogate_batch_loss / (1. * evals)
    if settings.predict_assigns:
        mean_assigns_surrogate_batch_loss = assigns_surrogate_batch_loss / (1. * evals)

    return (root_mean_batch_loss, mean_surrogate_batch_loss, points,
            assigns_surrogate_batch_loss, assigns_points)

def build_random_forest(training_set, validation_set, test_set, target='runtime'):
    X_train, y_train = training_set.get_X_y_matrices()
    X_valid, y_valid = validation_set.get_X_y_matrices()
    X_test, y_test = test_set.get_X_y_matrices()

    if target == 'runtime':
        mod = ske.RandomForestRegressor(n_estimators=10, max_features=0.5,
                min_samples_split=5)
    elif target == 'sat':
        mod = ske.RandomForestClassifier(n_estimators=99, max_features='log2',
                min_samples_split=5)
    else:
        raise Exception('Target %s not supported: must be SAT or runtime' % (target))

    loss_t = 0.
    mod = mod.fit(X_train, y_train)
    prediction = mod.predict(X_train)
    acc_t = 1. - np.mean(np.abs(prediction - y_train))
    loss_t = rmse_error(prediction, y_train)

    loss_v = 0.
    prediction = mod.predict(X_valid)
    acc_v = 1. - np.mean(np.abs(prediction - y_valid))
    loss_v = rmse_error(prediction, y_valid)

    loss_e = 0.
    prediction = mod.predict(X_test)
    acc_e = 1. - np.mean(np.abs(prediction - y_test))
    loss_e = rmse_error(prediction, y_test)

    if target == 'runtime':
        print('RF TRAINING. RMSE: {loss}.'.format(loss=loss_t))
        print('RF EVALUATION. RMSE: {loss}.'.format(loss=loss_v))
        print('RF TEST. RMSE: {loss}.'.format(loss=loss_e))
    elif target == 'sat':
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i, label in enumerate(y_test):
            if label == 1:
                if prediction[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if prediction[i] == 1:
                    fp += 1
                else:
                    tn += 1

        print('True Positive: %s, True Negative: %d, '
              'False Positive: %d, False Negative: %d' % (tp, tn, fp, fn))

        print('RF TRAINING. Accuracy: {loss}.'.format(loss=acc_t))
        print('RF EVALUATION. Accuracy: {loss}.'.format(loss=acc_v))
        print('RF TEST. Accuracy: {loss}.'.format(loss=acc_e))
    else:
        raise Exception('Target: %s not supported' % target)

    return loss_t, loss_v, loss_e

def main():
    args = get_args()
    if args.verbose:
        print('Arguments: %s' % args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    settings = Settings(args)

    if settings.load_path_just_features:
        settings.load_path_just_features = torch.load(settings.load_path_just_features)

    hyper_params = vars(args)
    if not settings.no_comet:
        experiment = Experiment(api_key=settings.comet_api_key,
                project_name=settings.project_name, workspace=settings.comet_username,
                auto_output_logging=None)
        experiment.log_parameters(hyper_params)
        experiment_name = '%s_%d_%d' % (settings.dataset, settings.arch.exchangeable_layers,
                settings.arch.units)
        experiment.set_name(experiment_name)

    files = glob.glob(settings.data_directory + '/*.npz')
    if len(files) == 0:
        raise Exception('No instances in with dataset: ' + settings.data_directory)
    if settings.test_data_directory:
        test_files = glob.glob(settings.test_data_directory + '/*.npz')
        if len(test_files) == 0:
            raise Exception('No instances in test dataset: ' + settings.test_data_directory)
    else:
        test_files = []

    train_files, validation_files, test_files = get_instance_partitions(files,
            partition_from_file=settings.delegate, data_path=settings.data_directory,
            test_data_path=settings.test_data_directory, test_files=test_files)

    if settings.verbose:
        print('Normalizing features')

    train_mean, train_sd = get_feature_normalization_params(train_files,
            settings.opt.filter_features)
    test_mean, test_sd = get_feature_normalization_params(test_files, 
            settings.opt.filter_features)
    feature_dim = train_mean.shape[0]

    if settings.verbose:
        print('Dataset has %d features' % (feature_dim))
        print('Building datasets')

    training_set = CombinatorialOptimizationDataset(train_files,
            features_model=settings.load_path_just_features,
            predict_residual=settings.predict_residual,
            predict_runtime=settings.predict_runtime,
            predict_sat=settings.predict_sat,
            predict_assigns=settings.predict_assigns,
            log_transform=settings.log_transform,
            features_mean=train_mean, features_sd=train_sd,
            filter_features=settings.opt.filter_features, device=settings.device)

    if not settings.arch.size_field_length:
        buffer = 1
        min_required_field_length = training_set.get_max_nonzeroes()
        settings.arch.size_field_length = min_required_field_length + buffer
        if settings.verbose:
            print('Field length not set. Max size in training set: %d. Adding %d buffer' % (
                min_required_field_length, buffer))

    validation_set = CombinatorialOptimizationDataset(validation_files,
            features_model=settings.load_path_just_features,
            predict_residual=settings.predict_residual,
            predict_runtime=settings.predict_runtime,
            predict_sat=settings.predict_sat,
            predict_assigns=settings.predict_assigns,
            log_transform=settings.log_transform,
            features_mean=train_mean, features_sd=train_sd,
            lower_bound=training_set.lower_bound, upper_bound=training_set.upper_bound,
            filter_features=settings.opt.filter_features, device=settings.device)

    if settings.dataset_test:
        test_set = CombinatorialOptimizationDataset(test_files,
                features_model=settings.load_path_just_features,
                predict_residual=settings.predict_residual,
                predict_runtime=settings.predict_runtime,
                predict_sat=settings.predict_sat,
                predict_assigns=settings.predict_assigns,
                log_transform=settings.log_transform,
                features_mean=test_mean, features_sd=test_sd,
                lower_bound=training_set.lower_bound, upper_bound=training_set.upper_bound,
                filter_features=settings.opt.filter_features, device=settings.device)
    else:
        test_set = CombinatorialOptimizationDataset(test_files,
                features_model=settings.load_path_just_features,
                predict_residual=settings.predict_residual,
                predict_runtime=settings.predict_runtime,
                predict_sat=settings.predict_sat,
                predict_assigns=settings.predict_assigns,
                log_transform=settings.log_transform,
                features_mean=train_mean, features_sd=train_sd,
                lower_bound=training_set.lower_bound, upper_bound=training_set.upper_bound,
                filter_features=settings.opt.filter_features, device=settings.device)

    if settings.mode == 'rf' or settings.add_random_forest:
        if settings.predict_sat:
            train_loss, validation_loss, testing_loss = build_random_forest(training_set,
                    validation_set, test_set, target='sat')
        elif settings.predict_runtime:
            train_loss, validation_loss, testing_loss = build_random_forest(training_set,
                    validation_set, test_set, target='runtime')
        else:
            raise Exception('Neither SAT or runtime selected as prediction target')

        if settings.mode == 'rf':
            if not settings.no_comet:
                loss_description = 'loss'
                experiment.log_metric('train_%s' % loss_description, train_loss)
                experiment.log_metric('validation_%s' % loss_description, validation_loss)
                experiment.log_metric('test_%s' % loss_description, testing_loss)
            exit()
        elif not settings.no_comet:
            loss_description = 'rf_loss'
            experiment.log_metric('train_%s' % loss_description, train_loss)
            experiment.log_metric('validation_%s' % loss_description, validation_loss)
            experiment.log_metric('test_%s' % loss_description, testing_loss)

    if settings.verbose:
        print('Building model')

    if settings.load_path:
        if not os.path.exists(settings.load_path):
            raise Exception('Model cannot be loaded from ' + settings.load_path)
        model_dict = torch.load(settings.load_path)
        model = model_dict['model']
        exchangeable_model = list(model.modules())[1]

        if settings.predict_assigns:
            assigns_model = model_dict['assigns_model']
        else:
            assigns_model = None
    else:
        sample_data_point = training_set.__getitem__(0)
        model, assigns_model, exchangeable_model, pool_both, batch_norm = setup_parameterized_model(
                settings, sample_data_point, feature_dim=feature_dim)

    model.to(settings.device)
    if settings.predict_assigns:
        assigns_model.to(settings.device)

    parameter_list = list(model.parameters())
    if settings.predict_assigns:
        parameter_list += list(assigns_model.parameters())

    optimizer = torch.optim.Adam(parameter_list, lr=settings.opt.learning_rate)
    if settings.verbose:
        print('Learning rate: %f' % settings.opt.learning_rate)
        print('Number of training examples: %d. Batch size: %d' % (len(train_files),
            settings.opt.batch_size))

    loss_function = None
    if settings.predict_runtime:
        loss_function = nn.MSELoss()
        loss_description = 'mse'
        surrogate_loss_description = 'se'
    elif settings.predict_sat:
        loss_function = nn.BCELoss()
        loss_description = 'bce'
        surrogate_loss_description = 'accuracy'
    assigns_loss_function = nn.BCELoss()
    assigns_surrogate_loss_description = 'accuracy_assigns'

    if settings.arch.bins != 1:
        surrogate_loss_function = nn.CrossEntropyLoss()
        bin_setup(settings, training_set.filenames)
    else:
        surrogate_loss_function = None

    batching_params = {'batch_size': settings.opt.batch_size,
                       'shuffle': True,
                       'num_workers': 0,
                       'collate_fn': collate_fn}

    training_generator = data.DataLoader(training_set, **batching_params)
    validation_generator = data.DataLoader(validation_set, **batching_params)
    test_generator = data.DataLoader(test_set, **batching_params)

    if settings.verbose:
        print('Beginning training')

    for epoch in range(settings.opt.num_epochs):
        if epoch % settings.update_frequency == 0 and settings.arch.use_batch_norm:
            if settings.verbose:
                print('Updating batch norm parameters')
            model.eval()
            if settings.arch.use_batch_norm:
                batch_norm.set_mode(running_updates=True)

            evaluate(model=model, assigns_model=assigns_model,
                     exchangeable_model=exchangeable_model, pool_both=pool_both,
                     generator=training_generator, optimizer=optimizer,
                     settings=settings, mode='validate', loss_function=loss_function,
                     assigns_loss_function = assigns_loss_function,
                     surrogate_loss_function=surrogate_loss_function,
                     batches_per_epoch=settings.opt.batches_per_epoch)

            if settings.arch.use_batch_norm:
                batch_norm.set_mode(running_updates=False)

        model.train()
        optimizer.zero_grad()

        training_loss = np.inf
        recon_training_loss = np.inf
        points = []
        assigns_points = []
        if not (settings.load_path and epoch == 0):
            (training_loss, recon_training_loss, points,
                    assigns_training_loss, assigns_points) = evaluate(model=model,
                    assigns_model=assigns_model, exchangeable_model=exchangeable_model,
                    pool_both=pool_both, generator=training_generator,
                    optimizer=optimizer, settings=settings,
                    mode='train', loss_function=loss_function,
                    assigns_loss_function = assigns_loss_function,
                    surrogate_loss_function=surrogate_loss_function,
                    batches_per_epoch=settings.opt.batches_per_epoch)

            if settings.verbose:
                if settings.predict_assigns:
                    print('TRAINING. Epoch: {ep}. Batch loss: {p_loss}. '
                          'Reconstructed: {r_loss}. Assigns accuracy: {a_loss}.'.format(
                            ep=epoch, p_loss=training_loss, r_loss=recon_training_loss, 
                            a_loss=assigns_training_loss))
                else:
                    print('TRAINING. Epoch: {ep}. Batch loss: {p_loss}. '
                          'Reconstructed: {r_loss}.'.format(
                            ep=epoch, p_loss=training_loss, r_loss=recon_training_loss)) 

            if not settings.no_comet:
                experiment.log_metric('train_%s' % loss_description, training_loss)
                experiment.log_metric('train_%s' % surrogate_loss_description,
                        recon_training_loss)
                if settings.predict_assigns:
                    experiment.log_metric('train_%s' % assigns_surrogate_loss_description,
                            assigns_training_loss)

        if epoch % settings.validation_frequency == 0:
            with torch.no_grad():
                if not args.offline_eval:
                    model.eval()
                    if settings.arch.use_batch_norm:
                        if settings.verbose:
                            print('Updating batch norm for validation')
                        batch_norm.set_mode(running_updates=True)
                        evaluate(model=model, assigns_model=assigns_model,
                                 exchangeable_model=exchangeable_model,
                                 pool_both=pool_both, generator=validation_generator,
                                 optimizer=optimizer, settings=settings,
                                 mode='validate', loss_function=loss_function,
                                 assigns_loss_function = assigns_loss_function,
                                 surrogate_loss_function=surrogate_loss_function,
                                 batches_per_epoch=None)
                        batch_norm.set_mode(running_updates=False)

                    (validation_loss, recon_validation_loss, points,
                            assigns_validation_loss, assigns_points) = evaluate(model=model,
                            assigns_model=assigns_model,
                            exchangeable_model=exchangeable_model,
                            pool_both=pool_both, generator=validation_generator,
                            optimizer=optimizer, settings=settings,
                            mode='validate', loss_function=loss_function,
                            assigns_loss_function=assigns_loss_function,
                            surrogate_loss_function=surrogate_loss_function,
                            batches_per_epoch=None)

                    if not settings.no_comet:
                        experiment.log_metric('validation_%s' % loss_description,
                                validation_loss)
                        experiment.log_metric('validation_%s' % surrogate_loss_description,
                                recon_validation_loss)
                        if settings.predict_assigns:
                            experiment.log_metric(
                                    'validation_%s' % assigns_surrogate_loss_description,
                                    assigns_validation_loss)

                    test_loss = None
                    recon_test_loss = None
                    if test_files:
                        if settings.arch.use_batch_norm:
                            if settings.verbose:
                                print('Updating batch norm for testing')
                            batch_norm.set_mode(running_updates=True)
                            evaluate(model=model, assigns_model=assigns_model,
                                     exchangeable_model=exchangeable_model,
                                     pool_both=pool_both, generator=test_generator,
                                     optimizer=optimizer, settings=settings, mode='test',
                                     loss_function=loss_function,
                                     ssigns_loss_function=assigns_loss_function,
                                     surrogate_loss_function=surrogate_loss_function,
                                     batches_per_epoch=None)
                            batch_norm.set_mode(running_updates=False)

                        (test_loss, recon_test_loss, points,
                                assigns_test_loss, assigns_points) = evaluate(model=model,
                                assigns_model=assigns_model,
                                exchangeable_model=exchangeable_model,
                                pool_both=pool_both, generator=test_generator,
                                optimizer=optimizer, settings=settings, mode='test',
                                loss_function=loss_function,
                                assigns_loss_function=assigns_loss_function,
                                surrogate_loss_function=surrogate_loss_function,
                                batches_per_epoch=None)

                        if not settings.no_comet:
                            experiment.log_metric('test_%s' % loss_description, test_loss)
                            experiment.log_metric('test_%s' % surrogate_loss_description,
                                    recon_test_loss)
                            if settings.predict_assigns:
                                experiment.log_metric(
                                        'test_%s' % assigns_surrogate_loss_description,
                                        assigns_test_loss)

                        if settings.save_predictions:
                            write_prediction_to_file(epoch, points, assigns_points,
                                    predictions_file=settings.predictions_path)

                    write_output(epoch=epoch,
                                 train=(training_loss, recon_training_loss,
                                        assigns_training_loss),
                                 validation=(validation_loss, recon_validation_loss,
                                     assigns_validation_loss, points, assigns_points),
                                 test=(test_loss, recon_test_loss, assigns_test_loss),
                                 predict_assigns=settings.predict_assigns,
                                 verbose=settings.verbose,
                                 write=settings.save_predictions,
                                 log_path=settings.performance_logger_path)

                    if settings.save_checkpts_logs:
                        checkpt_file = os.path.join(settings.checkpt_directory,
                                '%s_ep_%06d' % (settings.experiment_id, epoch) + ('_' + 
                                    settings.time if settings.timestamp else '') + '.pt')
                        if settings.verbose:
                            print('Checkpt file: %s' % (checkpt_file))
                        model_dict = {'model': model, 'assigns_model': assigns_model}
                        torch.save(model_dict, checkpt_file)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
