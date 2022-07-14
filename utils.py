import argparse
import cnf
from enum import Enum
from glob import glob
import json
import math
import numpy as np
import os
import random
from scipy.sparse import csr_matrix
import string
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import (xavier_uniform_, normal)

from datasets import CombinatorialOptimizationDataset
import exchangable_tensor
from exchangable_tensor.sp_layers import (mean_pool, SparsePool, SparseExchangeable,
        SparseSequential, prepare_global_index)

def get_size_features(index, base=None, field_length=None):
    index_add = lambda idx, size: torch.zeros(size).index_add_(0, idx,
            torch.ones_like(idx).float())
    features = np.zeros((index.shape[0], field_length * index.shape[1]))
    for k in range(index.shape[1]):
        size = index[:, k].max() + 1
        counts = index_add(torch.from_numpy(index[:, k]), size)[index[:, k]]
        features[:, k * field_length:(k + 1) * field_length] = to_bin(counts.numpy(),
                base=base, k=field_length)
    return features

def rmse_error(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def rmse_loss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))

def get_partition_file(file_reg_expression):
    file_matches = glob(file_reg_expression)
    if len(file_matches) > 1:
        print('Warning: More than one partition file: %s. Taking first' % (file_matches))

    for i in file_matches:
        if 'subsample' in i:
            return i

    return file_matches[0]

def get_feature_normalization_params(files, filter_features=False):
    features_train = []
    failed_feature_comp = 0

    for file in tqdm(files):
        try:
            features = np.load(file, allow_pickle=True, mmap_mode='r')['features']
            if filter_features:
                features = np.delete(features,
                        CombinatorialOptimizationDataset.PROBING_FEATURES)
                features = np.delete(features,
                        CombinatorialOptimizationDataset.TIMING_FEATURES)
            features_train.append(features)
        except:
            failed_feature_comp += 1
            features_train.append([1.0])
            raise

    if failed_feature_comp > 0:
        print('Warning: failed feature comp on %d/%d files. Setting dummy features...' % (
            failed_feature_comp, len(files)))
        features_train = []
        for file in tqdm(files):
            features_train.append([1.0])

    features_train = np.array(features_train, dtype='float32')
    features_train[features_train == -512] = np.nan

    colmeans = np.nanmean(features_train, axis=0)
    colmeans[np.isnan(colmeans)] = 0.0
    idx = np.where(np.isnan(features_train))
    features_train[idx] = np.take(colmeans, idx[1])

    train_mean, train_sd = (features_train.mean(axis=0), features_train.std(axis=0))

    train_sd[np.isnan(train_sd)] = 1.
    train_sd[train_sd == 0.] = 1e-16
    return train_mean, train_sd

def bin_setup(settings, files):
    bin_edges = None
    outputs = []

    print('Preprocessing instances to discretize outputs')
    files_itr = tqdm(files)

    for filename in files_itr:
        npz_file = np.load(filename, allow_pickle=True, mmap_mode='r')
        runtime = npz_file['runtime']

        if runtime != 0:
            outputs.append(np.log10(runtime))
        else:
            outputs.append(np.log10(0.005))

    hist, bin_edges = np.histogram(outputs, bins=settings.num_bins - 1)
    settings.binwidth = (bin_edges[1:] - bin_edges[0:-1])[0]
    settings.bin_edges = torch.from_numpy(bin_edges).float()

def randomString(stringLength=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def is_sat(dat, y):
    X_sp = csr_matrix((np.dot(dat.values, np.array([[1], [-1]])).flatten(),
                      (dat.index[:, 0],dat.index[:, 1])),
                      shape=dat.index.max(axis=0) + 1)
    if not isinstance(y, np.ndarray):
        y = y.cpu().data.numpy().flatten()
    y_hat = y.round()
    y_hat[y_hat == 0] = -1
    return (X_sp.dot(y_hat) == -3).sum(axis=0) == 0

class ExampleGraph(object):
    def __init__(self, index=[], problem=[]):
        self.index = np.asarray(index)
        self.problem = np.asarray(problem)

    @staticmethod
    def form_example(n_variables, n_clauses, n_items):
        instance = ExampleGraph()

        instance.index = np.concatenate([np.random.randint(n_clauses, size=(n_items, 1)),
                         np.random.randint(n_variables, size=(n_items, 1))], axis=1)
        instance.problem = np.random.randint(0, 2, n_items) * 2 - 1

        return instance

def to_bin(x, base=2, k=8):
    x = x.copy()
    y = np.zeros((x.shape[0], k), dtype=np.int)
    for i in range(k - 1, -1, -1):
        y[:, k - (i + 1)] = x // base ** (i)
        x = x % (base ** (i))
    return y

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def subsample(index, size, method):
    if method == 'neighbour':
        raise NotImplementedError()
    elif method == 'uniform':
        i = np.random.permutation(np.arange(index.shape[0]))
        return i[0:size]

def weights_init(m):
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight.data)

def detach(t):
    return t.cpu().detach().numpy()

def apply_discr(outputs, bin_edges, binwidth):
    outputs = outputs.cpu()[0]
    return np.array(np.logical_and(bin_edges <= outputs,
                                   bin_edges > outputs - binwidth), dtype='int')

def collate_fn(l):
    return l, None

def dimacs2cnf(file_handle):
    header, nvariables, clauses = dimacs2compressed_clauses(file_handle)

    cnf = cnf.CNF(header=header)

    for i in range(1, nvariables + 1):
        cnf.add_variable(i)

    cnf._add_compressed_clauses(clauses)

    cnf._check_coherence(force=True)
    return cnf

def dimacs2compressed_clauses(file_handle):
    n = -1
    m = -1

    my_header = ''
    my_clauses = []

    line_counter = 0
    literal_buffer = []

    for l in file_handle.readlines():
        line_counter += 1

        if l[0] == 'c':
            if l[1] == ' ':
                my_header += l[2:] or '\n'
            else:
                my_header += l[1:] or '\n'
            continue

        if l[0] == 'p':
            if n >= 0:
                raise ValueError('Syntax error: ' +
                                 'line {} contains a second spec line.'.format(line_counter))
            _, _, nstr, mstr = l.split()
            n = int(nstr)
            m = int(mstr)
            continue

        for lv in [int(lit) for lit in l.split()]:
            if lv == 0:
                my_clauses.append(tuple(literal_buffer))
                literal_buffer = []
            else:
                literal_buffer.append(lv)

    if len(literal_buffer) > 0:
        raise ValueError('Syntax error: last clause was incomplete')

    if m == '-1':
        raise ValueError('Warning: empty input formula ')

    if m != len(my_clauses):
        raise ValueError('Warning: input formula ' +
                         'contains {} instead of expected {}.'.format(len(my_clauses), m))

    return (my_header, n, my_clauses)

def convert_cnf(file):
    result = subprocess.run(['./dimacsparser', file], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    lines = output.split('\n')

    variables = 0
    clauses = 0
    indices = []
    values = []

    parse_vars = False
    parse_clauses = False
    parse_indices = False
    parse_values = False

    for line in lines:
        if line.startswith('c Variables'):
            parse_clauses = False
            parse_indices = False
            parse_values = False
            parse_vars = True
        elif line.startswith('c Clauses'):
            parse_vars = False
            parse_indices = False
            parse_values = False
            parse_clauses = True
        elif line.startswith('c Indices'):
            parse_vars = False
            parse_clauses = False
            parse_values = False
            parse_indices = True
        elif line.startswith('c Values'):
            parse_vars = False
            parse_clauses = False
            parse_indices = False
            parse_values = True
        elif parse_vars:
            variables = int(line.strip())
            continue
        elif parse_clauses:
            clauses = int(line.strip())
            continue
        elif parse_indices:
            split = line.strip().split(',')
            clause = int(split[0])
            variable = int(split[1])
            indices.append([clause, variable])
            continue
        elif parse_values:
            strip = line.strip()
            values.append(([1,0] if strip == '1' else [0,1]))
            continue

    return indices, values
