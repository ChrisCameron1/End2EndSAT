from collections import OrderedDict
from exchangable_tensor.sp_layers import SparseExchangeable
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
from utils import to_bin

def output_size(in_size=None, kernel_size=None, stride=None, padding=None):
    output = int(((in_size + 2 * (padding) - (kernel_size - 1) - 1 ) / stride) + 1.5)

    return output

class SimpleCNN(nn.Module):
    def __init__(self, sample_data_point, settings):
        super(SimpleCNN, self).__init__()

        num_rows = sample_data_point['indices'].max(axis=0)[0]
        num_cols = sample_data_point['indices'].max(axis=0)[1]

        self.conv1 = nn.Conv2d(2, 18, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        num_rows = output_size(in_size=num_rows, kernel_size=2.0, stride=2.0, padding=0)
        num_cols = output_size(in_size=num_cols, kernel_size=2.0, stride=2.0, padding=0)

        self.conv3 = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        num_rows = output_size(in_size=num_rows, kernel_size=2.0, stride=2.0, padding=0)
        num_cols = output_size(in_size=num_cols, kernel_size=2.0, stride=2.0, padding=0)
        self.num_rows = output_size(in_size=num_rows, kernel_size=2.0,
                stride=2.0, padding=0) - 1
        self.num_cols = output_size(in_size=num_cols, kernel_size=2.0,
                stride=2.0, padding=0)

        self.fc1 = nn.Linear(18 * self.num_rows * self.num_cols, 64)
        self.fc2 = nn.Linear(64, 1)

        if settings.predict_sat:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = self.pool1(x)
        x = func.relu(self.conv2(x))
        x = self.pool2(x)
        x = func.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(-1, 18 * self.num_rows * self.num_cols)
        x = func.relu(self.fc1(x))

        x = self.fc2(x)
        return (self.sigmoid(x))

class ResidualNetwork(nn.Module):
    def __init__(self, features, index, norm=False, identity=True):
        super(ResidualNetwork, self).__init__()
        self.exchange_layer1 = SparseExchangeable(features, features, index)
        self.exchange_layer2 = SparseExchangeable(features, features, index)
        self._index = index
        self.act1 = nn.LeakyReLU()
        self.act2 = nn.LeakyReLU()
        self.act3 = nn.LeakyReLU()
        if norm:
            self.batch_norm = nn.LayerNorm(features)
        self.norm = norm
        self.identity = identity

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index
        self.exchange_layer1.index = index
        self.exchange_layer2.index = index

    def forward(self, input):
        if self.norm:
            normed = self.batch_norm(input)
        else:
            normed = input
        step1 = self.act1(self.exchange_layer1(normed))
        step2 = self.act2(self.exchange_layer2(step1))
        return self.act3(input + step2)

    def cached_forward(self, input, index, batch_size):
        if self.norm:
            normed = self.batch_norm(input)
        else:
            normed = input
        pre_act1 = self.exchange_layer1.cached_forward(normed, index.cpu(), batch_size)
        step1 = self.act1(pre_act1)
        pre_act2 = self.exchange_layer1.cached_forward(step1, index.cpu(), batch_size)
        step2 = self.act2(pre_act2)
        return self.act3(input + step2)

class HybridNetwork(nn.Module):
    def __init__(self, exchangeable_model, exchangeable_model_dim, feature_dim, units,
                 out_dim, feedforward_layers, dropout=0., predict_features=False, 
                 append_size_info=False, zero_exch=False, predict_sat=False, 
                 predict_assigns=False, k=None, base=None, noise=0.):
        super(HybridNetwork, self).__init__()
        self.exchangeable_model = exchangeable_model
        self.predict_features = predict_features
        self.exchangeable_model_dim = exchangeable_model_dim
        self.feature_dim = feature_dim
        self.units = units
        self.feedforward_layers = feedforward_layers
        self.append_size_info = append_size_info
        self.zero_exch = zero_exch
        self.predict_sat = predict_sat
        self.predict_assigns = predict_assigns
        self.k = k
        self.base = base
        self.noise = noise

        model_input_dim = None
        if not append_size_info:
            if not predict_assigns:
                print('Using hybrid network with hand-crafted features, no size information')
            model_input_dim = exchangeable_model_dim + feature_dim
        else:
            if not predict_assigns:
                print('Using hybrid network with hand-crafted features plus size information')
            model_input_dim = exchangeable_model_dim + feature_dim + (self.k * 2)

        modules = [('lin1', nn.Linear(in_features=model_input_dim, out_features=units)),
                   ('relu1', nn.LeakyReLU())]
        
        for layer in range(feedforward_layers):
            lr_string = str(2 + layer)
            modules += [('lin' + lr_string, nn.Linear(in_features=units, out_features=units)),
                        ('relu' + lr_string, nn.LeakyReLU())]
            if dropout > 0:
                modules += [('dropout' + lr_string, nn.Dropout(p=dropout))]

        modules += [('lin' + str(2 + feedforward_layers),
                    nn.Linear(in_features=units, out_features=out_dim))]

        if predict_sat or predict_assigns:
            modules += [('sigmoid', nn.Sigmoid())]

        self.mlp = nn.Sequential(OrderedDict(modules))

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = index
        self.exchangeable_model.index = index

    def forward(self, input, features):
        exch = self.exchangeable_model(input)
        if not self.predict_assigns:
            exch = exch.mean(dim=0)[None, :]

        if self.zero_exch:
            exch *= 0

        if self.append_size_info:
            index = self._index if isinstance(self._index,
                    np.ndarray) else self._index.detach().cpu().numpy()
            maxes = multi_noise(np.max(index, axis=0), self.noise)
            size_features = to_bin(maxes, self.base, self.k).reshape(1, -1)
            size_features = torch.from_numpy(size_features).float().to(input.device)

        pred_features = None
        if self.predict_features:
            exch, pred_features = torch.split(exch,
                    [self.exchangable_model_dim, self.feature_dim])
            if not self.training:
                features = pred_features
            else:
                features = features + torch.normal(torch.zeros_like(features),
                    0.25 * torch.ones_like(features))

        features = features.view(1, int(list(features.size())[0]))

        concat_list = [exch, features]
        if self.append_size_info:
            concat_list += size_features

        concat = torch.cat(concat_list, dim=1)
        return self.mlp(concat), pred_features

def multi_noise(indices, noise=0.):
    if noise == 0.:
        return indices
    else:
        if not isinstance(indices, np.ndarray):
            indices = indices.cpu().detach().numpy()
        new_indices = np.zeros_like(indices)
        dat_org = indices
        dat_new = dat_org * (np.random.randn(*list(dat_org.shape)) * noise + 1)
        dat_new = np.maximum(dat_new, 1)
        return dat_new

class SparsePoolBoth(nn.Module):
    def __init__(self, keep_dims=False, index=None, base=None, k=None, noise=0.):
        super(SparsePoolBoth, self).__init__()
        self.keep_dims = keep_dims
        self.index = index
        self.base = base
        self.k = k
        self.noise = noise

    def forward(self, input):
        both_mean = torch.mean(input, dim=0)[None, :]
        if self.keep_dims:
            both_mean = both_mean.expand_as(input)
        if self.index is not None:
            index = self.index if isinstance(self.index,
                    np.ndarray) else self.index.detach().cpu().numpy()
            maxes = multi_noise(np.max(index, axis=0), self.noise)
            if self.base is not None:
                features = to_bin(maxes, self.base, self.k).reshape(1, -1)
                size_features = torch.from_numpy(features).float().to(input.device)
                both_mean = torch.cat([both_mean, size_features], dim=1)

        return both_mean

class CustomBatchNorm(nn.Module):
    def __init__(self,num_features, device=None):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.running_mean = torch.zeros(self.num_features).to(device)
        self.running_var = torch.ones(self.num_features).to(device)
        self.num_batches_tracked= 0
        self.running_updates = False
        self.device = device

    def set_mode(self, running_updates=True):
        self.running_updates = running_updates
        self.num_batches_tracked = 0
        self.running_mean = torch.zeros(self.num_features).to(self.device)
        self.running_var = torch.ones(self.num_features).to(self.device)

    def forward(self, input):
        self.num_batches_tracked += 1
        if self.running_updates:
            old_running_mean = self.running_mean
            self.running_mean = self.running_mean.data + (
                    input.data - self.running_mean.data) / float(self.num_batches_tracked)
            self.running_var = self.running_var.data + (
                    input.data - old_running_mean.data) * (
                    input.data - self.running_mean.data)

        normalized_output = (input - self.running_mean) / torch.sqrt(self.running_var)
        return normalized_output
