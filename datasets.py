import numpy as np
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import (Dataset, DataLoader)
from tqdm import tqdm

def get_bounds(train_files=None):
    upper_bound = -np.inf
    lower_bound = np.inf

    for i in tqdm(range(len(train_files))):
        dat = np.load(train_files[i])
        if 'runtime' not in dat.keys():
            continue
        runtime = max(float(dat['runtime']), 0.005)
        runtime = np.log10(runtime)
        if runtime > upper_bound:
            upper_bound = runtime
        if runtime < lower_bound:
            lower_bound = runtime

    return upper_bound, lower_bound

class CombinatorialOptimizationDataset(Dataset):
    PROBING_FEATURES = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
        76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
        96, 97, 98, 99, 100, 101, 102, 103, 111, 112, 113, 114, 115, 116, 117, 118,
        119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
        135,136,137]
    TIMING_FEATURES = [6, 21, 42, 53, 59, 66]
    def __init__(self, filenames, features_model=None, predict_residual=False,
                 predict_runtime=True, predict_sat=False, predict_assigns=False,
                 log_transform=True, features_mean=None, features_sd=None, upper_bound=None,
                 lower_bound=None, filter_features=True, device='cuda'):
        self.filenames = filenames
        self.features_model = features_model
        self.predict_residual = predict_residual
        self.size = len(filenames)
        self.predict_runtime = predict_runtime
        self.predict_sat = predict_sat
        self.predict_assigns = predict_assigns
        self.log_transform = log_transform
        self.features_mean = features_mean
        self.features_sd = features_sd
        self.filter_features = filter_features
        self.device = device

        if self.predict_runtime and self.predict_sat:
            raise Exception('Can\'t predict both runtime and SAT')

        if not upper_bound or not lower_bound:
            self.upper_bound, self.lower_bound = get_bounds(filenames)
        else:
            self.upper_bound = upper_bound
            self.lower_bound = lower_bound

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = np.load(self.filenames[index], allow_pickle=True, mmap_mode='r')
        assignment = []

        try:
            indices = data['indices']
            indices = np.array(indices, dtype='int')

            values = data['values']
            values = np.array(values[:len(indices)],
                    dtype='float32').reshape(indices.shape[0], -1)

            if 'features' not in data.keys():
                features = np.array([[1.0]], dtype='float32')
            else:
                features = np.array([data['features']], dtype='float32')

            if 'runtime' not in data.keys():
                runtime = max(0.005, 1.0)
            else:
                runtime = max(0.005, data['runtime'])

            if self.filter_features:
                features = np.delete(features, self.PROBING_FEATURES)
                features = [np.delete(features, self.TIMING_FEATURES)]

            features = features[0]
            normalized_features = features
            features[features == -512] = self.features_mean[(features == -512)]
            normalized_features = (features - self.features_mean) / self.features_sd

            self.runtime = np.array(runtime, dtype='float32')
            handcrafted_features_prediction = 0.

            if self.predict_runtime:
                if self.log_transform:
                    target = np.log10(self.runtime)
                else:
                    target = self.runtime

                if self.predict_residual:
                    features_model = self.features_model.to(self.device)
                    with torch.no_grad():
                        if not torch.cuda.is_available():
                            raise Exception('No cuda available!')

                        torch_normalized_features = torch.from_numpy(
                                np.array(normalized_features[0])).float()
                        torch_normalized_features = torch_normalized_features.to(self.device)
                        handcrafted_features_prediction = torch.clamp(features_model(
                            torch_normalized_features), self.lower_bound, self.upper_bound)

            elif self.predict_sat or self.predict_assigns:
                assignment = np.array(data['y_sat'], dtype='float32')
                target = np.asarray([0 if len(assignment) == 0 else 1])

            else:
                raise Exception('Not working for non-runtime targets...')

        except:
            print(self.filenames[index])
            raise

        return {'indices': indices,
                'values': values,
                'target': target,
                'assigns': assignment,
                'size': len(indices),
                'handcrafted_features_prediction': handcrafted_features_prediction,
                'filename': self.filenames[index],
                'features': features,
                'features_normalized': normalized_features,
                'lower_bound': self.lower_bound,
                'upper_bound': self.upper_bound}

    def get_max_nonzeroes(self):
        max_nonzeroes = 0

        for file in self.filenames:
            dat = np.load(file, allow_pickle=True, mmap_mode='r')
            indices = dat['indices']
            values = dat['values'][:len(indices)]
            #matrix = csr_matrix((values, (indices[:, 0], indices[:, 1])),
            #        shape=indices.max(axis=0) + 1)
            max_nonzeroes_point = len(indices)
            #max_nonzeroes_point = max(matrix.sum(axis=0), matrix.sum(axis=1))
            if max_nonzeroes_point > max_nonzeroes:
                max_nonzeroes = max_nonzeroes_point

        return max_nonzeroes

    def get_X_y_matrices(self):
        X = []
        y = []

        for i in range(self.size):
            dat = self.__getitem__(i)
            X.append(dat['features_normalized'])
            y.append(dat['target'])

        X = np.array(X)
        y = np.array(y).transpose()

        return X, y
