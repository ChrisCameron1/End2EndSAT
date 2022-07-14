import argparse
from comet_ml import Experiment
import glob
import gmatch4py as gm
import networkx as nx
import numpy as np
import os
import random
from tqdm import tqdm

from mypaths import (ROOT, COMET)

def get_graph_from_matrix(file, literal_graph=True):
    data = np.load(file)
    indices = data['indices']
    values = data['values']

    graph = nx.Graph()
    for index, edge in enumerate(indices):
        clause = edge[0]
        variable = edge[1]
        if np.array_equal(values[index], [1,0]):
            value = 1
        elif np.array_equal(values[index], [0,1]):
            value = 0
        else:
            raise Exception('Unknown value %s in file %s' % (value, file))
        graph.add_edge(clause, variable*2+value)
        graph.add_edge(variable * 2, variable * 2 + 1)

    return graph

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Dataset')
parser.add_argument('--distance_alg', type=str, default='greedy')
parser.add_argument('--num_training', type=int, default=None)
args = parser.parse_args()

hyper_params = vars(args)
experiment = Experiment(api_key=COMET['key'], project_name='Nearest Neighbour',
        workspace=COMET['un'], auto_output_logging=None)
experiment.log_parameters(hyper_params)
experiment_name = '%s_%s_%d_literal' % (args.dataset, args.distance_alg, args.num_training)
experiment.set_name(experiment_name)

data_directory = os.path.join(ROOT, 'data', args.dataset, '*.npz')
files = glob.glob(data_directory)
random.shuffle(files)

train_instances_file = os.path.join(ROOT, 'data', args.dataset, 'train.txt')
test_instances_file = os.path.join(ROOT, 'data', args.dataset, 'test.txt')

train_files = []
with open(train_instances_file, 'r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        train_file = os.path.join(ROOT, line)
        train_files.append(train_file)
if not args.num_training:
    num_training = len(train_files)
else:
    num_training = args.num_training

test_files = []
with open(test_instances_file, 'r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        test_file = os.path.join(ROOT, line)
        test_files.append(test_file)

random.shuffle(train_files)
random.shuffle(test_files)

correct = 0
incorrect = 0

node_deletion_penalty = 1
node_insertion_penalty = 1
edge_deletion_penalty = 1
edge_insertion_penalty = 1

if args.distance_alg == 'standard':
    ged = gm.GraphEditDistance(node_deletion_penalty, node_insertion_penalty,
            edge_deletion_penalty, edge_insertion_penalty)
elif args.distance_alg == 'greedy':
    ged = gm.GreedyEditDistance(node_deletion_penalty, node_insertion_penalty,
            edge_deletion_penalty, edge_insertion_penalty)
elif args.distance_alg == 'hausdorff':
    ged = gm.HED(node_deletion_penalty, node_insertion_penalty, edge_deletion_penalty,
                                edge_insertion_penalty)
elif args.distance_alg == 'bipartite':
    ged = gm.BP_2(node_deletion_penalty, node_insertion_penalty, edge_deletion_penalty,
                               edge_insertion_penalty)
else:
    raise Exception('Distance alg %s not supported' % (args.distance_alg))

for i, test_file in enumerate(test_files):
    print('File: %s' % test_file)
    min_distance = np.inf
    test_graph = get_graph_from_matrix(test_file)
    random.shuffle(train_files)

    for train_file in tqdm(train_files[:num_training]):
        if train_file == test_file:
            raise Exception('Test file and train file are identical')
        train_graph = get_graph_from_matrix(train_file)
        result = ged.compare([test_graph, train_graph], None)
        distance = np.sum(ged.distance(result))
        if distance == 0.0:
            print('Files %s and %s exactly isomorphic' % (train_file, test_file))
        if distance < min_distance:
            min_distance = distance
            closest_point = train_file

    print('Closest point: %s' % (closest_point))

    test_assignment = np.array(np.load(test_file)['y_sat'], dtype='float32')
    test_target = 0 if len(test_assignment) == 0 else 1
    print('Test: %d' % (test_target))

    predicted_assignment = np.array(np.load(closest_point)['y_sat'], dtype='float32')
    predicted_target = 0 if len(predicted_assignment) == 0 else 1
    print('Nearest neighbour: %d' % (predicted_target))

    if test_target == predicted_target:
        print('Correct!')
        correct += 1
    else:
        print('Incorrect!')
        incorrect += 1

    accuracy = correct / (correct + incorrect)
    experiment.log_metric('accuracy', accuracy)
    print('Num evals: %d, Accuracy: %f' % (i, accuracy))
