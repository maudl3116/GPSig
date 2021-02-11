from models import train_gpsig_classifier

import sys
import os
import json

GPU_ID = str(sys.argv[1]) if len(sys.argv) > 1 else '-1'

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

with open('./datasets.json', 'r') as f:
    datasets = json.load(f)
    
results_dir = './results/GPSig_IS/'
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

num_experiments = 3

for i in range(num_experiments):
    for dataset in datasets:

        results_filename = os.path.join(results_dir, '{}_{}.txt'.format(dataset, i))

        if os.path.exists(results_filename):
            print('{} already exists, continuing...'.format(results_filename))
            continue

        with open(results_filename, 'w'):
            pass

        train_gpsig_classifier(dataset, num_levels=4, num_inducing=dataset['M'], len_inducing=5, use_tensors=False, num_lags=0, increments=True, learn_weights=False,
                               val_split=dataset['val_split'], normalize_data=True, train_spec=dataset['train_spec'],minibatch_size=dataset['minibatch_size'], experiment_idx=i, save_dir=results_dir)      