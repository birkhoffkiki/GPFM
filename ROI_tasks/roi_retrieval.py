import torch
import argparse
from downstream_tasks.metrics import build_retrieval_metric
from torch.nn.functional import one_hot, softmax
import json
import os


# Setup the argument parser
parser = argparse.ArgumentParser(description='ROI retrieval')
parser.add_argument('--train_feat_path', type=str, default=None)
parser.add_argument('--val_feat_path', type=str, default=None)
parser.add_argument('--metric_file_path', type=str, default=None)
parser.add_argument('--me', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=1000)


args = parser.parse_args()

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

print('loading features')
# Load features and labels
train_features_w_ids = torch.load(args.train_feat_path)
val_features_w_ids = torch.load(args.val_feat_path)


# Convert data into numpy arrays, then into PyTorch tensors, and move them to the specified device (GPU if available)
train_features = train_features_w_ids['feature'].to(device)
val_features = val_features_w_ids['feature'].to(device)


# normlization
var, mean = torch.var_mean(train_features, dim=0, keepdim=True)
std = torch.sqrt(var)
print('Variance shape:', var.shape)
print('Mean shape:', mean.shape)
train_features = (train_features - mean)/std
val_features = (val_features - mean)/std

train_labels = train_features_w_ids['label']
val_labels = val_features_w_ids['label']
num_classes = val_labels.max() + 1

print('num of train features:', train_features.size(0))
print('num of val features:', val_features.size(0))

# Define a batch size
batch_size = args.batch_size  # Adjust based on the memory capacity

def knn_batch(train_features, val_features, k, batch_size):
    num_val = val_features.size(0)
    distances = []
    indices = []
    for i in range(0, num_val, batch_size):
        batch = val_features[i:i + batch_size]
        # Compute the distances from the batch to all train features
        dist = torch.cdist(batch, train_features)
        # Obtain the k nearest neighbors for the batch
        k_dist, k_idx = torch.topk(dist, k, largest=False, sorted=True)
        distances.append(k_dist)
        indices.append(k_idx)
    return torch.cat(distances), torch.cat(indices)

print('predicting')

# Call knn_batch function
distances, indices = knn_batch(train_features, val_features, num_classes, batch_size)


metrics = build_retrieval_metric(num_classes=num_classes)

# eval 
prediciton = []
target_labels = []

for i in range(len(val_labels)):
    if i % 100 == 0:
        print('Progress: [{}/{}]'.format(i, len(val_labels)))
        
    target_label = val_labels[i:i+1]
    prob = torch.zeros((1, num_classes))
    ratio = 1.0
    for j in indices[i][:num_classes]:
        prob[0, train_labels[j]] += ratio
        ratio -= 0.1
    prob = softmax(prob, dim=-1)
    prediciton.append(prob)
    target_labels.append(target_label)

prediciton = torch.cat(prediciton, dim=0)
target_labels = torch.cat(target_labels)

metrics.update(prediciton, target_labels)

results_dict = {k: m.compute() for k, m in metrics.items()}
final_dict = {}
for knn_name, result in results_dict.items():
    temp_dict = {}
    for k, v in result.items():
        temp_dict[k] = v.item()
    final_dict[knn_name] = temp_dict

# save result
prefix = os.path.split(args.metric_file_path)[0]
if not os.path.exists(prefix):
    os.makedirs(prefix)

with open(args.metric_file_path, "w") as f:
    for k, v in final_dict.items():
        f.write(json.dumps({k: v}) + "\n")    
    
