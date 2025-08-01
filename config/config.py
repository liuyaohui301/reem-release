from pathlib import Path
import torch

# env
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
hidden_size = 128
embedding_size = 64
seq_len = 24

# training
batch_size = 64
learning_rate = 1e-4
lambda_sparsity = 0.01
epochs = 100

# path
PROJECT_ROOT = Path(__file__).parent.parent
data_root_path = 'data/raw/osaka/2024'
data_cache_dir = 'data/cache/osaka/2024'
pool_root_path = 'models/pool'
