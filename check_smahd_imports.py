import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

import torch
import torch_geometric

import generate_datasets
import layer
import train
import utils

print("imports ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("pyg", torch_geometric.__version__)
print("train function", hasattr(train, "train_SMAHD"))
