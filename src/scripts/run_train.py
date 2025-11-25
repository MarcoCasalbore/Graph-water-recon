import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0+PTX"  # per nuove GPU

import wandb
import argparse

from src.config import Parameters
from src.utils.utils import get_model, set_params_wrt_dataset, get_datamodule
from src.utils.utils import train_and_test

# Parser
parser = argparse.ArgumentParser(description="Experiments parameters!")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size!")
parser.add_argument("--verbose", type=bool, default=True, help="Batch size!")

args = parser.parse_args()

# Parameters
run_params = Parameters(args)

# Get dataset
train_l, val_l, test_l, run_params = get_datamodule(run_params)

# Modify parameters for dataset
run_params = set_params_wrt_dataset(run_params, train_l)

# Import model
model = get_model(run_params)

# # Training
train_and_test(model, train_l, val_l, test_l, run_params)
