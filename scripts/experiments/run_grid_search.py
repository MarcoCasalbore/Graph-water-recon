"""Legacy Lightning grid search for non-water benchmark datasets."""

import logging
import os
import random
import sys
import itertools

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

import numpy as np
import wandb
from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import Parameters

from src.utils.utils import (
    initialize_parameters,
    update_current_configuration,
    get_datamodule,
    set_params_wrt_dataset,
    get_model,
    update_seed_metrics,
    update_run_metrics,
)

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

# Grid search setup
save_ckpts = False
early_stop_callback_flag = True
save_logs = True
reproducible = True

# Grid search parameters and space
# f_model_list = ['gcn1d', 'gat',]  # dgm
# g_model_list = ['gcn1d', 'gat',]  # diffusion
# dgm_model_list = [f'dgm_{f}_{g}' for f in f_model_list for g in g_model_list]
dgm_model_list = ['dgm_gcn_gcn1d']

model_list = list()
model_list += dgm_model_list
dataset_list = ['METR-LA']  # ['PV', 'wind', 'METR-LA', 'electricity', 'solar']
emd_dim_list = [64, 32]  # [256, 128, 64]
k_list = [9, 7, 5, 3]  # [3, 5, 7]
batch_size_list = [64, 32]  # [64, 32]
lags_list = [12]
prediction_window_list = [12]

combinations = list(itertools.product(model_list,
                                      dataset_list,
                                      emd_dim_list,
                                      k_list,
                                      batch_size_list,
                                      lags_list,
                                      prediction_window_list))
print(f'Total number of combinations: {len(combinations)}')
all_run_cont = len(combinations)
cont = -1
seed_list = [253, 3667, 8]  # , 917, 271
for run_combination in combinations:
    cont += 1
    grid_params_dict = initialize_parameters(cont, run_combination)

    # Reset experiments recorder list
    val_results = list()
    test_results = list()
    """ Step 0 - Parameters """
    for seed in seed_list:
        run_params = Parameters()
        run_params = update_current_configuration(run_combination, seed, run_params, reproducible, save_ckpts,
                                                  save_logs,
                                                  early_stop_callback_flag)

        # Reproducibility
        if run_params.reproducible:
            torch.manual_seed(run_params.seed)
            torch.cuda.manual_seed(run_params.seed)
            random.seed(run_params.seed)
            np.random.seed(run_params.seed)
            seed_everything(run_params.seed)

        # Get dataset
        dataModuleInstance, run_params = get_datamodule(run_params)

        # Modify parameters for dataset
        run_params = set_params_wrt_dataset(run_params, dataModuleInstance)

        # Import model
        model = get_model(run_params)

        # Define logger
        logger = False
        if run_params.logging:
            wandb_key = os.getenv("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key)
            else:
                wandb.login()
            wandb.init(project='DGM', name='0', entity="alessio_v")
            logger = WandbLogger(log_model=False)

        # Training
        early_stop_callback = EarlyStopping(
            monitor='val_mse',
            min_delta=0.00,
            patience=4,
            verbose=False,
            mode='min')
        print('run_params.ffun: ', run_params.ffun)
        print('run_params.gfun: ', run_params.gfun)


        trainer = pl.Trainer(accelerator=run_params.accelerator,
                             log_every_n_steps=run_params.log_every_n_steps,
                             max_epochs=run_params.max_epochs,
                             enable_progress_bar=run_params.enable_progress_bar,
                             enable_model_summary=False,
                             check_val_every_n_epoch=run_params.check_val_every_n_epoch,
                             logger=logger,
                             callbacks=[early_stop_callback])
        trainer.fit(model, datamodule=dataModuleInstance)

        # Testing
        res_test = trainer.test(model, datamodule=dataModuleInstance)

        val_results, test_results = update_seed_metrics(model, res_test, val_results, test_results)

    update_run_metrics(val_results, test_results, grid_params_dict, run_params)

    if run_params.logging:
        wandb.finish()
