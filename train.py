import os
import json
import yaml
import random
import argparse
import datetime
import numpy as np
import pickle
from torchinfo import summary

import torch

from data.dataloader import select_dataloader
from utils.log import print_log, CustomJSONEncoder
from utils.metrics import select_loss
from STKRA.runners.STKRARunner import STKRARunner
from model.STKRA import STKRA_student_residual, STKRA_student_general
from model.STAEFormer import STAEformer

parser = argparse.ArgumentParser()
# Specify the dataset
parser.add_argument('-d','--dataset_name',type=str,default='G56')
# Specify config file path
parser.add_argument('-cfg','--config_path',type=str,default='/STKRA/config/G56/G56_Seq96.yaml')
# Specify random seed
parser.add_argument('-sd','--seed',type=int,default=2026)
# checkpoint path for testing
parser.add_argument('-c','--checkpoint',type=str,default='')
# mode (teacher or student)
parser.add_argument('-m','--mode',type=str,default='student_residual') # teacher, student_residual, student_general
# dataset_mode 
parser.add_argument('-dm','--dataset_mode',type=str,default='overload') # normal, overload

args = parser.parse_args()

os.environ['PYTHONHASHSEED'] = str(args.seed)

random.seed(args.seed)
np.random.seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If the checkpoint is just the model state dict
        model.load_state_dict(checkpoint)
    
    print("Checkpoint loaded successfully")
    return model

def test_only(model, test_loader, runner, device, log):
    print_log("Testing pre-trained model from checkpoint...", log=log)
    runner.test_model(model, test_loader)

def train():
    # Set device
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    cfg_path = args.config_path
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    data_path = f'/STKRA/data/{args.dataset_name}/'

    # --------- Load the model --------- #
    if args.mode == 'teacher':
        model = STAEformer(**cfg['MODEL_PARAM_STAE']).to(DEVICE)
    elif args.mode == 'student_general':
        model = STKRA_student_general(**cfg['MODEL_PARAM']).to(DEVICE)
    elif args.mode == 'student_residual':
        model = STKRA_student_residual(**cfg['MODEL_PARAM']).to(DEVICE)
    else:
        raise ValueError('No such mode')

    # --------- Make log file --------- #
    log_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_path = f'/STKRA/logs/{args.dataset_name.upper()}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f'{args.dataset_name.upper()}-{log_time}.log')
    log = open(log, 'a')
    log.seek(0)
    log.truncate()

    # --------- Load the dataset --------- #
    print_log(f'Dataset used: {args.dataset_name.upper()}', log=log)
    (train_loader, val_loader, test_loader, SCALER) = select_dataloader(cfg["GENERAL"]["task_name"])(
        data_path,
        batch_size=cfg['GENERAL'].get('batch_size', 32),
        in_steps=cfg['DATA'].get('in_steps', 96),
        out_steps=cfg['DATA'].get('out_steps', 96),
        x_tod=cfg['DATA'].get('x_time_of_day'),
        x_dow=cfg['DATA'].get('x_day_of_week'),
        y_tod=cfg['DATA'].get('y_time_of_day'),
        y_dow=cfg['DATA'].get('y_day_of_week'),
        mode=args.dataset_mode,
        dataset=args.dataset_name,
        log=log
    )
    print_log(log=log)

    # --------- Set checkpoint path --------- #
    save_path = f'/STKRA/checkpoints/{args.dataset_name.upper()}/{args.mode}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f'{args.dataset_name.upper()}-{log_time}.pt')

    # --------- Set optim options --------- #
    criterion = select_loss(cfg['OPTIM'].get('loss', 'MSE'))(**cfg['OPTIM'].get('loss_args', {}))

    # TODO: Enable changing optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['OPTIM'].get('initial_lr', 0.001),
        weight_decay=cfg['OPTIM'].get('weight_decay', 0),
        eps=cfg['OPTIM'].get('eps', 1e-8)
    )

    lr_scheduler_type = cfg['OPTIM'].get('lr_scheduler_type', 'ExponentialLR')
    if lr_scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg['OPTIM'].get('lr_scheduler_gamma', 0.5),
            verbose=False
        )
    elif lr_scheduler_type == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            steps_per_epoch=len(train_loader),
            max_lr= cfg['OPTIM'].get('initial_lr'),
            epochs=cfg['GENERAL'].get('max_epochs'),
            pct_start=cfg['OPTIM'].get('lr_scheduler_pct_start')
        )
    elif lr_scheduler_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.get('milestones', []),
        gamma=cfg.get('lr_decay_rate', 0.1),
        verbose=False
        )
    else: 
        raise ValueError('No such lr scheduler') 

    # --------- Set model runner --------- #

    runner_name = cfg['GENERAL'].get('runner', 'LTSFRunner')
    if runner_name == 'STFRunner':
        runner = STKRARunner(
        cfg, device=DEVICE, scaler=SCALER, log=log
    )
    elif runner_name == 'LTSFRunner':
        runner = STKRARunner(
        cfg, device=DEVICE, scaler=SCALER, log=log
    )

    # --------- Print model args --------- #
    print_log(f'Random seed = {args.seed}', log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )

    # --------- Train the model --------- #
    print_log(f'Model checkpoint saved to: {save}', log=log)
    print_log(log=log)

    model = runner.train(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=cfg['GENERAL'].get('max_epochs', 10),
        early_stop_patience=cfg['GENERAL'].get('early_stop_patience', 3),
        compile_model=False,
        verbose=1,
        save=save,
    )

    # --------- Test the model --------- #
    runner.test_model(model, test_loader)

    log.close()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train()