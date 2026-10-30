import sys
import time
import copy
import datetime
import numpy as np
import argparse

import torch
import torch.nn as nn
from torchinfo import summary
import yaml

import warnings
import os
import pickle
warnings.filterwarnings('ignore')

import torch.nn.functional as F


sys.path.append('..')
from utils.log import print_log
from utils.metrics import MSE_MAE, RMSE_MSE_MAPE_MAE, MAE, MSE, RMSE, MAPE

# from model.STKRA import STKRA_teacher, STKRA_student_residual, STKRA_student_general
from model.STKRA import STKRA_student_residual, STKRA_student_general
from model.STAEFormer import STAEformer

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

class STKRARunner():
    def __init__(self, cfg:dict, device, scaler, log=None):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.scaler =scaler
        self.log = log

        self.clip_grad = cfg['OPTIM'].get('clip_grad')

        self.tmodel = self.loadTeacher()
        self.stmodel = self.loadStudent_general()

    def loadTeacher(self):
        tmodel = STAEformer(**self.cfg['MODEL_PARAM_STAE']).to(self.device)
        checkpoint_path = '/checkpoints/PEM/teacher/PEM_g56.pt' # G56
        # checkpoint_path = '/checkpoints/PEM/teacher/PEM_G60.pt' # G60
        if checkpoint_path:
            print_log(f"Loading teacher model from {checkpoint_path}", log=self.log)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            model_dict = tmodel.state_dict()

            pretrained_dict = {
                k: v for k, v in checkpoint.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }

            model_dict.update(pretrained_dict)
            
            if 'model_state_dict' in checkpoint:
                tmodel.load_state_dict(checkpoint['model_state_dict'])
            else:
                tmodel.load_state_dict(checkpoint, strict=False)
            
            print_log("Teacher model loaded successfully", log=self.log)
            return tmodel
        
    def loadStudent_general(self):
        sgmodel = STKRA_student_general(**self.cfg['MODEL_PARAM']).to(self.device)
        checkpoint_path = '/checkpoints/G56/student_general/G56_general.pt' # PEM
        # checkpoint_path = '/checkpoints/G60/student_general/G60_general.pt' # PEM
        if checkpoint_path:
            print_log(f"Loading student_general model from {checkpoint_path}", log=self.log)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                sgmodel.load_state_dict(checkpoint['model_state_dict'])
            else:
                # If the checkpoint is just the model state dict
                sgmodel.load_state_dict(checkpoint)
            
            print_log("Student_general model loaded successfully", log=self.log)
            return sgmodel

    def SimCLRLoss(self, z_i, z_j, temperature=0.1):
        batch_size = z_i.shape[0]
        
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0) 
        
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), 
            representations.unsqueeze(0), 
            dim=2
        )
        
        mask = torch.eye(batch_size, device=z_i.device).repeat(2, 2)
        pos_mask = mask.bool()
        
        neg_mask = ~pos_mask
        neg_mask = neg_mask.fill_diagonal_(False)  
        
        pos_sim = similarity_matrix[pos_mask].view(2 * batch_size, -1)
        
        neg_sim = similarity_matrix[neg_mask].view(2 * batch_size, 2 * batch_size - 2)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
        labels = torch.zeros(2 * batch_size, device=z_i.device, dtype=torch.long)
        
        loss = F.cross_entropy(logits, labels)
        return loss


    def train_one_epoch(self, model, train_loader, optimizer, scheduler, criterion):

        model.train()

        self.tmodel.eval()

        self.stmodel.eval()

        batch_loss_list = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            y_batch = y_batch[..., 0]

            # # teacher
            # out_batch, intermediates, U_s, U_t = model(x_batch)

            # # student_general
            # out_batch, intermediates, U_s, U_t = model(x_batch)
            # out_batch_teacher, intermediates_teacher, U_S, U_T = self.tmodel(x_batch)

            # student_residual
            out_batch_student_general, intermediates_student_general, U_s, U_t = self.stmodel(x_batch)
            out_batch, intermediates, out2, out3 = model(x_batch, intermediates_student_general)

            # # teacher
            # loss = criterion(out_batch, y_batch)

            # student_genaral
            # loss1 = criterion(out_batch, y_batch)
            # loss_t = criterion(out_batch, out_batch_teacher)
            
            # kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True)
            # loss_kl = kl_loss(nn.functional.log_softmax(intermediates, dim=1), nn.functional.log_softmax(intermediates_teacher, dim=1))
            # loss_rel = criterion(U_S, U_s) + criterion(U_T, U_t)
            # alpha = self.cfg['OPTIM'].get('alpha')
            # beta = self.cfg['OPTIM'].get('beta')
            # loss = loss1 + alpha * loss_t + (1 - alpha) * loss_kl + beta * loss_rel

            # student_residual
            out_batch = out_batch + out_batch_student_general
            loss1 = criterion(out_batch, y_batch)
            contrastive_loss = 0
            for i in range(intermediates.shape[0]): 
                orig_feat = intermediates[i, :] 
                noise_feat = out2[i, :]  
                mask_feat = out3[i, :] 
                
                contrastive_loss += self.SimCLRLoss(orig_feat, noise_feat)
                contrastive_loss += self.SimCLRLoss(orig_feat, mask_feat)

            loss2 = contrastive_loss / (intermediates.shape[1] * 2)
            loss_res = self.cfg['OPTIM'].get('loss_res')
            loss_contrast = self.cfg['OPTIM'].get('loss_contrast')
            loss = loss_res * loss1 + loss_contrast * loss2

            batch_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg['OPTIM'].get('clip_grad'))
            optimizer.step()

        epoch_loss = np.mean(batch_loss_list)
        scheduler.step()

        return epoch_loss

    @torch.no_grad()
    def eval_model(self, model, val_loader, criterion):

        model.eval()

        self.stmodel.eval()

        batch_loss_list = []
        for x_batch, y_batch in val_loader:

            x_batch = x_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)
            y_batch = y_batch[..., 0]

            # # not residual
            # out_batch, intermediates, U_S, U_T = model(x_batch)

            # residual
            out_batch_student_general, intermediates_student_general, U_s, U_t = self.stmodel(x_batch)
            out_batch, intermediates, out2, out3 = model(x_batch, intermediates_student_general)
            out_batch = out_batch + out_batch_student_general

            loss = criterion(out_batch.cpu(), y_batch.cpu())
            
            batch_loss_list.append(loss.item())

        return np.mean(batch_loss_list)


    @torch.no_grad()
    def predict(self, model, loader):

        model.eval()

        self.tmodel.eval()

        y = []
        out = []

        for x_batch, y_batch in loader:

            x_batch = x_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)
            y_batch = y_batch[..., 0]

            # # not residual
            # out_batch, intermediates, U_S, U_T = model(x_batch)

            # residual
            out_batch_student_general, intermediates_student_general, U_s, U_t = self.stmodel(x_batch)
            out_batch_residual, intermediates, out2, out3 = model(x_batch, intermediates_student_general)
            out_batch = out_batch_residual + out_batch_student_general
            
            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
        
            out.append(out_batch)
            y.append(y_batch)            

        out = np.vstack(out)  
        y = np.vstack(y)

        out = self.scaler.inverse_transform(out)
        y = self.scaler.inverse_transform(y)

        return y, out

    def train(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        max_epochs=10,
        early_stop_patience=3,
        compile_model=False,
        verbose=1,
        save=None):

        if torch.__version__ >= '2.0.0' and compile_model:
            model = torch.compile(model)

        wait = 0
        min_val_loss = np.inf

        train_loss_list = []
        val_loss_list = []

        start = time.time()
        for epoch in range(max_epochs):
            train_loss = self.train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion
            )
            train_loss_list.append(train_loss)

            val_loss = self.eval_model(model, val_loader, criterion)
            val_loss_list.append(val_loss)

            if (epoch + 1) % verbose == 0:
                print_log(
                    datetime.datetime.now(),
                    'Epoch',
                    epoch + 1,
                    ' \tTrain Loss = %.5f' % train_loss,
                    'Val Loss = %.5f' % val_loss,
                    log=self.log,
                )

            if val_loss < min_val_loss:
                wait = 0
                min_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= early_stop_patience:
                    break    
        end = time.time()   

        model.load_state_dict(best_state_dict)

        if save:
            torch.save(best_state_dict, save)  

        train_rmse, train_mse, train_mape, train_mae = RMSE_MSE_MAPE_MAE(*self.predict(model, train_loader))
        val_rmse, val_mse, val_mape, val_mae = RMSE_MSE_MAPE_MAE(*self.predict(model, val_loader))

        out_str = f'Finish at epoch: {epoch+1}\n'
        out_str += f'Best model at epoch {best_epoch+1}:\n'
        out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
        out_str += "Train RMSE = %.5f, MAPE = %.5f, MAE = %.5f\n" % (
            train_rmse,
            train_mape,
            train_mae,
        )
        out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
        out_str += "Val RMSE = %.5f, MAPE = %.5f, MAE = %.5f" % (
            val_rmse,
            val_mape,
            val_mae,
        )
        print_log(out_str, log=self.log)
        print_log("Traing time per epoch: %.3f s" % ((end - start)/epoch), log=self.log)

        return model


    @torch.no_grad()
    def test_model(self, model, test_loader):
        
        model.eval()

        print_log('--------- Test ---------', log=self.log)
        
        
        start = time.time()
        y_true, y_pred = self.predict(model, test_loader)
        end = time.time()

        out_steps = y_pred.shape[1]

        steps = [2, 4, 6]
        # steps = [4, 8, 12]
        mae_met = 0
        mse_met = 0
        rmse_met = 0
        mape_met = 0

        for i in range(out_steps):
            mae_met = mae_met + MAE(y_true[:, i, ...], y_pred[:, i, ...])
            mse_met = mse_met + MSE(y_true[:, i, ...], y_pred[:, i, ...])
            rmse_met = rmse_met + RMSE(y_true[:, i, ...], y_pred[:, i, ...])
            mape_met = mape_met + MAPE(y_true[:, i, ...], y_pred[:, i, ...])

            if (i + 1) in steps:
                step = i + 1
                out_str = "Step %d: RMSE = %.5f, MAPE = %.5f, MAE = %.5f" % (
                    step,
                    rmse_met/step,
                    mape_met/step,
                    mae_met/step
                )

                print_log(out_str, log=self.log)

    def model_summary(self, model, dataloader):

        x_shape = next(iter(dataloader))[0].shape

        return summary(
            model,
            x_shape,
            verbose=0,  # avoid print twice
            device=self.device
        )