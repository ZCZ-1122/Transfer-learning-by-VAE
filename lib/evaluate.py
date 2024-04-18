"""Evaluates the model"""

# import argparse
import logging
import os

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import torch
from cs230 import utils
from cs230 import net
from cs230.autoencoder import sample
from cs230.visu import show_reconstruction, show_save_imgs
# from model.data_loader import DataLoader

# from torcheval.metrics.classification import BinaryRecall, BinaryPrecision, \
                                            #  BinaryAccuracy
# from torcheval.metrics import MulticlassAccuracy

def predict(loader, model, args):
    out_dict = {'all_imgs' : [],
           'all_x_hat' : [],
           'all_z' : [],
           'all_mu' : [],
           'all_logvar' : [],
           }

    # inference
    model.eval()
    keys = ['x_hat','z','mu','logvar']
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch['image'].to(device=args.device, dtype=args.dtype)
            output_batch = model(batch)

            # stock batch input and outputs
            out_dict['all_imgs'].append(batch)
            for k in keys:
                out_dict[f'all_{k}'].append(output_batch[k])
        # concatenation and to numpy
        for k in keys + ['imgs']:
            tmp_tensor = torch.cat(out_dict[f'all_{k}'], dim=0)
            out_dict[f'all_{k}'] = tmp_tensor.detach().cpu().numpy()

    return out_dict


def predict_unet(loader, model, args):
    model.eval()
    
    out_dict = {'all_imgs' : [],
                'all_labs' : [],
                'all_segs' : [],
        }
    with torch.no_grad():
        for batch in tqdm(loader):
            image = batch["image"].to(args.device)
            logits = model(image)
            pred = F.softmax(logits, dim=1).argmax(dim=1)
            
            out_dict['all_imgs'].append(image)
            out_dict['all_labs'].append(batch["label"])
            out_dict['all_segs'].append(pred)

        for k, v in out_dict.items():
            tmp_tensor = torch.cat(v, dim=0)
            out_dict[k] = tmp_tensor.detach().cpu().numpy()
    return out_dict
        
    
### for VAE
# def evaluate(model, loss_fn, dataloader, metrics, params, 
#              show_imgs, recons_save_path, gene_save_path):
#     """Evaluate the model on `num_steps` batches.

#     Args:
#         model: (torch.nn.Module) the neural network
#         loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
#         metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
#         params: (Params) hyperparameters
#     """

#     # set model to evaluation mode
#     model.eval()

#     # summary for current eval loop
#     summ = []
    
#     with torch.no_grad():
#         # compute metrics over the dataset
#         for data_batch in dataloader:
#             data_batch = data_batch['image']
#             data_batch = data_batch.to(device=params.device, dtype=params.dtype)
            
#             # compute model output
#             output_batch = model(data_batch)
#             loss = loss_fn(output_batch, data_batch,params.kld_weight)

#             # compute all non-streaming metrics on this batch
#             data_batch = data_batch.detach().cpu().numpy()
#             recons_batch = output_batch['x_hat'].detach().cpu().numpy()

#             summary_batch = {m: metrics[m](recons_batch, data_batch)
#                             for m in metrics}
#             summary_batch['loss'] = loss.item()
#             summ.append(summary_batch)
        
#         if show_imgs:
#             # reconstruction
#             show_reconstruction(data_batch, recons_batch, nb_picked=4, save=True,
#                                 save_path=recons_save_path)
#             # generation
#             samples = sample(model, num_samples=8, latent_dim=params.latent_dim,
#                             device=params.device, seed=230)
#             show_save_imgs(samples, save=True, save_path=gene_save_path)
            

#         # compute and log mean of all metrics in summary
#         metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 

#         metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
#         logging.info("- Eval metrics : " + metrics_string)

#     return metrics_mean

def evaluate(model, loss_fn, dataloader, metrics, params,):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    
    with torch.no_grad():
        # compute metrics over the dataset
        for batch in dataloader:
            image_batch = batch['image'].to(device=params.device, dtype=params.dtype)
            label_batch = batch['label'].to(device=params.device, dtype=torch.long)
            logits_batch = model(image_batch)
            
            loss = loss_fn(logits_batch, label_batch)

            # compute all non-streaming metrics on this batch
            logits_batch = logits_batch.detach()
            label_batch = label_batch.detach()

            summary_batch = {m: metrics[m](logits_batch, label_batch)
                            for m in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # compute and log mean of all metrics in summary
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean