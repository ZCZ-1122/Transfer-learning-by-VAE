"""Train the model"""

# import argparse
import logging
import os
# from ipywidgets.widgets.widget_float import validate

import numpy as np
import datetime
import torch.optim as optim
# from torcheval.metrics import MulticlassAccuracy

# from torch.autograd import Variable
from tqdm import tqdm
import torch

from cs230 import utils
from cs230 import net

# import model.data_loader as data_loader
from cs230.evaluate import evaluate

from torch.utils.tensorboard import SummaryWriter


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []

    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, train_batch in enumerate(dataloader):
            # move to GPU if available
            train_batch = train_batch['image']
            train_batch  = train_batch.to(device=params.device, dtype=params.dtype)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, train_batch, params.kld_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the average loss and streaming accuracy for this batch
            loss_avg.update(loss.item())

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                train_batch = train_batch.detach().cpu().numpy()
                recons_batch = output_batch['x_hat'].detach().cpu().numpy()
                
                # compute all non-streaming metrics on this batch
                summary_batch = {m: metrics[m](recons_batch, train_batch)
                                 for m in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Compute and log mean of all metrics in summary
    metrics_mean = {metric: np.mean([ x[metric]
                                     for x in summ ]) for metric in summ[0]}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    tr_epoch_hist = metrics_mean  
    tr_epoch_hist['loss'] = loss_avg()

    return tr_epoch_hist



def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # Tensorboard
    writer = SummaryWriter()
    
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, restore_file +'.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    history = {'train_loss' : [],
                'val_loss' : [],
                'train_mse' : [],
                'val_mse' : [],
    }
    
    current_time = datetime.datetime.now()
    date_time = current_time.strftime("%Y%m%d_%H%M%S")
    recons_img_dir = os.path.join(params.model_dir, f'{date_time}_reconstruction')
    gene_img_dir = os.path.join(params.model_dir, f'{date_time}_generation')
    os.makedirs(recons_img_dir, exist_ok=True)
    os.makedirs(gene_img_dir, exist_ok=True)

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        tr_epoch_hist = train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        show_imgs = epoch % params.show_imgs_every == 1
        recons_save_path = os.path.join(recons_img_dir, f'reconstructed_val_imgs_epoch_{epoch}.png')
        gene_save_path = os.path.join(gene_img_dir, f'generated_imgs_epoch_{epoch}.png')
        val_epoch_hist = evaluate(model, loss_fn, val_dataloader, metrics, params, 
                                  show_imgs, recons_save_path, gene_save_path)
        
        val_acc = val_epoch_hist['acc']
        is_best = val_acc >= best_val_acc
             
        # Log train, val loss and metrics to TensorBoard
        writer.add_scalar("Loss/train", tr_epoch_hist['loss'], epoch)
        writer.add_scalar("Loss/val", val_epoch_hist['loss'], epoch)
        
        writer.add_scalar("MSE/train", tr_epoch_hist['mse'], epoch)
        writer.add_scalar("MSE/val", val_epoch_hist['mse'], epoch)
        
        # Save train, val loss and metrics on epoch end
        for (k,v) in tr_epoch_hist.items():
            history[f'train_{k}'].append(v) 
        for (k,v) in val_epoch_hist.items():
            history[f'val_{k}'].append(v)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            if params.save_best:
                # Save best val metrics in a json file in the model directory
                # best_json_path = os.path.join(
                    # model_dir, "metrics_val_best_weights.json")
                # utils.save_dict_to_json(val_metrics, best_json_path)
                best_model_path = os.path.join(params.model_dir, "so_far_best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f'checkpoint saved to {best_model_path}')

        # # Save latest val metrics in a json file in the model directory
        # last_json_path = os.path.join(
        #     model_dir, "metrics_val_last_weights.json")
        # utils.save_dict_to_json(val_metrics, last_json_path)
    
    writer.flush() # make sure that all pending events have been written to disk.
    writer.close()
    
    # restore the best model
    if params.restore_best : 
        model.load_state_dict(torch.load(best_model_path))
        print(f'\n*** Best model reloaded from {best_model_path} ***\n')

    return history
