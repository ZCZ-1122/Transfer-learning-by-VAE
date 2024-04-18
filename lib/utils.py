import json
import logging
import os
import shutil
from scipy import stats

import torch

import numpy as np
import cv2
from tqdm import tqdm
import glob

import torch.nn.functional as F

class Params():
    def __init__(self,params_dict):
        self.__dict__.update(params_dict)

    def update(self,params_dict):
        self.__dict__.update(params_dict)


def subdirs(
    folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> list[str]:
    """Get a list of subdirectories in a folder.

    Args:
        folder: The path to the folder.
        join: Whether to join the folder path with subdirectory names. Defaults to True.
        prefix: Filter subdirectories by prefix. Defaults to None.
        suffix: Filter subdirectories by suffix. Defaults to None.
        sort: Whether to sort the resulting list. Defaults to True.

    Returns:
        A list of subdirectory names in the given folder.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E731, E741
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)
    

def interpreate_shapiro_test(data, alpha=0.05, verbose=1):
    statistic, p_value = stats.shapiro(data)

    # Set significance level (alpha)
    if verbose:
        print(f"{'Null hypothesis':30} : data was drawn from a normal distribution.")
        print(f"{'Shapiro-Wilk Test Statistic':30} : {statistic:.3f} (Normal law if it is smaller but close to 1)")
        print(f"{'alpha':30} : {alpha}")
        print(f"{'p-value':30} : {p_value:.3f}")
        print()
    # Interpret the results
    if p_value > alpha:
        if verbose:
            print("p_value is larger than alpha")
            print("We fail to reject the null hypothesis")
            print("We accept that the data follows a Gaussian distribution (not significant)")
        return True
    else:
        if verbose:
            print("The data does not follow a Gaussian distribution (significant)")
        return False


# Load dataset
def load_CAMUS_dataset(images_path, img_size):

    # Create list of files to load for both images and the corresponding labels
    masks_files  = glob.glob("{}/labels/*.png".format(images_path))
    images_files = glob.glob("{}/images/*.png".format(images_path))

    # Introduce some randomness in the order the images will be loaded
    images_files.sort()
    masks_files.sort()
    permutation_index = np.random.permutation( len(masks_files))
    images_files_rnd = [images_files[i] for i in permutation_index]
    masks_files_rnd = [masks_files[i] for i in permutation_index]

    # Reading images and corresponding labels
    X = ReadImages(images_files_rnd, size=(img_size, img_size))
    y = ReadMasks(masks_files_rnd, size=(img_size, img_size))

    # Return the images (X) and corresponding labels (y) into numpy array
    return X, y


# Reading images
def ReadImages(images_files, size, crop=None):
    X = []
    for index in tqdm(range(len(images_files))):
        image_read = cv2.imread(images_files[index], cv2.IMREAD_GRAYSCALE)
        if crop is not None:
            image_read = image_read[crop[0]:crop[2],crop[1]:crop[3]]
        image_read = cv2.resize(image_read, dsize = size, interpolation = cv2.INTER_LINEAR)
        image_read = image_read / 255.0
        X.append(image_read)
    X = np.asarray(X, dtype=np.float32)
    X = np.expand_dims(X, axis=1)
    return X

# Reading masks
def ReadMasks(images_files, size, crop=None):
    y = []
    for index in tqdm(range(len(images_files))):
        image_read = cv2.imread(images_files[index], cv2.IMREAD_GRAYSCALE)
        if crop is not None:
            image_read = image_read[crop[0]:crop[2],crop[1]:crop[3]]
        image_read = cv2.resize(image_read, dsize = size, interpolation = cv2.INTER_NEAREST)
        y.append(image_read)
    y = np.asarray(y, dtype=np.int16)
    y[y==255]=3
    y[y==170]=2
    y[y==85]=1
    # num_classes = 3
    # y_one_hot = F.one_hot(y, num_classes=num_classes)
    return y

class StreamingAcc():
    """A simple class that maintains the streaming accuracy
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.total_correct = 0
        self.total_sample = 0
    
    def update(self, outputs, labels):

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        N = outputs.shape[0]

        preds = (outputs >= 0.0)  # sigmoid(z) > 0.5 if z > 0
        nb_correct = np.sum(preds==labels)

        self.total_correct += nb_correct
        self.total_sample += N
    
    def __call__(self):
        return self.total_correct/float(self.total_sample)
        


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint