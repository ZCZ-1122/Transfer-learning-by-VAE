import glob
import os
from sklearn.model_selection import train_test_split
import numpy as np
import cv2 
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path

from cs230 import utils


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    NormalizeIntensityd,
    
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
)

class sumacDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_dict = {'image': self.data[index]}
        return data_dict


def load_SUMAC_dataset(dataset_dir, img_size, extension="nii.gz",
                        random_state=42, return_paths=False):
    """Load SUMAC images from a  given folder

    Returns:
        output: dictionary{
            X : torch Tensor of shape (N, C, H, W), dtype : torch.float32
            y : torch Tensor of shape (N, Cls_num, H, W), dtype : torch.float32
            image_paths (optional) : list of shuffled image paths
            label_paths (optional) : list of shuffled label paths
        }
    """
    
    all_sub_dirs = os.listdir(dataset_dir)
    all_sub_dirs.sort()

    all_img_paths, all_mask_paths = [], []
    for sub_dir in all_sub_dirs:
        for suffix in ['ED', 'ES']:
            img_name = f"{sub_dir}_img_{suffix}.{extension}"
            mask_name = f"{sub_dir}_mask_{suffix}.{extension}"

            img_path = os.path.join(dataset_dir, sub_dir, img_name)
            mask_path = os.path.join(dataset_dir, sub_dir, mask_name)

            if os.path.exists(img_path) and os.path.exists(mask_path):
                all_img_paths.append(img_path)
                all_mask_paths.append(mask_path)
            else:
                print(f'\n*** {img_path} or {mask_path} does not exist ! ***\n')

    if random_state is not None:
        np.random.seed(random_state)
    permutation_index = np.random.permutation( len(all_img_paths))
    images_files_rnd = [all_img_paths[i] for i in permutation_index]
    masks_files_rnd = [all_mask_paths[i] for i in permutation_index]

    # Reading images and corresponding labels
    output = {}
    X = ReadImagesNIIGZ(images_files_rnd, size=(img_size, img_size))
    X = np.moveaxis(X,-1,1) # move channel axis to the second positioin
    X = np.moveaxis(X,-1,-2) # transpose each image (rotation 90 degrees)
    output['X'] = torch.from_numpy(X)
    
    output['y'] = ReadMasksNIIGZ(masks_files_rnd, size=(img_size, img_size))    
    
    if return_paths:
        output['image_paths'] = images_files_rnd
        output['label_paths'] = masks_files_rnd
    
    return output


def ReadImagesNIIGZ(images_files, size, crop=None):
    X = []
    for index in tqdm(range(len(images_files))):
        image_read = sitk.GetArrayFromImage(sitk.ReadImage(images_files[index]))
        if crop is not None:
            image_read = image_read[crop[0]:crop[2],crop[1]:crop[3]]
        image_read = cv2.resize(image_read, dsize = size, interpolation = cv2.INTER_LINEAR)
        image_read = image_read / 255.0
        X.append(image_read)
    print('\n*** Image intensity values divided by 255 ***\n')
    X = np.asarray(X, dtype=np.float32)
    X = np.expand_dims(X,-1)
    return X


# Reading masks
def ReadMasksNIIGZ(images_files, size, crop=None):
    """_summary_

    Returns:
        y_one_hot: torch Tensor of shape (N, class_num, H, W)
    """
    y = []
    for index in tqdm(range(len(images_files))):
        image_read = sitk.GetArrayFromImage(sitk.ReadImage(images_files[index]))
        if crop is not None:
            image_read = image_read[crop[0]:crop[2],crop[1]:crop[3]]
        image_read = cv2.resize(image_read, dsize = size, interpolation = cv2.INTER_NEAREST)
        y.append(image_read)
    y = np.asarray(y, dtype=np.int16) 
    y = torch.from_numpy(y).long()
    
    n_cls = torch.unique(y).size()[0]
    print(n_cls)
    y_one_hot = F.one_hot(y, n_cls).permute(0,3,1,2).float() # shape (N, C, H, W)
    return y_one_hot


def create_datalist(data_dir, set_name):
    # get all image paths of `set_name` set
    img_dir = os.path.join(data_dir, set_name, 'images')
    img_paths = glob.glob(os.path.join(img_dir,'*.png'))
    img_paths.sort()

    datalist = []
    label_dir = os.path.join(data_dir, set_name, 'labels')
    for img_path in img_paths:
        label_name = os.path.basename(img_path).split('.')[0] + '_gt'
        label_path = os.path.join(label_dir, label_name + '.png')
        assert os.path.exists(label_path),f'\n{label_path} not found\n'
        
        datalist.append({'image':img_path,
                        'label':label_path,})
    return datalist


# def create_datalist(data_dir, set_name):
#     # get all image paths of `set_name` set
#     img_dir = os.path.join(data_dir, set_name, 'images')
#     img_paths = glob.glob(os.path.join(img_dir,'*.png'))
#     img_paths.sort()

#     datalist = []
#     for img_path in img_paths:
#         datalist.append({'image':img_path,
#                         'label':img_path,})
#     return datalist


def create_CAMUS_datalists(data_dir):
    """ 
    Gather image and label paths from data_dir
    Split them into train/val/test sets
    Organize each set into a list of dictionaries 
        
    return : (train_datalist, val_datalist, test_datalist)
                Each one of them is a list of dictionaries. 
                Each dictionary contains an image path and the associated label path
    """
    # Specify the data directory
    data_dir = Path(data_dir).resolve()

    # List all the patients id
    keys = utils.subdirs(data_dir, prefix="patient", join=False)

    # Split the patients into 80/10/10 train/val/test sets
    train_keys, val_and_test_keys = train_test_split(keys, train_size=0.8, random_state=12345)
    val_keys, test_keys = train_test_split(val_and_test_keys, test_size=0.5, random_state=12345)

    train_keys = sorted(train_keys)
    val_keys = sorted(val_keys)
    test_keys = sorted(test_keys)

    # Create train, val and test datalist
    viws_instants = ["2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES"]
    train_datalist = [
        {
            "image": str(data_dir / key / f"{key}_{view}.nii.gz"),
            "label": str(data_dir / key / f"{key}_{view}_gt.nii.gz"),
        }
        for key in train_keys
        for view in viws_instants
    ]

    val_datalist = [
        {
            "image": str(data_dir / key / f"{key}_{view}.nii.gz"),
            "label": str(data_dir / key / f"{key}_{view}_gt.nii.gz"),
        }
        for key in val_keys
        for view in viws_instants
    ]

    test_datalist = [
        {
            "image": str(data_dir / key / f"{key}_{view}.nii.gz"),
            "label": str(data_dir / key / f"{key}_{view}_gt.nii.gz"),
        }
        for key in test_keys
        for view in viws_instants
    ]
    
    return (train_datalist, val_datalist, test_datalist)


def get_load_and_augmentation_transforms(img_size=None):
    # Transforms to load data
    load_transforms = [
        LoadImaged(keys=["image", "label"], image_only=True),  # Load image and label
        EnsureChannelFirstd(keys=["image", "label"]),
    ]
    
    resized_print = ""
    if img_size:
        load_transforms.append(
            Resized(keys=['image','label'], spatial_size=(img_size, img_size))
            )
        resized_print = "Resized"
    load_transforms.append(NormalizeIntensityd(keys=["image"]))
    
    
    print(f"""
    Transforms to load data : 
    -----------------------------------
    LoadImaged
    EnsureChannelFirstd
    {resized_print}
    NormalizeIntensityd
    """)

    # Transforms to augment data
    range_x = [-15.0 / 180 * np.pi, 15.0 / 180 * np.pi]
    data_augmentation_transforms = [
        RandRotated(
            keys=["image", "label"],
            range_x=range_x,
            range_y=0,
            range_z=0,
            mode=["bicubic", "nearest"],
            padding_mode="zeros",
            prob=0.2,
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=0.7,
            max_zoom=1.4,
            mode=["bicubic", "nearest"],
            padding_mode="constant",
            align_corners=(True, None),
            prob=0.2,
        ),
        RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
            prob=0.15,
        ),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.15),
        RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    ]
    
    print("""
    Transforms for image augmentation : 
    -----------------------------------
    RandRotated
    RandZoomd
    RandFlipd
    RandGaussianNoised
    RandGaussianSmoothd
    RandScaleIntensityd
    RandAdjustContrastd
    """)

    # Define transforms for training, validation and testing
    train_transforms = Compose(load_transforms + data_augmentation_transforms)
    val_transforms = Compose(load_transforms)
    test_transforms = Compose(load_transforms)
    
    return train_transforms, val_transforms, test_transforms
