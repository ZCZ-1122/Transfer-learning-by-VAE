import os
import sys
FOLDERNAME = os.getcwd()
sys.path.append(FOLDERNAME)
os.chdir(FOLDERNAME)
print(f'Current folder changed to {os.getcwd()}')

import numpy as np
import time
import logging
from scipy.spatial.distance import mahalanobis

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from monai.data import CacheDataset

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots

from sklearn.manifold import TSNE

from cs230 import utils, data, visu
from cs230.evaluate import predict
from cs230.autoencoder import VariationalAutoencoder2d

################################################################
#   Some simple functions to increase code readability         #
################################################################
def set_torch_seed(device, seed):
    torch.manual_seed(seed)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(seed)

def side_by_side_box_plots(data_left, data_right, save_path=None):
    fig, ax = plt.subplots(1,2, figsize=(1.5*2,4*1))
    ax = ax.ravel()

    ax[0].boxplot(data_left)
    ax[0].grid()
    ax[0].set_title(r'Box plot of $\bar{\mu}\in \mathbb{R}^{\text{N}}$')

    ax[1].boxplot(data_right)
    ax[1].grid()
    ax[1].set_title(r'Box plot of $\bar{\sigma}\in \mathbb{R}^{\text{N}}$')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.close()

def t_sne_visualization(data, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(data)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    plt.title('t-SNE Visualization of latent representation')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def show_two_MD_distributions(MD_camus, MD_sumac, save_path):
    # Shapiro tests
    MD_camus_is_normal = utils.interpreate_shapiro_test(MD_camus, verbose=0)
    MD_samuc_is_normal = utils.interpreate_shapiro_test(MD_sumac, verbose=0)
    
    # Show MD distribution
    fig, ax = plt.subplots(figsize=(5,5))
    title = 'MD distribution \n'
    title += 'MD_camus passed Shapiro test\n' if MD_camus_is_normal else 'MD_camus did not passe Shapiro test\n'
    title += 'MD_sumac passed Shapiro test' if MD_samuc_is_normal else 'MD_sumac did not passe Shapiro test'
    visu.plot_hist_with_kde(MD_camus, ax=ax, label_prefix='camus', show_hist=True)
    visu.plot_hist_with_kde(MD_sumac, ax=ax, label_prefix='sumac',show_hist=True, title=title)
    
    fig.savefig(save_path, dpi=300)
    plt.close()

def save_str_list_to_txt(str_list, file_path):
    with open(file_path, "w") as file:
        # Write each string in the list to the file
        for string in str_list:
            file.write(string + "\n")    


################################################################
#                             Main                             #
################################################################
if __name__ == "__main__":
    print('hihihhi')
    
    glb_tic = time.time()
    
    ############################################################
    #                Parameters to specify                     #
    ############################################################
    result_dir = './results/20231215_222202_seed230_run-2'
    
    data_dir_camus = '../dataset/segmentation/'
    dataset_dir_sumac = "../dataset/sumac_256/"
    
    model_path = "./experiments/20231210-vae_second_try/20231210-vae-mse003.pth"
    PERC_THRE = 99 # percentile threshold to determine outliers
    
    ############################################################
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = 64
    print('\nModel Hyperparameters : ')
    print('---------------------------')
    params_dict = {
                # Data
                'IMG_SIZE' : IMG_SIZE,
                'batch_size' : 64,
                
                # model
                'input_shape' : (1, IMG_SIZE, IMG_SIZE),
                'blocks' : 3,
                'init_channels' : 64,
                'latent_dim' : 256,
                'activation' : "ELU",
                'use_batchnorm' : True,
                
                'dtype' : torch.float32,
                'device' : device,
    }
    args = utils.Params(params_dict)
    os.makedirs(result_dir, exist_ok=True)

    for k,v in args.__dict__.items():
        print(f'{k:20} : {v}')
    
    ################################################################################################################################

    ######################################### Load CAMUS #########################################
    assert args.IMG_SIZE == 64, "the model only supports 64x64 pixel images"
    
    print('\nLoading CAMUS dataset...')
    train_datalist = data.create_datalist(data_dir_camus, 'train')
    val_datalist = data.create_datalist(data_dir_camus, 'valid')
    test_datalist = data.create_datalist(data_dir_camus, 'test')
    print(f'training sample number   : {len(train_datalist)}')
    print(f'validation sample number : {len(val_datalist)}')
    print(f'testing sample number    : {len(test_datalist)}')
    
    # Get transforms for training, validation and testing
    trans_train, trans_val, trans_test = data.get_load_and_augmentation_transforms(args.IMG_SIZE)

    # Datasets
    print('\nCreate pytorch Datasets for CAMUS...')
    train_ds = CacheDataset(data=train_datalist, transform=trans_train, cache_rate=1.0)
    # val_ds = CacheDataset(data=val_datalist, transform=trans_val, cache_rate=1.0)
    # test_ds = CacheDataset(data=test_datalist, transform=trans_test, cache_rate=1.0)
    print('Done\n')
    
    # Data loaders
    train_loader_camus = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)
    # val_loader_camus = DataLoader(dataset=val_ds, batch_size=args.batch_size, shuffle=False)
    # test_loader_camus = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False)
    
    ######################################### Load the model #########################################
    # Initiciate the model
    model = VariationalAutoencoder2d(args.input_shape, args.blocks, args.init_channels,
                                    args.latent_dim, activation=args.activation,
                                    use_batchnorm=args.use_batchnorm)
    model = model.to(device=args.device)
    print('Model (VAE) iniciated')

    # Print the summary of the network
    summary_kwargs = dict(
                            col_names = ["input_size", "output_size", "num_params"],
                            depth = 3,
                            verbose = 0,
    )
    input_shape = (1, 1, args.IMG_SIZE, args.IMG_SIZE)
    print('\nShowing model architecture...')
    print(summary(model, input_shape, **summary_kwargs))
    
    # Load weights
    model.load_state_dict(torch.load(model_path))
    print(f'Model weights loaded from {model_path}\n')
    
    ######################################### Inference CAMUS #########################################
    print('\nEncoding CAMUS images...')
    
    set_torch_seed(args.device, seed=230)
    out_dict_camus = predict(train_loader_camus, model, args)

    # Check mu and var (l1 error and Boxplots)
    avg_mu_camus = np.mean(out_dict_camus['all_mu'], axis=0)
    all_var_camus = np.exp(out_dict_camus['all_logvar'])
    avg_var_camus = np.mean(all_var_camus, axis=0)
    print('CAMUS average l1 error')
    print(f'Average l1 error of mu_bar  : {np.abs(avg_mu_camus- 0).mean()}')
    print(f'Average l1 error of var_bar : {np.abs(avg_var_camus- 1).mean()}\n')
    
    path = os.path.join(result_dir, 'CAMUS_average_mu_var_box_plots.png')
    side_by_side_box_plots(avg_mu_camus, avg_var_camus, save_path=path)
    print(f'Box plots of average mu and var saved to {path}\n')
    
    # Show some reconstructed images
    path = os.path.join(result_dir, 'examples_of_reconstruction.png')
    visu.show_reconstruction(out_dict_camus['all_imgs'], out_dict_camus['all_x_hat'], nb_picked=4, 
                            random_show=True, save=True, save_path=path)
    print(f'Figures of examples of reconstruction saved to {path}\n')
    
    # Show latent space by tSNE
    path = os.path.join(result_dir, 'latent_space_visualization.png')
    t_sne_visualization(out_dict_camus['all_z'], path)
    print(f'tSNE visualization of latent space saved to {path}\n')
    
    ######################################### Load SUMAC #########################################
    print('\n Loading SUMAC dataset...')
    sumac_dataset_dict = data.load_SUMAC_dataset(dataset_dir_sumac, img_size=args.IMG_SIZE, return_paths=True,
                                              extension="nii.gz",random_state=42)
    X_sumac = sumac_dataset_dict['X']
    print(f'X_sumac shape : {X_sumac.shape}')
    
    shuffle = False
    assert not shuffle,"Do not shuffle when creating dataloader to ensure consistency"
    sumac_dataset = data.sumacDataset(X_sumac)
    sumac_loader = DataLoader(dataset=sumac_dataset, batch_size=args.batch_size,
                              shuffle=shuffle)
    
    ######################################### Inference SUMAC #########################################
    print('\nEncoding CAMUS images...')
    
    set_torch_seed(args.device, seed=230)
    out_dict_sumac = predict(sumac_loader, model, args)

    avg_mu_sumac = np.mean(out_dict_sumac['all_mu'], axis=0)
    all_var_sumac = np.exp(out_dict_sumac['all_logvar'])
    avg_var_sumac = np.mean(all_var_sumac, axis=0)

    print('\nSUMAC Average l1 error : ')
    print(f'Average l1 error of mu_bar  : {np.abs(avg_mu_sumac- 0).mean()}')
    print(f'Average l1 error of var_bar : {np.abs(avg_var_sumac- 1).mean()}\n')
    
    
    ######################################### MD calculation #########################################
    all_z_camus = out_dict_camus['all_z']
    all_z_sumac = out_dict_sumac['all_z']
    all_imgs_sumac = out_dict_sumac['all_imgs']
    

    avg_z_camus = np.mean(all_z_camus, axis=0)
    cov_z_camus = np.cov(all_z_camus, rowvar=False) # cov_z_camus = np.diag(avg_var_camus) WRONG BUT BETTER RESULT?
    cov_z_camus_inv = np.linalg.inv(cov_z_camus)

    # MD_camus
    MD_camus = np.array([mahalanobis(z_c, avg_z_camus, cov_z_camus_inv) for z_c in all_z_camus])
    utils.log('MD_camus', MD_camus)
    avg_MD_camus = MD_camus.mean()
    print(f'Average MD of Camus : {avg_MD_camus}\n')

    # MD_SUMAC
    MD_sumac = np.array([mahalanobis(z_s, avg_z_camus, cov_z_camus_inv) for z_s in all_z_sumac])
    utils.log('MD_sumac', MD_sumac)
    
    # Show two MD distributions (Histogram and KDE)
    path = os.path.join(result_dir, 'MD_distributions.png')
    show_two_MD_distributions(MD_camus, MD_sumac, save_path=path)
    print(f'MD distribution plot saved to {path}\n')
    
    # Find outliers
    percentile_value = np.percentile(MD_sumac, PERC_THRE)
    outlier_ixs = np.where(MD_sumac >= percentile_value)[0]
    outlier_imgs = all_imgs_sumac[outlier_ixs]
    print(f'outlier_imgs shape : {outlier_imgs.shape} ')
    
    # Show outlier images
    path = os.path.join(result_dir, f'Outliers_images_with_percentile_{PERC_THRE}.png')
    visu.show_images(outlier_imgs, s=3, cs=5, save_path=path)
    print(f'Outliers_images_with_percentile_{PERC_THRE} saved to {path}\n')
    
    # Save outlier image paths and associated label paths to txt
    outlier_paths = [sumac_dataset_dict['image_paths'][ix] for ix in outlier_ixs]
    txt_path = os.path.join(result_dir, 'outlier_image_paths.txt')
    save_str_list_to_txt(outlier_paths, txt_path)
    print(f'outlier image paths saved to {txt_path}\n')
    
    outlier_label_paths = [sumac_dataset_dict['label_paths'][ix] for ix in outlier_ixs]
    txt_path = os.path.join(result_dir, 'outlier_image_label_paths.txt')
    save_str_list_to_txt(outlier_label_paths, txt_path)
    print(f'outlier label paths saved to {txt_path}\n')

    # Find "inliers"
    percentile_value = np.percentile(MD_sumac, 100-PERC_THRE)
    inlier_ixs = np.where(MD_sumac <= percentile_value)[0]
    inlier_imgs = all_imgs_sumac[inlier_ixs]
    print(f'inlier_imgs shape : {inlier_imgs.shape}')

    path = os.path.join(result_dir, f'Inlier_images_with_percentile_{100-PERC_THRE}.png')
    visu.show_images(inlier_imgs, s=3, cs=5, save_path=path)
    print(f'Inlier_images_with_percentile_{100-PERC_THRE} saved to {path}\n')
    
    glb_tok = time.time()
    print(f'\n*** The whole process took {(glb_tok-glb_tic):.3f} seconds***\n')