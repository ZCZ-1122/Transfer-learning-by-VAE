import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from typing import Optional, Union

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import color
import colorsys
import random
from torch import Tensor

from scipy.stats import gaussian_kde

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
    
    
def plot_overlay_segmentation(x, mask, spacing=(1,1), alpha=0.3, bright=True,
                              ax=None, title="", colors=None):
    """_summary_

    Args:
        x : ndarray of shape (H,W,C)
        mask : ndarray of shape (H,W)
        axarr : ndarray, shape : (2,), dtype : matplotlib.axes._axes.Axes
    """
    asp = spacing[0]/spacing[1]
    
    mask = np.array(mask, dtype='uint8') # np.argmax(y[i],axis=-1)
    H,W = mask.shape
    
    # RGB image with transparency
    true_color = np.zeros((H, W, 4), dtype='float32') 

    if colors is None:
        n_class = len(np.unique(mask))
        colors = [ c + (alpha, ) for c in random_colors(n_class, bright)]
    for i, c in enumerate(colors):
        true_color[mask == (i+1)] = c

    if ax is None:
        _, ax = plt.subplots(figsize=(5,5))

    ax.imshow(x.squeeze(), 'gray', interpolation='none', aspect=asp)
    ax.imshow(true_color, interpolation='none', aspect=asp)
    ax.set_title(title)


def imagesc(
    ax: matplotlib.axes,
    image: Union[Tensor, np.ndarray],
    title: Optional[str] = None,
    colormap: matplotlib.colormaps = plt.cm.gray,
    clim: Optional[tuple[float, float]] = None,
    show_axis: bool = False,
    show_colorbar: bool = True,
    **kwargs,
) -> None:
    """Display image with scaled colors. Similar to Matlab's imagesc.

    Args:
        ax: Axis to plot on.
        image: Array to plot.
        title: Title of plotting.
        colormap: Colormap of plotting.
        clim: Colormap limits.
        show_axis: Whether to show axis when plotting.
        show_colorbar: Whether to show colorbar when plotting.
        **kwargs: Keyword arguments to be passed to `imshow`.

    Example:
        >>> plt.figure("image", (18, 6))
        >>> ax = plt.subplot(1, 2, 1)
        >>> imagesc(ax, np.random.rand(100,100), "image", clim=(-1, 1))
        >>> plt.show()
    """

    if clim is not None and isinstance(clim, (list, tuple)):
        if len(clim) == 2 and (clim[0] < clim[1]):
            clim_args = {"vmin": float(clim[0]), "vmax": float(clim[1])}
        else:
            raise ValueError(
                f"clim should be a list or tuple containing 2 floats with clim[0] < clim[1], "
                f"got {clim} instead.",
            )
    else:
        clim_args = {}

    if isinstance(image, Tensor):
        image = image.cpu().detach().numpy()

    im = ax.imshow(image, colormap, **clim_args, **kwargs)
    plt.title(title)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im, cax)

    if not show_axis:
        ax.set_axis_off()


def overlay_mask_on_image(
    image: np.ndarray,
    segmentation: np.ndarray,
    bg_label: int = 0,
    alpha: float = 0.3,
    colors: Optional[Union[list, list[list]]] = None,
) -> np.ndarray:
    """Overlay segmentation mask on given image.

    Args:
        image: Image to overlay.
        segmentation: Segmentation mask.
        bg_label: Label of background.
        alpha: Opacity of colorized labels.
        colors: Colors of overlay of the segmentation labels.

    Returns:
        RGB Numpy array of image overlaid with segmentation mask.

    Raises:
        ValueError: When image and segmentation have different shapes.
    """
    if not np.all(image.shape == segmentation.shape):
        raise ValueError(
            f"image {image.shape} does not have the same dimension as segmentation {segmentation.shape}!"
        )

    return color.label2rgb(segmentation, image, bg_label=bg_label, alpha=alpha, colors=colors)


def plot_hist_with_kde(data, ax=None, figsize=(5,5), show_hist=True,
                       show_kde=True, label_prefix='', title=""):
    """
    Args:
        data : 1D numpy array
    """
    
    kde = gaussian_kde(data) # compute kde
    x_plot = np.linspace(min(data), max(data), 1000)
    kde_values = kde(x_plot)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if show_hist:
        ax.hist(data, bins='auto', density=True, alpha=0.5, label=f'{label_prefix}_Histogram')
    if show_kde:
        ax.plot(x_plot, kde_values, label=f'{label_prefix}_KDE')
    # Add labels and a legend
    ax.grid()
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    
    

def show_images(imgs, cs=2, s=5, titles=None, cmap='gray', save_path=None):
    """
    imgs : ndarray of shape (N, C, H, W)
    """
    
    N = len(imgs)
    rs = ( N + cs-1) // cs
    fig, ax = plt.subplots(rs, cs, figsize=(cs * s, rs * s))
    ax = ax.ravel()
    
    if titles is None:
        titles = [''] * N
    for a, img, title in zip(ax, imgs, titles):
        a.imshow(img.squeeze(), cmap=cmap)
        a.set_title(title)
    
    if save_path:
        fig.savefig(save_path,dpi=300)
        plt.close()


def show_save_imgs(imgs, cs=4, save=False, save_path=None, s=5):
    
    rs = (len(imgs)+cs-1) // cs
    fig, ax = plt.subplots(rs, cs, figsize=(cs*s,rs*s))
    ax = ax.ravel()
    for i, (img, a) in enumerate(zip(imgs, ax)):
        a.imshow(img, cmap='gray')
        a.set_title(f'sample_{i}')
    
    if save:
        fig.savefig(save_path, dpi=300)
        plt.close()


def show_reconstruction(imgs, recons_imgs, nb_picked=4, random_show=False,
                        save=True, save_path=None):
    """
    imgs        : np array of shape (batch_size, C, H, W)
    recons_imgs : np array of shape (batch_size, C, H, W)
    """
    if random_show:
        ixs = np.random.choice(len(imgs), nb_picked)
    else:
        ixs = np.arange(nb_picked)
    picked_imgs = imgs[ixs].squeeze()
    recons_imgs = recons_imgs[ixs].squeeze()

    # plot gt vs reconstruction
    cs = 4
    rs = (nb_picked*2+cs-1) // cs
    fig, ax = plt.subplots(rs,cs,figsize=(cs*5,rs*5))
    ax = ax.ravel()

    for i in range(nb_picked):
        ax[2*i].imshow(picked_imgs[i], cmap='gray')
        ax[2*i].set_title(f'gt_{ixs[i]}')

        ax[2*i+1].imshow(recons_imgs[i],cmap='gray')
        ax[2*i+1].set_title(f'recons_{ixs[i]} (mse = {((picked_imgs[i] - recons_imgs[i])**2).mean():.3f})')
    
    if save:
        fig.savefig(save_path, dpi=300)
        plt.close()

def imagesc(
    ax: matplotlib.axes,
    image: Union[Tensor, np.ndarray],
    title: Optional[str] = None,
    colormap: matplotlib.colormaps = plt.cm.gray,
    clim: Optional[tuple[float, float]] = None,
    show_axis: bool = False,
    show_colorbar: bool = True,
    **kwargs,
) -> None:
    """Display image with scaled colors. Similar to Matlab's imagesc.

    Args:
        ax: Axis to plot on.
        image: Array to plot.
        title: Title of plotting.
        colormap: Colormap of plotting.
        clim: Colormap limits.
        show_axis: Whether to show axis when plotting.
        show_colorbar: Whether to show colorbar when plotting.
        **kwargs: Keyword arguments to be passed to `imshow`.

    Example:
        >>> plt.figure("image", (18, 6))
        >>> ax = plt.subplot(1, 2, 1)
        >>> imagesc(ax, np.random.rand(100,100), "image", clim=(-1, 1))
        >>> plt.show()
    """

    if clim is not None and isinstance(clim, (list, tuple)):
        if len(clim) == 2 and (clim[0] < clim[1]):
            clim_args = {"vmin": float(clim[0]), "vmax": float(clim[1])}
        else:
            raise ValueError(
                f"clim should be a list or tuple containing 2 floats with clim[0] < clim[1], "
                f"got {clim} instead.",
            )
    else:
        clim_args = {}

    if isinstance(image, Tensor):
        image = image.cpu().detach().numpy()

    im = ax.imshow(image, colormap, **clim_args, **kwargs)
    plt.title(title)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(im, cax)

    if not show_axis:
        ax.set_axis_off()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds,
                                       ax=None,figsize=(10,8)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
    ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
    ax.grid()
    ax.legend()
    ax.set_xlabel('thresholds')
    ax.set_title('precision recall vs threshold')



def plot_precision_vs_recall(precisions, recalls,
                             ax=None,figsize=(10,8)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recalls[:-1], precisions[:-1],'m-')
    ax.grid()
    ax.set_xlabel('recalls')
    ax.set_ylabel('precisions')
    ax.set_title('precision vs recall')


def plot_attributions(data_arr, ixs, importance_scores, preds, labels, ax=None):
    set_fig = False
    if ax is None:
        fig, ax = plt.subplots(len(ixs),1, figsize=(12,3*len(ixs)))
        set_fig = True
        
    ax = ax.ravel()

    for i, (a, ix) in tqdm(enumerate(zip(ax, ixs))):
        a.scatter(range(len(data_arr[i])), data_arr[i], 
                  s=60 * importance_scores[i], marker='o')
        a.plot(data_arr[i], '--', lw=0.5)
        a.set_title(f'ix = {ix}, pred = {preds[i]}, gt = {labels[i]}')
        a.grid()
    
    if set_fig:
        fig.tight_layout()

