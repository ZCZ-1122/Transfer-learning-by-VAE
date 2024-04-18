import matplotlib.pyplot as plt
import numpy as np


def test_img_aug(ix, dataset, s=5):
    """
    show the augmented image dataset[ix] and the associated label
    """
    # each time we access the image, random image augmentataion transforms
    # may be applied to the image
    aug_sample = dataset[ix]

    # Extract the image from the sample
    augmented_image = aug_sample["image"].numpy().squeeze()
    augmented_label = aug_sample["label"].numpy().squeeze()

    # Plot the image
    fig, ax = plt.subplots(1,2, figsize=(2*s,1*s))
    ax[0].imshow(augmented_image, cmap='gray')
    title = f'augmented_img_{ix}\n'
    title += f'min={augmented_image.min():.2f}, max={augmented_image.max():.2f}'
    ax[0].set_title(title)

    ax[1].imshow(augmented_label, cmap='gray')
    title = f'augmented_label_{ix}\n'
    title += f'min={augmented_label.min():.2f}, max={augmented_label.max():.2f}'
    ax[1].set_title(title)