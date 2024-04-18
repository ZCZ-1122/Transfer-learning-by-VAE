"""Defines the neural network, losss function and metrics"""
import numpy as np
import torch
import torch.nn.functional as F

from .types_ import *
from torch import nn
from abc import abstractmethod

from cs230.dice_loss import multiclass_dice_coeff

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 output_channel : int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        # hidden_dims.reverse()
        rev_hidden_dims = hidden_dims[::-1]

        for i in range(len(rev_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(rev_hidden_dims[i],
                                       rev_hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(rev_hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(rev_hidden_dims[-1],
                                               rev_hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(rev_hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(rev_hidden_dims[-1], out_channels=output_channel,
                                      kernel_size= 3, padding= 1),
                            # nn.Tanh(),
                            nn.Sigmoid(),
                            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return {'x_hat' : self.decode(z),
                'z' : z,
                'mu': mu,
                'logvar' : log_var,
        }

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



def loss_fn(outputs, originals, kld_weight, return_mse_dkl=False):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = outputs['x_hat']
        mu = outputs['mu']
        log_var = outputs['logvar']

        recons_loss = F.mse_loss(recons, originals)
        
        kld_loss_vec = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(),dim = 1) # shape (batch_size,)
        kld_loss = torch.mean(kld_loss_vec, dim = 0)
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss

        if not return_mse_dkl:
            return loss
        else:
            return {'loss': loss, 
                    'Reconstruction_Loss' : recons_loss.detach(), 
                    'KLD' : - kld_loss.detach()}

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) shape (N,1) - logits of the model
        labels: (np.ndarray) shape (N,1)  - ground truth

    Returns: (float) accuracy in [0,1]
    """
    pass

def mse(outputs, labels):
    """
    Compute the average of squared l2 norm.

    Args:
        outputs: (np.ndarray) shape (N,1) - logits of the model
        labels: (np.ndarray) shape (N,1)  - ground truth

    Returns: (float)
    """
    return np.mean((outputs - labels)**2)


def multiclass_dice(logits_batch, label_batch):
    
    n_classes = torch.unique(label_batch).size()[0]
    
    # Convert dense matrix to one-hot representation
    label_batch = F.one_hot(label_batch, n_classes).squeeze()
    label_batch = label_batch.permute(0, 3, 1, 2).float() # (N,H,W,C) -> (N,C,H,W)
    
    pred_batch = F.one_hot(logits_batch.argmax(dim=1), n_classes)
    pred_batch = pred_batch.permute(0, 3, 1, 2).float()
    
    return multiclass_dice_coeff(pred_batch[:, 1:], label_batch[:, 1:], # background excluded
                                 reduce_batch_first=False).item()

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'dice': multiclass_dice,
    # could add more metrics such as accuracy for each token type
}
