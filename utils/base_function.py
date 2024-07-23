from torch import nn
from torch.nn import init
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math

def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False

def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True
    
def weights_init(w,init_type="normal", gain=0.02):
    classname = w.__class__.__name__        
    if hasattr(w, 'weight') and (classname.find('Conv')!=-1 or classname.find('Linear')!=-1):
        if init_type == 'normal':
            init.normal_(w.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            init.xavier_normal_(w.weight.data, gain=gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(w.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(w.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(w, 'bias') and w.bias is not None:
            init.constant_(w.bias.data, 0.0)   
            
class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_metric):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        elif gan_mode in ['wgangp','wgandiv']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real, is_disc=False):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            labels = (self.real_label if target_is_real else self.fake_label).expand_as(prediction).type_as(prediction)
            loss = self.loss(prediction, labels)
        elif self.gan_mode in ['hinge', 'wgangp', 'wgandiv']:
            if is_disc:
                if target_is_real:
                    prediction = -prediction
                if self.gan_mode == 'hinge':
                    loss = self.loss(1 + prediction).mean()
                elif self.gan_mode in ['wgangp','wgandiv']:
                    loss = prediction.mean()
            else:
                loss = -prediction.mean()

        return loss
    
def criterion_psnr(tensor1, tensor2, data_range=3):
    mse = F.mse_loss(tensor1, tensor2)
    psnr = 10 * math.log10((data_range ** 2) / mse.item())
    return psnr