import torch
import numpy as np
from torch.utils import data
from torchvision import transforms

# own routines
import utils


class Dataset(data.Dataset):
    """
    Characterizes a dataset for PyTorch
    see also https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, ):
        """
        Initialization Args:
        state here what the arguments are
        we generate the data for the toy model with three samples. 
        """

        ##define inputs and their labels
        #repeat the array of temperature to have the same vector size for input and labels
        inputs = torch.linspace(0.0, 1.0, 51)[1:]
        self.inputs = torch.zeros(50 * 6 + 1)
        self.inputs[:50] = inputs

        #1 additional element for penalizing negative spectral conductivity
        self.labels = torch.zeros(50 * 6 + 1)

        #integration cutoff
        Wc = 5.0

        #chemical potential
        muA = 0.5 * inputs**2
        muB = 0.2 - 1.5 * inputs**2
        muC = 0.3 + 1. * inputs**2

        #generate L11 and L12 from toy model
        for i in range(inputs.shape[0]):
            ##integrate sigma(E) between -Wc*x[j]+mu and Wc*x[j]+mu.
            #define toy model
            newE = torch.linspace((-Wc * inputs[i]).item(),
                                  (Wc * inputs[i]).item(), 101)
            newEA = newE + muA[i]
            newEB = newE + muB[i]
            newEC = newE + muC[i]
            newYA = 2. * torch.exp(-((newEA - 2))**2) + torch.exp(-(
                (newEA + 2))**2)
            newYB = 2. * torch.exp(-((newEB - 2))**2) + torch.exp(-(
                (newEB + 2))**2)
            newYC = 2. * torch.exp(-((newEC - 2))**2) + torch.exp(-(
                (newEC + 2))**2)

            #calculate L11 and L12 for each sample
            newfA_1 = newYA * torch.exp((newE) / inputs[i]) / (torch.exp(
                (newE) / inputs[i]) + 1)**2 / inputs[i]
            newfA_2 = (newE) * newYA * torch.exp(
                (newE) / inputs[i]) / (torch.exp(
                    (newE) / inputs[i]) + 1)**2 / inputs[i]
            newfB_1 = newYB * torch.exp((newE) / inputs[i]) / (torch.exp(
                (newE) / inputs[i]) + 1)**2 / inputs[i]
            newfB_2 = (newE) * newYB * torch.exp(
                (newE) / inputs[i]) / (torch.exp(
                    (newE) / inputs[i]) + 1)**2 / inputs[i]
            newfC_1 = newYC * torch.exp((newE) / inputs[i]) / (torch.exp(
                (newE) / inputs[i]) + 1)**2 / inputs[i]
            newfC_2 = (newE) * newYC * torch.exp(
                (newE) / inputs[i]) / (torch.exp(
                    (newE) / inputs[i]) + 1)**2 / inputs[i]
            self.labels[i] = torch.trapz(newfA_1, newEA)
            self.labels[i + 50] = -torch.trapz(newfA_2, newEA)
            self.labels[i + 100] = torch.trapz(newfB_1, newEB)
            self.labels[i + 150] = -torch.trapz(newfB_2, newEB)
            self.labels[i + 200] = torch.trapz(newfC_1, newEC)
            self.labels[i + 250] = -torch.trapz(newfC_2, newEC)

        self.total_num_of_samples = self.inputs.shape[0]

    def __len__(self):
        # Denotes the total number of samples
        return self.total_num_of_samples

    def __getitem__(self, index):
        # return batches that are associated with 'index'
        X = self.inputs[index].view(
            -1)  # view -1 ensures right input dimensions for network
        Y = self.labels[index].view(-1)
        return X, Y
