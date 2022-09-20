import torch
import torch.nn as nn
import numpy as np
from scipy import integrate

import torch_interpolations
#The source of this module is originally from https://github.com/sbarratt/torch_interpolations


class FCN(nn.Module):
    # A network with fully-connected (dense) layers
    def __init__(self, fcs):
        super(FCN, self).__init__()

        # Define activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()

        # Relu is the activation function
        self.actfunc = self.relu

        # define network
        # fc layers
        fc_layers = []
        n_fc = len(fcs) - 1
        for i in range(n_fc):
            fc_layers.extend([nn.Linear(fcs[i], fcs[i + 1]), self.actfunc])
        # remove the last activation function
        fc_layers.pop()

        self.fc_layers = nn.Sequential(*fc_layers)

        # to define coefficients of functions that are potentially helpful,
        # see https://pytorch.org/docs/stable/nn.html?highlight=nn%20parameter#torch.nn.Parameter

        self.parameter1 = nn.Parameter(torch.randn(1, ) * 0.0,
                                       requires_grad=True)

    def func1(self, x):
        #define a function here if needed
        return self.parameter1 * torch.sin(x)

    def forward(self, x, Wc):
        #x is input of NN, which is $\xi_i=T_i/T_N$ for i=1,2,...,N
        #Wc is cutoff for energy integrals
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        size = x.shape[0]
        energy = (torch.linspace(-Wc, Wc, 2 * size + 1)).to(device)
        out = torch.zeros(2 * size).to(device)

        Q = self.fc_layers(x)
        # averaging 5 consequtive elements of the output vector to compute the spectral conductivity
        Sigma = torch.zeros(2 * size + 1).to(device)
        for j in range(Sigma.shape[0]):
            Sigma[j] = (Q[j] + Q[j + 1] + Q[j + 2] + Q[j + 3] + Q[j + 4] +
                        Q[j + 5] + Q[j + 6] + Q[j + 7] + Q[j + 8] +
                        Q[j + 9]) / 10.0

        #prepare interpolation of sigma(E) for energy integrals
        gi2 = torch_interpolations.RegularGridInterpolator([energy], Sigma)
        for i in range(size):
            #interpolate sigma(E) between -Wc*x[j]+mu and Wc*x[j]+mu. chemical potential is set to be zero.
            newE = torch.linspace((-Wc * x[i]).item(), (Wc * x[i]).item(),
                                  101).to(device)
            newSigma = gi2([newE])

            #calculate L11 and L12
            newf1 = newSigma * torch.exp((newE) / x[i]) / (torch.exp(
                (newE) / x[i]) + 1)**2 / x[i]
            newf2 = newE * newSigma * torch.exp((newE) / x[i]) / (torch.exp(
                (newE) / x[i]) + 1)**2 / x[i]
            out[i] = torch.trapz(newf1, newE)
            out[i + size] = -torch.trapz(newf2, newE)

        return out
