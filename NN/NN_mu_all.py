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

    def forward(self, x, Wc, Nsample):
        #x is input of NN, which is $\xi_i=T_i/T_N$ for i=1,2,...,N
        #Wc is cutoff for energy integrals
        #Nsample means the number of samples, i.e., Nsample=Ns
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        size = x.shape[0]
        energy = (torch.linspace(-Wc * 2, Wc * 2, size * 4 + 1)).to(device)
        out = torch.zeros(2 * size * Nsample + 1).to(device)

        Q = self.fc_layers(x)[:size * 4 + 10]
        # averaging 10 consequtive elements of the output vector to compute the spectral conductivity
        Sigma = torch.zeros(size * 4 + 1).to(device)
        for j in range(Sigma.shape[0]):
            Sigma[j] = (Q[j] + Q[j + 1] + Q[j + 2] + Q[j + 3] + Q[j + 4] +
                        Q[j + 5] + Q[j + 6] + Q[j + 7] + Q[j + 8] +
                        Q[j + 9]) / 10.0

        #prepare interpolation of sigma(E) for energy integrals
        gi2 = torch_interpolations.RegularGridInterpolator([energy], Sigma)

        #parameters for chemical potential
        chem_Q = self.fc_layers(x)[size * 4 + 10:]

        for i in range(Nsample):
            for j in range(size):
                #chemical potential for each sample
                chem = chem_Q[i * 3] + 0 * chem_Q[i * 3 + 1] * x[j] + chem_Q[
                    i * 3 + 2] * x[j]**2
                chem = torch.atan(chem) / (np.pi / 2.0) * Wc

                #interpolate sigma(E) between -Wc*x[j]+mu and Wc*x[j]+mu.
                newE = torch.linspace((-Wc * x[j]).item(), (Wc * x[j]).item(),
                                      101).to(device)
                newE = newE + chem
                newSigma = gi2([newE])
                newE = newE - chem

                #calculate L11 and L12
                newf1 = newSigma * torch.exp((newE) / x[j]) / (torch.exp(
                    (newE) / x[j]) + 1)**2 / x[j]
                newf2 = newE * newSigma * torch.exp(
                    (newE) / x[j]) / (torch.exp((newE) / x[j]) + 1)**2 / x[j]
                out[j + size * (2 * i)] = torch.trapz(newf1, newE)
                out[j + size * (2 * i + 1)] = -torch.trapz(newf2, newE)

        ## penalize the negative spectral conductivity
        out[-1] = torch.sum(torch.min(Sigma, torch.zeros(1).to(device)))

        return out
