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
        x,y : parameters for experimental data
        """
        xarray = np.array([0.0, 0.0, 0.001, 0.002, 0.01, 0.02, 0.05])
        yarray = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

        #number of sample and number of data points in temperatures
        sample = xarray.size
        size = 41
        #L11, L12*sample + 1 for negative elements
        self.inputs = torch.zeros(size * sample * 2 + 1)
        self.labels = torch.zeros(size * sample * 2 + 1)

        for i in range(sample):
            temp=np.loadtxt('./source_data/L11_L12_'\
                            +str(int(xarray[i]*1000))+'_'+str(int(yarray[i]*1000))+'.txt')
            tempp = np.zeros((2, temp.shape[1] * 2))  # L11, L12
            tempp[0, :temp.shape[1]] = temp[0]
            tempp[0, temp.shape[1]:] = temp[0]
            tempp[1, :temp.shape[1]] = temp[1]
            tempp[1, temp.shape[1]:] = temp[2]
            tens = torch.from_numpy(tempp).float()
            self.inputs[size * 2 * i:size * 2 * (i + 1)] = tens[0]
            self.labels[size * 2 * i:size * 2 * (i + 1)] = tens[1]

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
