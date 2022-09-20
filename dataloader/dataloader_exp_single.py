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

    def __init__(self, x, y):
        """
        Initialization Args:
        x,y : parameters for experimental data
        """

        temp=np.loadtxt('./source_data/L11_L12_'\
                        +str(int(x*1000))+'_'+str(int(y*1000))+'.txt')
        #number of data points in temperatures
        size = temp[0].shape[0]
        #L11, L12
        self.inputs = torch.zeros(size * 2)
        self.labels = torch.zeros(size * 2)

        ##define inputs and their labels
        tempp = np.zeros((2, size * 2))
        tempp[0, :size] = temp[0]
        #load L11 and L12
        tempp[1, :size] = temp[1]
        tempp[1, size:2 * size] = temp[2]
        #convert numpy to torch
        tens = torch.from_numpy(tempp).float()

        # store inputs and their labels
        self.inputs = tens[0]
        self.labels = tens[1]

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
