
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
        state here what the arguments are.
        we generate the data for the toy model with a single sample. 
        """
        
        ##define inputs and their labels
        #repeat the array of temperature to have the same vector size for input and labels
        inputs = torch.linspace(0.0, 1.0, 51)[1:]
        self.inputs=torch.zeros(50*2)
        self.inputs[:50]=inputs
        
        #L11, L12
        self.labels=torch.zeros(50*2)
       
        #integration cutoff
        Wc=5.0

        #chemical potential
        muA=0.0  
        
        #generate L11 and L12 from toy model
        for i in range(inputs.shape[0]):
            ##integrate sigma(E) between -Wc*x[j]+mu and Wc*x[j]+mu.   
            #define toy model              
            newE=torch.linspace((-Wc*inputs[i]).item(),(Wc*inputs[i]).item(),101)
            newE=newE+muA
            newSigma=2.*torch.exp(-((newE-2))**2)+torch.exp(-((newE+2))**2)
            newE=newE-muA
            
            #calculate L11 and L12 
            newfA_1=newSigma*torch.exp((newE)/inputs[i])/(torch.exp((newE)/inputs[i])+1)**2/inputs[i]
            newfA_2=(newE)*newSigma*torch.exp((newE)/inputs[i])/(torch.exp((newE)/inputs[i])+1)**2/inputs[i]
            self.labels[i]=torch.trapz(newfA_1,newE)
            self.labels[i+50]=-torch.trapz(newfA_2,newE)            
            
        self.total_num_of_samples = self.inputs.shape[0]

    def __len__(self):
        # Denotes the total number of samples
        return self.total_num_of_samples

    def __getitem__(self, index):
        # return batches that are associated with 'index'
        X = self.inputs[index].view(-1) # view -1 ensures right input dimensions for network
        Y = self.labels[index].view(-1)
        return X, Y

    