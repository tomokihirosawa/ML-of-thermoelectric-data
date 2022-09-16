import numpy as np

# own modules
import hyperparameters.hyperparameters_exp_focus as hyperparameters
import NN.NN_focus as NN
import dataloader.dataloader_exp_focus as dataloader
import utils
import matplotlib.pyplot as plt

# load PyTorch modules
import torch
from torch.utils import data # to define a dataset class
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

# set random seed for reproducibility 
seed = 200
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.RandomState(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(device, epoch, params, model, optimizer,Tot_epoch,loss_array,x,y,alpha,cond,Delta):
    # define data set and associated loader
    loaded_set = dataloader.Dataset(x,y)
    loader = data.DataLoader(loaded_set, **params)

    # set model to training mode
    model.train()
    Wc=10.0
    sample=1
    size=41
    epoch_loss = 0.0
    # dict that consists of two lists to store inputs and labels
    results = {'inputs':[], 'preds':[], 'labels':[]}

    for inputs, labels in loader:
        inputs = inputs[:,0].to(device)
        labels = labels[:,0].to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # computed values of L11 and L12 from the NN
        preds = model(inputs[:size],Wc,alpha,cond,Delta)
        #the penalty term of negative spectral conductivity
        b=1.0
        preds[-1]=preds[-1]*b
        
        #evaluate loss
        loss=criterion(preds, labels)   
        
        # backpropagation and parameter updates
        loss.backward()
        optimizer.step()

        # running loss update
        epoch_loss += loss*inputs.size(0)

        with torch.no_grad():
            # push data to dict
            results['inputs'].append(inputs.detach().cpu().numpy())
            results['preds'].append(preds.detach().cpu().numpy())
            results['labels'].append(labels.detach().cpu().numpy())
            
    epoch_loss = epoch_loss/len(loaded_set)
    loss_array[epoch]=epoch_loss.detach()
    
    #generate output files after all epochs 
    if (epoch+1==hyperparameters.epochs):
        print('epoch {} Training Loss: {:.4f}'.format(epoch, epoch_loss))
        
        #output of NN
        O=model.fc_layers(inputs[:size]).detach().cpu().numpy()
        
        ##output the spectral conductivity
        # averaging 10 consequtive elements of the output vector to compute the spectral conductivity
        Q=O[:size*4+10]
        Sigma=np.zeros(size*4+1)  
        for j in range(Sigma.shape[0]):
            Sigma[j]=(Q[j]+Q[j+1]+Q[j+2]+Q[j+3]+Q[j+4]+Q[j+5]+Q[j+6]+Q[j+7]+Q[j+8]+Q[j+9])/10.0
        res=np.zeros((2,size*4+1))
        res[0]=np.linspace(-1.0, 1.0, size*4+1)*Wc*2
        res[1]=Sigma+cond.detach().cpu().numpy()
        
        ##output the chemical potential
        chemQ=O[size*4+10:]
        chem=alpha.detach().cpu().numpy()+np.arctan(chemQ)/(np.pi/2.0)*Wc*Delta     
        
        ##output L11 and L12 computed from the spectral conductivity
        res2=np.zeros((sample,6,size))
        #res2[0]: temperature, res2[1]: true L11, res2[2]: true L12
        #res2[3]: computed L11, res2[4]: computed L12, res2[5]: chemical potential
        
        for i in range(sample):
            res2[i,0]=np.linspace(0,1, size+1)[1:]
            res2[i,1]=labels.detach().cpu().numpy()[2*i*size:(2*i+1)*size]
            res2[i,2]=labels.detach().cpu().numpy()[(2*i+1)*size:(2*i+2)*size]
            res2[i,3]=preds.detach().cpu().numpy()[2*i*size:(2*i+1)*size]
            res2[i,4]=preds.detach().cpu().numpy()[(2*i+1)*size:(2*i+2)*size]
            res2[i,5]=chem

        res.tofile('./data/exp_focus_cond_'+str(int(x*1000))+'_'+str(int(y*1000))+'_'+str(epoch+1)+'_delta_'+str(int(Delta*1000))+'.dat')
        res2.tofile('./data/exp_focus_L11_L12_'+str(int(x*1000))+'_'+str(int(y*1000))+'_'+str(epoch+1)+'_delta_'+str(int(Delta*1000))+'.dat')


if __name__ == '__main__':
    # choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    # initialize model (and put it to device)
    model = NN.FCN(hyperparameters.fcs).to(device)

    # define loss function
    criterion = nn.MSELoss()

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.lr)
    
    # loss function
    loss_array=np.zeros(hyperparameters.epochs)
    
    #chemical potential and spectral conductivity are imported
    epoch=5000
    sample=7
    data1 = np.fromfile('./data/exp_all_cond_'+str(epoch)+'_'+str(sample)+'.dat', dtype=np.float64, count=5000)
    data2 = np.fromfile('./data/exp_all_L11_L12_'+str(epoch)+'_'+str(sample)+'.dat', dtype=np.float64, count=7*6*1000)
    nsite=len(data1)//2
    nsite2=len(data2)//(6*7)
    #index for a given doping concentration
    xarray=np.array([0.0,0.0,0.001,0.002,0.01,0.02,0.05])
    yarray=np.array([0.0,0.05,0.0,0.0,0.0,0.0,0.0])
    ind=np.where((xarray==hyperparameters.x) & (yarray==hyperparameters.y))[0]
    #load spectral conductivity and chemical potential for a given doping concentration
    data2=data2.reshape(7,6,nsite2)
    data1=data1.reshape(2,nsite)
    cond=torch.from_numpy(data1[1].astype(np.float32)).clone()
    cond=cond.to(device)
    alpha=torch.from_numpy(data2[ind,5].astype(np.float32)).clone()
    alpha=alpha[0].to(device)
    ## parameter for small correction in chemical potential
    Delta=0.0
   
    lr=hyperparameters.lr
    lmin=lr/5.0
    for epoch in range(hyperparameters.epochs):
        if epoch>1:
            if loss_array[epoch-2]<loss_array[epoch-1]:
                for g in optimizer.param_groups:                   
                    if loss_array[epoch-1]<lmin:
                        lmin=loss_array[epoch-1]
                    lr=max(lr*0.8,lmin)
                    g['lr'] = lr
        if epoch%100==0:
            with utils.Timer('epoch {} timing: '.format(epoch)):
                train(device, epoch, hyperparameters.params, model, optimizer,hyperparameters.epochs,loss_array,hyperparameters.x,hyperparameters.y,alpha,cond,Delta)
            print('--'*10)
        else:
            train(device, epoch, hyperparameters.params, model, optimizer,hyperparameters.epochs,loss_array,hyperparameters.x,hyperparameters.y,alpha,cond,Delta)
    loss_array.tofile('./data/exp_focus_loss_'+str(int(hyperparameters.x*1000))\
                +'_'+str(int(hyperparameters.y*1000))+'_'+str(epoch+1)+'_delta_'+str(int(Delta*1000))+'.dat')