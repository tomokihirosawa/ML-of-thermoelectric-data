# NN parameters
fcs = [41, 512, 195]

# Parameters for training
params = {
    'batch_size': 575,  #full batch size=size of labels
    'shuffle': False,
    'num_workers': 1
}

# learning rate of the optimizer
lr = 1e-4

# number of epochs
epochs = 5000
