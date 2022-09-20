# NN parameters
fcs = [50, 512, 219]

# Parameters for training
params = {
    'batch_size': 301,  #full batch size=size of labels
    'shuffle': False,
    'num_workers': 1
}

# learning rate of the optimizer
lr = 1e-3

# number of epochs
epochs = 3000
