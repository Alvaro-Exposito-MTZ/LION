from LION.classical_algorithms.fdk import fdk
from Sparse2InverseSolver import Sparse2InverseSolver
from LION.models.CNNs.UNets.Unet import UNet
import LION.experiments.ct_experiments as ct_experiments
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import torch.nn as nn
import torch
import pathlib
import torch.utils.data as data_utils
import random


seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set Device
#%%
# % Chose device:
device = torch.device("cuda:1")
torch.cuda.set_device(device)

# Define your data paths
savefolder = pathlib.Path("/store/LION/ea692/LION/LION/trained_models/Sparse2Inverse/Train/SparseAngleLowDoseCTRecon/30sin2000ep/16Angles")
# Creates the folders if they does not exist
savefolder.mkdir(parents=True, exist_ok=True)
final_result_fname = "S2I.pt"
checkpoint_fname = "S2I_check_*.pt"

# Define experiment
experiment = ct_experiments.SparseAngleLowDoseCTRecon()
train_dataset = experiment.get_training_dataset()
indices = torch.arange(30)
train_dataset = data_utils.Subset(train_dataset, indices)

# Data to train
batch_size = 1
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define model
model = UNet()

# Create optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

#Sparse2InverseSolver.
s2i_params = Sparse2InverseSolver.default_parameters()
# Sparse to inverse requires certain user specifications.
s2i_params.sino_split_count = 4
s2i_params.recon_fn = fdk

# Initialize the solver as the other solvers in LION
solver = Sparse2InverseSolver(
    model,
    optimizer,
    loss_fn,
    solver_params=s2i_params,
    geometry=experiment.geometry,
    verbose=True,
    device=device,
)

solver.set_training(dataloader)
solver.set_checkpointing(checkpoint_fname, 100, save_folder=savefolder)

epochs = 100

solver.train(epochs)
solver.save_final_results(final_result_fname, savefolder)
solver.clean_checkpoints()
