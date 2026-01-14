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
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from LION.metrics.haarpsi import HAARPsi

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
savefolder = pathlib.Path("/store/LION/ea692/LION/LION/trained_models/Sparse2Inverse/Train/SparseAngleLowDoseCTRecon")
# Creates the folders if they does not exist
savefolder.mkdir(parents=True, exist_ok=True)
final_result_fname = "S2I.pt"
checkpoint_fname = "S2I_check_*.pt"

# Define experiment
experiment = ct_experiments.SparseAngleLowDoseCTRecon()
train_dataset = experiment.get_training_dataset()
#30 sinograms for the experiment
indices = torch.arange(30)
train_dataset = data_utils.Subset(train_dataset, indices)

# Data to train
batch_size = 1
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Define model. In the original paper used UNet
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

# Test using the training data
savefolder = pathlib.Path("/home/ea692/LION/LION/trained_models/Sparse2Inverse/Test/SparseAngleLowDoseCTRecon/SparseVSNoise/30sin2000ep/64Angles_Haarpsi_and_SSIM")
savefolder.mkdir(parents=True, exist_ok=True)

#Load the trained model of Sparse2Inverse
model_Sparse, _, _ = UNet().load("/store/LION/ea692/LION/LION/trained_models/Sparse2Inverse/Train/SparseAngleLowDoseCTRecon/S2I.json")
model_Sparse.eval()
çsolver_params = Sparse2InverseSolver.default_parameters()
solver_params.sino_split_count = 4
solver_params.recon_fn = fdk
optimizer = Adam(model_Sparse.parameters())
#Not used directly, the solver defines its own loss.
loss_fn = nn.MSELoss()

solver_sparse = Sparse2InverseSolver(
    model_Sparse,
    optimizer,
    loss_fn,
    solver_params=solver_params,
    geometry=experiment.geometry,
    verbose=False,
    device=device,
)

#Normalization in order to ensure a fair comparison of structural and perceptual image quality.
def normalize_01(x,y):
    x = (x - y.min())/ (y.max() - y.min())
    x[x>1]=1
    x[x<0]=0
    return x

#HAARPsi metric
haarpsi = HAARPsi(C=5.0, a=4.9)
haarpsi.eval()

haarpsi_values_sparse = []

#SSIM metric
def my_ssim(x, y):
    x = x.cpu().numpy().squeeze()
    y = y.cpu().numpy().squeeze()
    return ssim(x, y, data_range=x.max() - x.min())

ssim_values_sparse = []

# Fixed visualization window for all images to ensure fair visual comparison.
vmin, vmax = 0, 5

for idx, (sino, target) in enumerate(dataloader):
    sino = sino.to(device)
    with torch.no_grad():
        model_reco_sparse = solver_sparse.reconstruct(sino).detach().cpu()
        target_cpu = target.cpu()

        target_n = normalize_01(target_cpu,target_cpu)
        sparse_n = normalize_01(model_reco_sparse,target_cpu)

        haarspi_sparse,_,_ = haarpsi(target_n, sparse_n)
        ssim_sparse=my_ssim(target_n,sparse_n)
    
    ssim_values_sparse.append(ssim_sparse)

    haarpsi_values_sparse.append(haarspi_sparse.item())

    #Figure the comparison between target and reconstruction. 
    #Raw reconstructions are shown without normalization.
    if idx == 0:
        plt.figure(figsize=(12,4))
        
        plt.subplot(1,2,1)
        plt.title("Target (clean)")
        im0 = plt.imshow(target[0,0].cpu(), cmap="gray")
        plt.axis("off")
        im0.set_clim(vmin, vmax)
        
        plt.subplot(1,2,2)
        plt.title(f"Model raw reconstruction Sparse\nhaarpsi={haarspi_sparse.item():.3f}\nssim={ssim_sparse:.3f}")
        im2 = plt.imshow(model_reco_sparse[0,0], cmap="gray")
        plt.axis("off")
        im2.set_clim(vmin, vmax)
        
        plt.tight_layout()
        plt.savefig(savefolder / "Reconstruction_Sparse2Inverse_SparseAngleLowDoseCTRecon_Haarspi_SSIM.png", dpi=150)
        plt.close()

haarpsi_mean_sparse = np.mean(haarpsi_values_sparse)
haarpsi_std_sparse = np.std(haarpsi_values_sparse)

ssim_mean_sparse = np.mean(ssim_values_sparse)
ssim_std_sparse = np.std(ssim_values_sparse)

#Plot the metrics obtained throw all the sinograms tested
print(f"haarpsi mean Sparse SparseAngleLowDoseCTRecon 30sin2000ep 64 angles: {haarpsi_mean_sparse:.4f}, haarpsi std Sparse: {haarpsi_std_sparse:.4f}")
print(f"ssim mean Sparse SparseAngleLowDoseCTRecon 30sin2000ep 64 angles: {ssim_mean_sparse:.4f}, ssim std Sparse: {ssim_std_sparse:.4f}")
