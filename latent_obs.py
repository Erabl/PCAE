import numpy as np
import time
import utils
import matplotlib.pyplot as plt
#%matplotlib inline
import torch
import model
import datetime
import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

import warnings
warnings.filterwarnings("ignore")

batch_size = 1
output_folder = "output/" # folder path to save the results
save_results = True # save the results to output_folder
use_GPU = True # use GPU, False to use CPU
latent_size = 128 # bottleneck size of the Autoencoder model

import autoencoder
import ds_model

train_pc = torch.load("data/preprocessed_train_point_cloud.pt", weights_only=False)
val_pc = torch.load("data/preprocessed_val_point_cloud.pt", weights_only=False)
test_pc = torch.load("data/preprocessed_test_point_cloud.pt", weights_only=False)

merged_pc = test_pc + val_pc

test_pc = merged_pc

train_loader = DataLoader(train_pc, shuffle=True, num_workers=0, batch_size=batch_size, pin_memory=True)
test_loader = DataLoader(test_pc, shuffle=True, num_workers=0, batch_size=batch_size, pin_memory=True)
    

# Assuming all models have the same size, get the point size from the first model
point_size = len(train_loader.dataset[0])
print(point_size)

# model = ds_model.PointCloudAutoencoder.load_from_checkpoint(
#     "best_model.ckpt",
#     map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# )
#path = 'model_checkpoint_2025-04-23_11-56-40.pth'
path = "model_checkpoint_2025-04-23_12-42-42.pth"
model = ds_model.PointCloudAutoencoder(
    point_size=1024,       # Must match checkpoint
    latent_dim=72,         # From checkpoint's final_conv layer
    hidden_dim=128,        # Inferred from error messages (was 128, now 256)
    lr=1e-3                # Default, adjust if needed
)
model.load_state_dict(torch.load(path, weights_only = True))

model.eval()

if(use_GPU):
    device = torch.device("mps")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(model)
else:
    device = torch.device("cpu")

model = model.to(device)



# trainer = pl.Trainer(
#     max_epochs=500, 
#     log_every_n_steps=10, 
#     accelerator='mps',  # Specify MPS accelerator
#     devices='auto'      # Use 1 device
# )

#trainer.fit(model, train_loader, test_loader)    

#timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#torch.save(model.state_dict(), f"model_checkpoint_{timestamp}.pth")


utils.evaluate_model(model, [train_loader, test_loader])  

torch.cuda.empty_cache()