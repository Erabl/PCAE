import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import optim
import matplotlib.pyplot as plt
import datetime
import utils
import uuid
import sys

import torch_geometric.nn as gnn
'''
PointNet AutoEncoder
Learning Representations and Generative Models For 3D Point Clouds
https://arxiv.org/abs/1707.02392

adapted from: https://github.com/cihanongun/Point-Cloud-Autoencoder/blob/master/README.md

'''

class PointCloudAE(nn.Module):
    def __init__(self, point_size=1024, LATENT_DIM=128, HIDDEN_DIM=256):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = LATENT_DIM
        self.hidden_dim = HIDDEN_DIM
        self.point_size = point_size
        
        step_dim1 = self.hidden_dim // 4
        step_dim2 = self.hidden_dim // 2

        self.gcn1 = gnn.GCNConv(2, step_dim1)
        self.gcn2 = gnn.GCNConv(step_dim1, step_dim2)
        self.gcn3 = gnn.GCNConv(step_dim2, self.hidden_dim)
        
        #self.conv1 = torch.nn.Conv1d(2, 256, 1)
        self.conv1 = torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.hidden_dim, self.hidden_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.hidden_dim, self.latent_size, 1)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size, self.hidden_dim)
        self.dec2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dec3 = nn.Linear(self.hidden_dim, point_size*2)
        
        self.dropout = nn.Dropout(0.05)

    def encoder(self, x, edge_index): 
        x = x.squeeze(0).transpose(0, 1)  # [1, 1024, 2] → [1024, 2]
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = self.gcn3(x, edge_index).relu()

        x = x.T.unsqueeze(0)  # → [1, 256, 1024]

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        #x = self.dropout(x)
        x = F.relu(self.dec2(x))
        #x = self.dropout(x)
        x = self.dec3(x)
        x = x.view(-1, self.point_size, 2)
        return x
    
    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.decoder(x)
        return x
    
def chamfer_distance(x, y, batch_reduction="mean", point_reduction="mean"):
    """
    Compute the chamfer distance between two point clouds x and y
    
    Parameters:
    -----------
    x: torch.Tensor
        First point cloud, shape (batch_size, num_points_x, dim)
    y: torch.Tensor
        Second point cloud, shape (batch_size, num_points_y, dim)
    batch_reduction: str
        Reduction operation to apply across the batch dimension: 'mean', 'sum' or None
    point_reduction: str
        Reduction operation to apply across the point dimension: 'mean', 'sum' or None
        
    Returns:
    --------
    dist_chamfer: torch.Tensor
        Chamfer distance between the point clouds
    """
    # Check input dimensions
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError("Input point clouds must be 3-dimensional tensors")
    
    # Extract batch size and number of points
    #batch_size, num_points_x, dim = x.shape
    _, num_points_y, _ = y.shape
    
    # Reshape to compute pairwise distances efficiently
    x_expanded = x.unsqueeze(2)  # [batch_size, num_points_x, 1, dim]
    y_expanded = y.unsqueeze(1)  # [batch_size, 1, num_points_y, dim]
    
    # Compute squared distances between each pair of points
    # |x-y|^2 = |x|^2 + |y|^2 - 2*x·y
    dist = ((x_expanded - y_expanded) ** 2).sum(dim=3)  # [batch_size, num_points_x, num_points_y]
    
    # For each point in x, find the distance to the closest point in y
    # Use values, not tuples from min()
    x_to_y = torch.min(dist, dim=2).values  # [batch_size, num_points_x]
    
    # For each point in y, find the distance to the closest point in x
    y_to_x = torch.min(dist, dim=1).values  # [batch_size, num_points_y]
    
    # Apply point reduction
    if point_reduction == "mean":
        x_to_y = x_to_y.mean(dim=1)  # [batch_size]
        y_to_x = y_to_x.mean(dim=1)  # [batch_size]
    elif point_reduction == "sum":
        x_to_y = x_to_y.sum(dim=1)  # [batch_size]
        y_to_x = y_to_x.sum(dim=1)  # [batch_size]
    elif point_reduction is None:
        pass
    else:
        raise ValueError(f"Invalid point_reduction: {point_reduction}")
    
    # Combine the two directional distances
    chamfer_dist = x_to_y + y_to_x  # [batch_size] or [batch_size, num_points]
    
    # Apply batch reduction
    if batch_reduction == "mean":
        chamfer_dist = chamfer_dist.mean()
    elif batch_reduction == "sum":
        chamfer_dist = chamfer_dist.sum()
    elif batch_reduction is None:
        pass
    else:
        raise ValueError(f"Invalid batch_reduction: {batch_reduction}")
    
    return chamfer_dist

class PointCloudAutoencoder(pl.LightningModule):
    def __init__(self, save_results=False, output_folder=None):
        super().__init__()
        self.net = PointCloudAE()
        self.save_results = save_results
        self.output_folder = output_folder
        self.train_loss_list = []
        self.test_loss_list = []
        self.to_print = False
        self.output_folder = 'output/'
        if save_results and output_folder:
            utils.clear_folder(self.output_folder)
        self.utils = utils if save_results else None

    def forward(self, data, edge_index):
        #print(x.shape)
        num_nodes = data.shape[1]  # Should be 1024
        valid_mask = (edge_index < num_nodes).all(dim=0)  # Shape: [num_edges]
        edge_index = edge_index[:, valid_mask]  # Keep only valid edges
        return self.net(data.permute(0, 2, 1), edge_index)

    def encode(self, data):
        #print(x.shape)
        num_nodes = data.pos.shape[1]  # Should be 1024
        valid_mask = (data.edge_index < num_nodes).all(dim=0)  # Shape: [num_edges]
        edge_index = data.edge_index[:, valid_mask]  # Keep only valid edges
        return self.net.encoder(data.pos.permute(0, 2, 1), edge_index)

    def training_step(self, batch, batch_idx):
        output = self.forward(batch.pos, batch.edge_index)
        loss = chamfer_distance(batch.pos, output)
        self.log("train_loss", loss, batch_size=batch.batch_size, prog_bar=True, on_epoch=True)
        if self.current_epoch % 50 == 0:
            self.to_print = True
            utils.plot_pointcloud_comparison(batch.pos.cpu(), output.cpu(), save=True, name=f"{self.output_folder}training_plot_{self.current_epoch}")
        else:
            self.to_print=False
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch.pos, batch.edge_index)
        loss = chamfer_distance(batch.pos, output)
        self.log("val_loss", loss, batch_size=batch.batch_size, prog_bar=True, on_epoch=True)
        if self.to_print is True:
            utils.plot_pointcloud_comparison(batch.pos.cpu(), output.cpu(), save=True, name=f"{self.output_folder}validation_plot_{self.current_epoch}")
        return {"loss": loss, "input": batch.pos.cpu(), "output": output.cpu()}

    # def configure_optimizers(self):
    #     return optim.Adam(self.net.parameters(), lr=0.001) #5e-4

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }