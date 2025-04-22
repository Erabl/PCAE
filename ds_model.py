import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import torch_geometric.nn as gnn

class PointCloudAE(nn.Module):
    def __init__(self, point_size=1024, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.point_size = point_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Encoder layers
        self.gcn_layers = nn.ModuleList([
            gnn.GCNConv(2, hidden_dim // 4),
            gnn.GCNConv(hidden_dim // 4, hidden_dim // 2),
            gnn.GCNConv(hidden_dim // 2, hidden_dim)
        ])
        
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(2)
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, latent_dim, 1),
            nn.BatchNorm1d(latent_dim)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, point_size * 2)
        )

    def encoder(self, x, edge_index):
        x = x.squeeze(0).transpose(0, 1)
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index).relu()
        
        x = x.T.unsqueeze(0)
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        x = self.final_conv(x)
        return torch.max(x, 2, keepdim=True)[0].view(-1, self.latent_dim)
    
    def decode(self, x):
        return self.decoder(x).view(-1, self.point_size, 2)
    
    def forward(self, x, edge_index):
        return self.decode(self.encoder(x, edge_index))

class PointCloudAutoencoder(pl.LightningModule):
    def __init__(self, 
                 point_size=1024,
                 latent_dim=128,
                 hidden_dim=256,
                 lr=1e-3,
                 save_results=False,
                 output_folder='output/'):
        super().__init__()
        self.save_hyperparameters()
        self.net = PointCloudAE(point_size, latent_dim, hidden_dim)
        self.save_results = save_results
        self.output_folder = output_folder
        self.lr = lr
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def update_architecture(self, latent_dim=None, hidden_dim=None):
        """Dynamically update model architecture while preserving weights where possible"""
        latent_dim = latent_dim or self.hparams.latent_dim
        hidden_dim = hidden_dim or self.hparams.hidden_dim
        
        # Create new network
        new_net = PointCloudAE(
            point_size=self.hparams.point_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        # Transfer compatible weights
        for (name, old_param), new_param in zip(
            self.net.named_parameters(), 
            new_net.parameters()
        ):
            if old_param.shape == new_param.shape:
                new_param.data.copy_(old_param.data)
        
        self.net = new_net
        self.hparams.latent_dim = latent_dim
        self.hparams.hidden_dim = hidden_dim

    def forward(self, data, edge_index):
        return self.net(data.permute(0, 2, 1), edge_index)

    def training_step(self, batch, batch_idx):
        output = self(batch.pos, batch.edge_index)
        loss = chamfer_distance(batch.pos, output)
        self.log("train_loss", loss, 
                batch_size=batch.batch_size,
                prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.pos, batch.edge_index)
        loss = chamfer_distance(batch.pos, output)
        self.log("val_loss", loss,
                batch_size=batch.batch_size,
                prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

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
